"""Compute ipSAE for each design from the saved AF2 PAE matrix, offline (no GPU,
no re-prediction), and write an ``ipsae`` column into ``<runs>/<fold>/results.csv``.

FoldCraft saves the full predicted-aligned-error matrix per design
(``aux['all']['pae']``, in Angstroms), which is all ipSAE needs. The target/binder
split is read from each design PDB's own chains -- chain A = target, chain B =
binder, in the same order as the PAE matrix -- so it is correct regardless of which
template or config produced the run. (Reading binder lengths from a global
config.tsv was wrong for runs built from different templates, e.g. the
author-config ``baseline/repro`` whose folds reuse the same names with different
binder lengths.)

Usage:  python baseline/add_ipsae.py [runs_dir] [pae_cutoff]   (default: baseline/runs, 10)
Fails loudly if a design pickle/PDB referenced by results.csv is missing or its
chain lengths disagree with the PAE matrix.
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

from ipsae import ipsae  # baseline/ is on sys.path[0] when run as a script

HERE = os.path.dirname(os.path.abspath(__file__))
_parser = PDBParser(QUIET=True)


def complex_chain_lens(pdb):
    """(target_len, binder_len) = (#residues in chain A, #residues in chain B).

    FoldCraft saves the design as chain A = target, chain B = binder, in the same
    order as the PAE matrix, so these lengths give the exact split ipSAE needs.
    Raises if either chain is absent.
    """
    model = _parser.get_structure("c", pdb)[0]
    for cid in ("A", "B"):
        if cid not in model:
            raise ValueError(f"{pdb}: missing chain {cid} (expected target=A, binder=B)")
    n = {cid: sum(1 for r in model[cid] if r.id[0] == " ") for cid in ("A", "B")}
    return n["A"], n["B"]


def main():
    RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
    PAE_CUTOFF = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    if not os.path.isdir(RUNS):
        sys.exit(f"runs dir not found: {RUNS}")
    for fold in sorted(os.listdir(RUNS)):
        csvf = os.path.join(RUNS, fold, "results.csv")
        if not os.path.exists(csvf):
            continue
        df = pd.read_csv(csvf)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        vals = []
        for name in df["name"]:
            base = os.path.join(RUNS, fold, "designs", name)
            pdb, pk = base + ".pdb", base + ".pickle"
            if not os.path.exists(pk):
                sys.exit(f"design pickle missing, cannot compute ipsae: {pk}")
            if not os.path.exists(pdb):
                sys.exit(f"design PDB missing, cannot determine chain split: {pdb}")
            tlen, blen = complex_chain_lens(pdb)
            with open(pk, "rb") as fh:
                pae = np.asarray(pickle.load(fh)["pae"])[0]
            if tlen + blen != pae.shape[0]:
                sys.exit(f"{pdb}: chain lengths {tlen}+{blen} != PAE dim "
                         f"{pae.shape[0]} -- target/binder split is inconsistent")
            vals.append(round(ipsae(pae, tlen, blen, PAE_CUTOFF), 4))
        df["ipsae"] = vals
        df.to_csv(csvf, index=False)
        print(f"{fold}: ipsae for {len(vals)} designs (pae_cutoff={PAE_CUTOFF}) -> {csvf}")


if __name__ == "__main__":
    main()
