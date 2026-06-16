"""Compute ipSAE for each design from the saved AF2 PAE matrix, offline (no GPU,
no re-prediction), and write an ``ipsae`` column into ``runs/<fold>/results.csv``.

FoldCraft saves the full predicted-aligned-error matrix per design
(``aux['all']['pae']``, in Angstroms), which is all ipSAE needs. The binder
length per fold comes from ``config.tsv`` (the target length is the remainder of
the PAE dimension). See ipsae.py for the algorithm + reference.

Usage:  python baseline/add_ipsae.py [runs_dir] [pae_cutoff]   (default: baseline/runs, 10)
Fails loudly if a design pickle referenced by results.csv is missing.
"""
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from ipsae import ipsae  # baseline/ is on sys.path[0] when run as a script

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
PAE_CUTOFF = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0


def binder_lens_from_config():
    lens = {}
    with open(os.path.join(HERE, "config.tsv")) as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                lens[parts[0]] = int(parts[3])
    return lens


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"runs dir not found: {RUNS}")
    lens = binder_lens_from_config()
    for fold in sorted(os.listdir(RUNS)):
        csvf = os.path.join(RUNS, fold, "results.csv")
        if not os.path.exists(csvf):
            continue
        if fold not in lens:
            sys.exit(f"no binder_len for fold '{fold}' in config.tsv")
        blen = lens[fold]
        df = pd.read_csv(csvf)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        vals = []
        for name in df["name"]:
            pk = os.path.join(RUNS, fold, "designs", f"{name}.pickle")
            if not os.path.exists(pk):
                sys.exit(f"design pickle missing, cannot compute ipsae: {pk}")
            with open(pk, "rb") as fh:
                pae = np.asarray(pickle.load(fh)["pae"])[0]
            tlen = pae.shape[0] - blen
            vals.append(round(ipsae(pae, tlen, blen, PAE_CUTOFF), 4))
        df["ipsae"] = vals
        df.to_csv(csvf, index=False)
        print(f"{fold}: ipsae for {len(vals)} designs (pae_cutoff={PAE_CUTOFF}) -> {csvf}")


if __name__ == "__main__":
    main()
