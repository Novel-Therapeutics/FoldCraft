"""Populate an ``rmsd`` column (designed binder vs its fold template) in each
``runs/<fold>/results.csv``, computed from the raw design PDBs.

This is the one scoring step that needs the raw designs (large, not committed),
so run it where the designs live. ``score.py`` then reads the ``rmsd`` column
and works from the committed CSVs alone.

The fold template is read from ``<runs>/<fold>/template.pdb`` -- the binder
template that run actually conditioned on, recorded by the runner. (A global
``baseline/templates/<fold>.pdb`` was wrong for runs built from other templates,
e.g. the author-config ``baseline/repro`` whose folds reuse the same names with
different templates.) Fails loudly if the template or a referenced design PDB is
missing.

Usage:  python baseline/add_rmsd.py [runs_dir]     (default: baseline/runs)
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, Superimposer
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
_parser = PDBParser(QUIET=True)


def template_ca_atoms(template_pdb):
    """CA atoms of the fold template's first chain (parsed once per fold)."""
    t = _parser.get_structure("t", template_pdb)
    return [r["CA"] for r in list(t[0])[0] if "CA" in r]


def rmsd_to_template(design_pdb, templ):
    """CA RMSD of the designed binder (chain B) to the fold template, superposed.

    ``templ`` is the template's pre-extracted CA atom list (the template is the
    same for every design in a fold, so it is parsed once by the caller rather
    than re-parsed per design). Superposition is stateless, so this is
    bit-identical to parsing the template inline.
    """
    d = _parser.get_structure("d", design_pdb)
    binder = [r["CA"] for r in d[0]["B"] if "CA" in r]
    n = min(len(binder), len(templ))
    if n < 3:
        raise ValueError(f"{design_pdb}: fewer than 3 comparable CA atoms")
    sup = Superimposer()
    sup.set_atoms(templ[:n], binder[:n])
    return round(sup.rms, 2)


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"runs dir not found: {RUNS}")
    for fold in sorted(os.listdir(RUNS)):
        csvf = os.path.join(RUNS, fold, "results.csv")
        if not os.path.exists(csvf):
            continue
        tmpl = os.path.join(RUNS, fold, "template.pdb")
        if not os.path.exists(tmpl):
            sys.exit(f"missing per-fold template: {tmpl} -- the runner must record "
                     f"the binder template it used (see run_repro.sh / run_campaign.sh)")
        df = pd.read_csv(csvf)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop stale index col
        templ = template_ca_atoms(tmpl)  # parse the template once per fold
        rmsds = []
        for name in df["name"]:
            dp = os.path.join(RUNS, fold, "designs", f"{name}.pdb")
            if not os.path.exists(dp):
                sys.exit(f"design PDB missing, cannot compute rmsd: {dp}")
            rmsds.append(rmsd_to_template(dp, templ))
        df["rmsd"] = rmsds
        df.to_csv(csvf, index=False)
        print(f"{fold}: wrote rmsd for {len(rmsds)} designs -> {csvf}")


if __name__ == "__main__":
    main()
