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
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_validation import model_artifacts
from run_state import sha256

warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, Superimposer
try:
    from .structure_checks import chain_ca, matching_ca
except ImportError:
    from structure_checks import chain_ca, matching_ca
try:
    from .result_io import publish_columns, write_json
except ImportError:
    from result_io import publish_columns, write_json

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")
_parser = PDBParser(QUIET=True)


def template_ca_atoms(template_pdb):
    """CA atoms of the fold template's first chain (parsed once per fold)."""
    t = _parser.get_structure("t", template_pdb)
    return chain_ca(list(t[0])[0])


def rmsd_to_template(design_pdb, templ):
    """CA RMSD of the designed binder (chain B) to the fold template, superposed.

    ``templ`` is the template's pre-extracted CA atom list (the template is the
    same for every design in a fold, so it is parsed once by the caller rather
    than re-parsed per design). Superposition is stateless, so this is
    bit-identical to parsing the template inline.
    """
    d = _parser.get_structure("d", design_pdb)
    binder = chain_ca(d[0]["B"])
    matching_ca(templ, binder)
    sup = Superimposer()
    sup.set_atoms(templ, binder)
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
        rmsds, model_scores = [], {}
        for name in df['name']:
            records = {}
            for model, pdb, _ in model_artifacts(Path(RUNS)/fold, name):
                records[model] = dict(rmsd=rmsd_to_template(str(pdb), templ), pdb_sha256=sha256(pdb))
            rmsds.append(next(iter(records.values()))['rmsd'])
            model_scores[name] = records
        write_json(Path(RUNS)/fold/'rmsd.models.json', dict(template_sha256=sha256(tmpl), candidates=model_scores))
        df["rmsd"] = rmsds
        publish_columns(csvf, df, ['rmsd'])
        print(f"{fold}: wrote rmsd for {len(rmsds)} designs -> {csvf}")


if __name__ == "__main__":
    main()
