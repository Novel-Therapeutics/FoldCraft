"""Score a FoldCraft baseline run: per-criterion + combined success rate.

Reads ``runs/<fold>/results.csv``. The `rmsd` column is required (part of the
success gate). The `ipsae` column is *reported* when present (it is NOT a
success gate) — populate it with add_ipsae.py. Success = pLDDT>0.8 ∧ ipTM>0.5 ∧
iPAE<0.35 ∧ RMSD<3.5. Works from the committed CSVs and from any working
directory.

Usage:  python baseline/score.py [runs_dir]        (default: baseline/runs)
"""
import os
import sys

import pandas as pd
import numpy as np
try:
    from .gates import af2_mask, validation_status
except ImportError:
    from gates import af2_mask, validation_status

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")

# success gate (frozen): all four must hold.
GATE = [
    ("pLDDT>.8", lambda r: r["plddt"] > 0.8),
    ("ipTM>.5",  lambda r: r["iptm"] > 0.5),
    ("iPAE<.35", lambda r: r["ipae"] < 0.35),
    ("RMSD<3.5", lambda r: r["rmsd"] < 3.5),
]
# reported, NOT a success gate; shown between iPAE and RMSD when the column exists.
IPSAE = ("ipSAE>.3", lambda r: r["ipsae"] > 0.3)


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"runs dir not found: {RUNS}")
    folds = []
    for fold in sorted(os.listdir(RUNS)):
        csvf = os.path.join(RUNS, fold, "results.csv")
        if os.path.exists(csvf):
            folds.append((fold, pd.read_csv(csvf), csvf))
    if not folds:
        sys.exit(f"no scorable runs/<fold>/results.csv found under {RUNS}")

    for fold, df, csvf in folds:
        if "rmsd" not in df.columns:
            sys.exit(f"{csvf} has no 'rmsd' column — run "
                     f"`python baseline/add_rmsd.py {RUNS}` first (needs the raw designs).")
        if df["rmsd"].isna().any():
            sys.exit(f"{csvf}: {int(df['rmsd'].isna().sum())} rows are missing an rmsd value.")

    has = ["ipsae" in df.columns for _, df, _ in folds]
    if any(has) and not all(has):
        missing = [fold for (fold, _, _), h in zip(folds, has) if not h]
        sys.exit(f"inconsistent: some results.csv have an 'ipsae' column and some don't "
                 f"({missing}) — run `python baseline/add_ipsae.py {RUNS}` on all folds.")
    for fold, df, csvf in folds:
        required = ['plddt','iptm','ipae','rmsd']
        if df.empty or any(c not in df for c in required):
            sys.exit(f'{csvf}: nonempty data and all acceptance metrics are required')
        values = df[required].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            sys.exit(f'{csvf}: unknown/nonfinite acceptance metrics; cannot report an exact success rate')
    print('Descriptive candidate counts only; siblings are correlated, so no binomial confidence interval is reported.')
    show_ipsae = all(has)

    disp = GATE[:3] + ([IPSAE] if show_ipsae else []) + GATE[3:]  # report column order
    print(f"{'fold':9} {'n':>3} " + " ".join(f"{k:>8}" for k, _ in disp)
          + f" {'ALL':>4} {'success%':>10}")
    for fold, df, csvf in folds:
        if show_ipsae and df["ipsae"].isna().any():
            sys.exit(f"{csvf}: rows are missing an ipsae value.")
        recs = df.to_dict("records")
        n = len(recs)
        counts = [sum(1 for r in recs if fn(r)) for _, fn in disp]
        if validation_status(df).isna().any():
            sys.exit(f'{csvf}: unknown ensemble decisions; cannot report an exact success rate')
        npass = int((af2_mask(df) & (df['rmsd'] < 3.5)).sum())
        sr = round(100*npass/n, 1)
        print(f"{fold:9} {n:>3} " + " ".join(f"{c:>8}" for c in counts)
              + f" {npass:>4} {str(sr)+'%':>10}")


if __name__ == "__main__":
    main()
