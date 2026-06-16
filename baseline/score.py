"""Score a FoldCraft baseline run: per-criterion + combined success rate.

Reads ``runs/<fold>/results.csv``, which must include an ``rmsd`` column (RMSD
of the designed binder to its fold template). Populate that column with
``add_rmsd.py`` (which needs the raw design PDBs); this script then works from
the committed CSVs alone and from any working directory.

Usage:  python baseline/score.py [runs_dir]        (default: baseline/runs)
"""
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "runs")

CRIT = [
    ("plddt>.8", lambda r: r["plddt"] > 0.8),
    ("iptm>.5",  lambda r: r["iptm"] > 0.5),
    ("ipae<.35", lambda r: r["ipae"] < 0.35),
    ("rmsd<3.5", lambda r: r["rmsd"] < 3.5),
]


def wilson(k, n):
    z = 1.96
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (round(100 * ph, 1), round(100 * max(0, c - h), 1), round(100 * min(1, c + h), 1))


def main():
    if not os.path.isdir(RUNS):
        sys.exit(f"runs dir not found: {RUNS}")
    print(f"{'fold':9} {'n':>3} " + " ".join(f"{k:>8}" for k, _ in CRIT)
          + f" {'ALL':>4} {'success% [95%CI]':>20}")
    scored = 0
    for fold in sorted(os.listdir(RUNS)):
        csvf = os.path.join(RUNS, fold, "results.csv")
        if not os.path.exists(csvf):
            continue
        df = pd.read_csv(csvf)
        if "rmsd" not in df.columns:
            sys.exit(
                f"{csvf} has no 'rmsd' column — run "
                f"`python baseline/add_rmsd.py {RUNS}` first (needs the raw "
                f"design PDBs) to populate RMSD-to-template."
            )
        if df["rmsd"].isna().any():
            sys.exit(f"{csvf}: {int(df['rmsd'].isna().sum())} rows are missing an rmsd value.")
        recs = df.to_dict("records")
        n = len(recs)
        counts = [sum(1 for r in recs if fn(r)) for _, fn in CRIT]
        npass = sum(1 for r in recs if all(fn(r) for _, fn in CRIT))
        sr, lo, hi = wilson(npass, n)
        print(f"{fold:9} {n:>3} " + " ".join(f"{c:>8}" for c in counts)
              + f" {npass:>4} {f'{sr}% [{lo}-{hi}]':>20}")
        scored += 1
    if scored == 0:
        sys.exit(f"no scorable runs/<fold>/results.csv found under {RUNS}")


if __name__ == "__main__":
    main()
