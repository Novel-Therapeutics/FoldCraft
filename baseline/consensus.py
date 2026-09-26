"""Compute the FoldCraft-vs-BoltzProt consensus table from the oracle-scored
results.csv files.

Consensus gate = a design passes BOTH an AF2 leg AND a Boltz-2 leg (the two are
from different model families, so requiring both removes single-model/self bias):
  AF2 leg     : plddt>0.8 & iptm>0.5 & ipae<0.35
                (FoldCraft: design-time plddt/iptm/ipae; BoltzProt: af2_* columns)
  Boltz-2 leg : boltz2_iptm>0.5   (boltz2_ipae is in A, reported for context)
ESMFold rmsd (binder monomer vs design chain B) reports fold fidelity.

For FoldCraft, Boltz-2 was run on every AF2-passer, so the consensus count is
exact. For BoltzProt, Boltz-2 ran on a random sample; rates over that sample are
extrapolations. Usage: python baseline/consensus.py
"""
import glob
import os
from math import sqrt

import pandas as pd


def wilson(k, n):
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (centre - half), 100 * (centre + half))


HERE = os.path.dirname(os.path.abspath(__file__))


def af2_mask(d):
    p, i, e = (("plddt", "iptm", "ipae") if "iptm" in d.columns
               else ("af2_plddt", "af2_iptm", "af2_ipae"))
    return (d[p] > 0.8) & (d[i] > 0.5) & (d[e] < 0.35)


def main():
    print("=" * 80)
    print("FoldCraft (fold-conditioned) -- n/fold; Boltz-2 ran on every AF2-passer")
    print("  consensus = AF2(plddt>.8 iptm>.5 ipae<.35) AND Boltz-2(boltz2_iptm>.5)")
    print("=" * 80)
    print(f"{'fold':9s}{'AF2 pass':>11s}{'Boltz2|AF2':>12s}"
          f"{'CONSENSUS':>11s}{'95% CI':>14s}{'ESM rmsd':>10s}")
    tot_n = tot_af2 = tot_cons = 0
    for f in sorted(glob.glob(os.path.join(HERE, "repro", "*", "results.csv"))):
        d = pd.read_csv(f)
        fold = os.path.basename(os.path.dirname(f))
        n = len(d)
        af2 = af2_mask(d)
        bz = d["boltz2_iptm"] > 0.5
        cons = af2 & bz
        na, nc = int(af2.sum()), int(cons.sum())
        lo, hi = wilson(nc, n)
        esm = d[cons]["esmfold_rmsd"].median() if nc else float("nan")
        print(f"{fold:9s}{na:6d}/{n:<4d}{int((bz & af2).sum()):8d}/{na:<3d}"
              f"{nc:6d}={100*nc/n:4.1f}%{f'[{lo:.1f}-{hi:.1f}]':>14s}{esm:8.2f} A")
        tot_n += n; tot_af2 += na; tot_cons += nc
    lo, hi = wilson(tot_cons, tot_n)
    print(f"{'POOLED':9s}{tot_af2:6d}/{tot_n:<4d}{'':>12s}"
          f"{tot_cons:6d}={100*tot_cons/tot_n:4.1f}%{f'[{lo:.1f}-{hi:.1f}]':>14s}")

    print("\n" + "=" * 80)
    print("BoltzProt-1 (unconstrained) -- AF2 on all 200; Boltz-2 on a random sample")
    print("=" * 80)
    d = pd.read_csv(os.path.join(HERE, "boltzprot", "results.csv"))
    af2 = af2_mask(d)
    print(f"  AF2-pass (independent judge):  {int(af2.sum())}/{len(d)} = "
          f"{100*af2.mean():.1f}%")
    sb = d[d["boltz2_iptm"].notna()].copy()
    bz = sb["boltz2_iptm"] > 0.5
    sa = af2_mask(sb)
    print(f"  Boltz-2 sample n={len(sb)}:")
    print(f"    Boltz-2 ipTM>.5 (self-family) : {int(bz.sum()):3d} = {100*bz.mean():.0f}%")
    print(f"    AF2-pass (independent)        : {int(sa.sum()):3d} = {100*sa.mean():.0f}%")
    print(f"    CONSENSUS (both)              : {int((bz & sa).sum()):3d} = "
          f"{100*(bz & sa).mean():.0f}%")
    print(f"  median boltz2_iptm {sb['boltz2_iptm'].median():.2f} vs af2_iptm "
          f"{sb['af2_iptm'].median():.2f}  (self-bias gap)")
    em = d[d["esmfold_plddt"].notna()]
    print(f"  ESMFold (fold fidelity): rmsd median {em['esmfold_rmsd'].median():.2f} A, "
          f"plddt median {em['esmfold_plddt'].median():.1f}")


if __name__ == "__main__":
    main()
