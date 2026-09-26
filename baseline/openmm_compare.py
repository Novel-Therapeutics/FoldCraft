"""Exploratory OpenMM interaction-energy comparison of recorded design sets.

Reports raw and selected populations under the existing selection rules.
Interaction energy is not binding free energy; unequal budgets, different
selection rules, predicted geometries, and shared trajectories limit inference.
The sequence-level Mann-Whitney statistics do not establish generator equivalence
or superiority in binding accuracy. See REPORT.md for the historical results.

Usage: python baseline/openmm_compare.py
"""
import glob
import os
from math import sqrt

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def af2_mask(d):
    p, i, e = (("plddt", "iptm", "ipae") if "iptm" in d.columns
               else ("af2_plddt", "af2_iptm", "af2_ipae"))
    return (d[p] > 0.8) & (d[i] > 0.5) & (d[e] < 0.35)


def self_iptm_column(d):
    """The column holding a method's OWN ipTM, for the 'each method's own top
    designs' (forwarded) framing.

    FoldCraft hallucinates against AF2, so its design-time ``iptm`` IS its
    self-score; BoltzProt-1 is Boltz-family and self-reports ``boltz_iptm``.
    Neither ``af2_iptm`` (the independent AF2 *judge*) nor ``boltz2_iptm`` (the
    open Boltz-2 *oracle*) is a method's own ranking, so they must NOT be used to
    pick a method's own top designs -- doing so silently judges one method by
    another's model. Fail loud rather than fall back to a judge column (the bug
    this replaces: with no bare ``iptm`` column, BoltzProt was ranked by
    ``af2_iptm`` instead of ``boltz_iptm``).
    """
    for col in ("iptm", "boltz_iptm"):
        if col in d.columns:
            return col
    raise SystemExit(
        "no self-ipTM column ('iptm' or 'boltz_iptm') in results.csv; cannot "
        "rank a method's own top designs. Columns: " + ", ".join(map(str, d.columns)))


def mann_whitney(a, b):
    """Two-sided Mann-Whitney U of a vs b with tie-corrected normal approx.
    Returns (U, z, p_two_sided, rank_biserial). rank_biserial>0 means a tends
    to be LARGER than b; for dE 'more negative is better' so we care about a<b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    Ra = ranks[:na].sum()
    Ua = Ra - na * (na + 1) / 2.0
    U = Ua
    mu = na * nb / 2.0
    # tie correction
    _, c = np.unique(allv, return_counts=True)
    tie = (c ** 3 - c).sum()
    n = na + nb
    sig = sqrt(na * nb / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    z = (U - mu) / sig if sig > 0 else 0.0
    # two-sided p via erfc
    from math import erfc
    p = erfc(abs(z) / sqrt(2))
    rank_biserial = 2 * Ua / (na * nb) - 1
    return U, z, p, rank_biserial


def describe(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    q = np.percentile(x, [25, 50, 75])
    # 5% trimmed mean: a handful of designs have a stuck minimization (clashes not
    # relieved -> energy ~1e19), which is non-physical and would swamp the plain
    # mean; the median/Mann-Whitney are already rank-robust, the trimmed mean keeps
    # the printed mean honest. The >0 fraction (net-repulsive interface) is itself
    # a quality signal and is reported separately.
    lo, hi = np.percentile(x, [5, 95])
    tm = x[(x >= lo) & (x <= hi)].mean()
    return (f"n={len(x):4d}  median={q[1]:7.2f}  IQR[{q[0]:.1f},{q[2]:.1f}]  "
            f"trimMean={tm:7.2f}  dE>0(repulsive):{100*(x>0).mean():4.0f}%  "
            f"<-20:{100*(x<-20).mean():4.0f}%")


def load():
    fc = []
    for f in sorted(glob.glob(os.path.join(HERE, "repro", "*", "results.csv"))):
        d = pd.read_csv(f)
        if "openmm_dE" not in d.columns or d["openmm_dE"].notna().sum() == 0:
            continue
        d["fold"] = os.path.basename(os.path.dirname(f))
        fc.append(d)
    fc = pd.concat(fc, ignore_index=True) if fc else pd.DataFrame()
    bp = pd.read_csv(os.path.join(HERE, "boltzprot", "results.csv"))
    return fc, bp


def main():
    fc, bp = load()
    fc_dE = fc["openmm_dE"].dropna()
    bp_dE = bp["openmm_dE"].dropna()

    print("=" * 78)
    print("OpenMM interface interaction energy  (kcal/mol, more negative = better)")
    print("Exploratory proxy; sequence-level statistics do not account for shared trajectories")
    print("=" * 78)

    print("\n(A) RAW -- every design emitted (selection-free)")
    print(f"  FoldCraft  {describe(fc_dE)}")
    print(f"  BoltzProt  {describe(bp_dE)}")
    U, z, p_raw, rb = mann_whitney(fc_dE, bp_dE)
    p = p_raw
    better = "FoldCraft lower median energy" if fc_dE.median() < bp_dE.median() else "BoltzProt lower median energy"
    print(f"  Mann-Whitney: z={z:.2f} p={p:.1e} rank-biserial={rb:+.2f}  -> {better}")
    print(f"  per-fold medians: " +
          ", ".join(f"{k}={v:.1f}" for k, v in
                    fc.groupby('fold')['openmm_dE'].median().items()))

    print("\n(B) FORWARDED -- each method's pipeline output")
    fc_fwd = fc[af2_mask(fc)]["openmm_dE"].dropna()
    bp_af2 = bp[af2_mask(bp)]
    if len(bp_af2) >= 5:
        bp_fwd = bp_af2["openmm_dE"].dropna()
        bp_lbl = f"AF2-passers (n={len(bp_fwd)})"
    else:
        # BoltzProt has ~no AF2-passers; approximate its pipeline output by its
        # own top designs, ranked by its self-reported confidence (boltz_iptm) --
        # NOT af2_iptm/boltz2_iptm, which are external judges (see self_iptm_column).
        k = max(20, len(bp) // 10)
        col = self_iptm_column(bp)
        bp_fwd = bp.nlargest(k, col)["openmm_dE"].dropna()
        bp_lbl = f"self-top-{k} by {col} (only {len(bp_af2)} AF2-passers)"
    print(f"  FoldCraft  AF2-passers (n={len(fc_fwd)})  {describe(fc_fwd)}")
    print(f"  BoltzProt  {bp_lbl}  {describe(bp_fwd)}")
    if len(fc_fwd) >= 5 and len(bp_fwd) >= 5:
        U, z, p, rb = mann_whitney(fc_fwd, bp_fwd)
        better = "FoldCraft" if fc_fwd.median() < bp_fwd.median() else "BoltzProt"
        print(f"  Mann-Whitney: z={z:.2f} p={p:.1e} rank-biserial={rb:+.2f}  "
              f"-> {better} lower median energy")

    print("\nINTERPRETATION LIMITS")
    print(f"  RAW medians: FoldCraft {fc_dE.median():.1f}, BoltzProt "
          f"{bp_dE.median():.1f} kcal/mol; sequence-level p={p_raw:.2g}.")
    print("  A non-significant difference does not demonstrate equivalence.")
    print(f"  Positive interaction energy: FoldCraft {100*(fc_dE>0).mean():.0f}%, "
          f"BoltzProt {100*(bp_dE>0).mean():.0f}%.")
    print(f"  SELECTED medians: FoldCraft {fc_fwd.median():.1f}, BoltzProt "
          f"{bp_fwd.median():.1f} kcal/mol.")
    print("  Unequal generation budgets and different selection rules limit comparison.")
    print("  OpenMM energy is not binding free energy or experimental binding evidence.")
    print("  These measurements do not resolve predictor bias or establish better binders.")


if __name__ == "__main__":
    main()
