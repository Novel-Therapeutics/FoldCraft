"""Compute the FoldCraft-vs-BoltzProt consensus table from the oracle-scored
results.csv files.

Consensus gate = a design passes BOTH an AF2 leg AND a Boltz-2 leg (the two are
from different model families, so requiring both removes single-model/self bias):
  AF2 leg     : plddt>0.8 & iptm>0.5 & ipae<0.35
                (FoldCraft: design-time plddt/iptm/ipae; BoltzProt: af2_* columns)
  Boltz-2 leg : boltz2_iptm>0.5   (boltz2_ipae is in A, reported for context)
ESMFold rmsd (binder monomer vs design chain B) reports fold fidelity.

Missing scores are unknown outcomes, reported as bounds. Observed-sample
rates do not establish full-cohort rates without a verified sampling protocol. Usage: python baseline/consensus.py
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


def consensus_counts(d):
    import numpy as np
    p, i, e = (("plddt", "iptm", "ipae") if "iptm" in d.columns
               else ("af2_plddt", "af2_iptm", "af2_ipae"))
    specs = [(p, lambda x:x>.8), (i, lambda x:x>.5), (e, lambda x:x<.35),
             ('boltz2_iptm', lambda x:x>.5)]
    known_fail = pd.Series(False, index=d.index)
    known_pass = pd.Series(True, index=d.index)
    for col, gate in specs:
        values = pd.to_numeric(d[col], errors='coerce') if col in d else pd.Series(float('nan'),index=d.index)
        present = np.isfinite(values)
        passed = gate(values)
        known_fail |= present & ~passed
        known_pass &= present & passed
    n_pass = int(known_pass.sum())
    unknown = int((~known_fail & ~known_pass).sum())
    return dict(n=len(d), known_pass=n_pass, unknown=unknown, lower=n_pass, upper=n_pass+unknown)


def main():
    print('FoldCraft consensus: AF2(plddt>.8 iptm>.5 ipae<.35) AND Boltz-2(iptm>.5)')
    print('Counts are known passes plus unresolved outcomes; bounds are not confidence intervals.')
    all_frames = []
    for f in sorted(glob.glob(os.path.join(HERE, "repro", "*", "results.csv"))):
        d = pd.read_csv(f)
        all_frames.append(d)
        c = consensus_counts(d)
        fold = os.path.basename(os.path.dirname(f))
        print(f"{fold:9s}: {c['known_pass']} known + {c['unknown']} unknown; "
              f"consensus {c['lower']}-{c['upper']}/{c['n']}")
    if all_frames:
        c = consensus_counts(pd.concat(all_frames, ignore_index=True))
        print(f"POOLED   : {c['known_pass']} known + {c['unknown']} unknown; "
              f"consensus {c['lower']}-{c['upper']}/{c['n']}")
    print("\n" + "=" * 80)
    print("BoltzProt-1 (unconstrained) -- AF2 on all 200; Boltz-2 on an observed subset (sampling provenance unverified)")
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
