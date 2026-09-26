# FoldCraft vs BoltzProt-1 on PD-L1: exploratory in-silico comparison

June 2026 measurements; interpretation revised during September 2026 integration.
No GPU experiments were rerun for this documentation revision. The committed
measurements and original report remain available in Git history.

## What the measurements support

The recorded runs compare FoldCraft with BoltzProt-1 on one PD-L1 epitope.
Confidence scores disagree across predictors. OpenMM provides an additional
interaction-energy measurement, but neither those energies nor model confidence
establish experimental binding, equivalent generators, or superior accuracy.

## Setup

- Target: `examples/targets/pd-l1-1.pdb`, epitope `30-34,50-54,69-76`.
- FoldCraft: author example configurations, 200 sequences per fold; five folds
  scored, totaling 1,000 sequences. TIM was excluded after a template-numbering
  failure. Similar aggregate pass rates do not prove reproduction of the paper.
- BoltzProt-1: `protein:design`, no template, binder length 70–185, 200 designs.
  This is an unconstrained design task; FoldCraft specifies a binder fold.
- This report's AF2 gate is pLDDT>0.8, ipTM>0.5, iPAE<0.35. It does **not** include
  the RMSD requirement used by `baseline/score.py` for fold-conditioning success.
- The AF2 scorer uses AF2-ptm in ColabDesign binder mode, not AF2-multimer.
  It lists two model names but does not explicitly request `num_models=2`;
  consequently this is not evidence of a two-model ensemble (see INTEGRATION.md).

## Predictor agreement

AF2 passed 188/1,000 FoldCraft designs and 0/200 BoltzProt designs. FoldCraft
optimizes against AF2, so this is vulnerable to evaluation bias. Protocol
sensitivity also limits interpretation of this difference.

The reported Boltz-2 pass rates were approximately 73% among AF2 failures,
82% among AF2 passers, and 84% among BoltzProt designs. This shows limited
agreement with AF2's rejection decisions. AF2 rejection is not ground truth;
these results do not establish that Boltz-2 is producing false positives.

Reported ESMFold monomer C-alpha RMSDs of roughly 0.5–1.1 Å measure consistency
with the designed binder chain. They do not demonstrate experimental folding,
retention of the intended template fold, or a correct binding interface.

## OpenMM interaction energy

`score_openmm.py` uses Amber ff14SB with GBN2 implicit solvent. After minimizing
the predicted complex, it evaluates all three energies on those same coordinates:

```
ΔE = E(complex) − E(target alone) − E(binder alone) [kcal/mol]
```

This is an interaction-energy proxy, not binding free energy or Kd. It omits
entropy and unbound-state relaxation. A force field is architecturally distinct
from the predictors, but this does not establish unbiased ranking of their
outputs: geometry, size, composition, and selection can affect the comparison.

| Population | FoldCraft | BoltzProt-1 | Reported sequence-level statistic |
|------------|-----------|-------------|----------------------------------|
| All designs | Median −33.1; n=1,000 | Median −33.2; n=200 | Mann–Whitney p=0.22 |
| ΔE>0 | 9% | 0% | Descriptive proportions |
| Selected designs | AF2 passers: median −46.4; n=188 | Top 20 by self-ipTM: median −36.5 | p=2.2×10⁻⁴; rank-biserial −0.50 |

The raw medians are similar; p=0.22 does not demonstrate equivalence. The selected
FoldCraft set has a lower median energy under this protocol. That observation
does not establish better binders or resolve AF2 evaluation bias.

## Limits on interpretation

- One target and epitope, no experimental binding labels or calibrated negatives.
- Unequal generation budgets (1,000 versus 200) and different selection rules
  (188 AF2 passers versus 20 self-ranked designs).
- Each method is scored on its own predictor's structure.
- Multiple sequences share a design trajectory. The reported sequence-level
  tests and Wilson intervals do not model that dependence; use trajectory-level
  resampling for follow-up comparisons.
- Raw PDBs and model outputs are not committed. Tables allow aggregate
  recalculation, but a fresh clone cannot independently rerun structural scoring.
- The Boltz API is a dated, moving service; complete execution manifests and
  seeds are needed for future reproducible comparisons.

## FoldCraft observations worth following up

- Ig-like: the committed reproduction table has 95/200 designs above the ipTM
  threshold, 187/200 with template RMSD<3.5 Å, and 66/200 passing the combined
  confidence-plus-RMSD gate. The original report's 9/200 RMSD count was inconsistent
  with that table and is corrected here. This is a table recalculation, not an
  independent verification of the missing raw structures.
- Positive OpenMM interaction energies were concentrated in ankyrin and solenoid
  (reported rates 16% and 17%). This motivates inspection of the structures; it
  is not an experimental failure label.
- TIM: gaps in the original `5bvl1.pdb` numbering caused a model-length mismatch.
  The repository now includes `5bvl_af2.pdb` and a gap guard; TIM was not rerun
  for this comparison.
- Inclusive range parsing matters for contact maps. Do not assume every bundled
  map matches current runtime conventions; VHH map consistency remains open.

## Recalculate and reproduce

`python baseline/openmm_compare.py` recalculates energy summaries from tracked
`baseline/repro/<fold>/results.csv` and `baseline/boltzprot/results.csv`.
`python baseline/score.py baseline/repro` reports the separate fold-retention gate.

Scorers: `score_af2.py`, `score_boltz2.py`, `score_esmfold.py`, and
`score_openmm.py`. They need `results.csv` plus raw design structures, and write
new columns back to the results table. Use copies for new experiments.

The original report records the structures on
`reg-box-1:/root/FoldCraft/baseline/`; their availability was not verified during
integration. BoltzProt provenance: run `prot_des_sqpsGbr8wv1N3FNFGt6Z`, engine
`boltzprot v1.0`, June 18, 2026. See [BENCHMARK.md](BENCHMARK.md) for the protocol
and [INTEGRATION.md](INTEGRATION.md) for outstanding implementation limitations.
