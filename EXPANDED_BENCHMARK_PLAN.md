# Expanded structural benchmark — frozen before execution

Baseline: `validated-baseline-2026-09-27`, commit `c5e3f78`. Earlier tags and
historical results remain unchanged. This experiment measures computational
structural consistency, not binding accuracy.

## Performance gate

Compare full and losslessly compact prediction artifacts on short (IFNAR/Top7),
medium (PD-L1/ankyrin) and long (EGFR/TIM) complexes. Each mode gets its own empty
persistent compilation cache for cold measurement and reuses that cache for warm
measurement. Use three design iterations, two MPNN candidates, two AF2 validation
models and three recycles. Measure elapsed time, sampled aggregate process-tree
RSS, device-used memory and artifact bytes. These short trajectories isolate
compilation/storage costs; they do not estimate full-search throughput.

Retain compact arrays without quantization: PAE, coordinates, masks, residue
identity/index, pLDDT, pTM and ipTM when present. Preserve model/seed/metric receipts
and complete optimization histories. Promote compact storage only after exact
prediction parity on all three sizes, downstream-reader compatibility, >=80%
pickle-size reduction and no consistent >10% warm-runtime regression. Full debug
artifacts remain available. Do not infer a speed or memory improvement from a
storage improvement alone.

## Expanded generation and evaluation

Four bundled targets × three scaffolds (Top7, barrel, ankyrin) × two seeds × two
MPNN temperature arms = **48 full trajectories, 96 candidates, 192 AF2 predictions**.
Temperature is 0.1 versus 0.2, with all other settings fixed: 100/100/20 design
stages, two MPNN draws, three validation recycles, both AF2 models retained.
Randomize arm order deterministically and verify identical paired backbones.
Run at most two owned generation jobs concurrently; the campaign's allocated
GPU wall time includes both. No throughput gain is inferred from overlapping jobs.

PD-L1 and PD-1 share a conservative immune-checkpoint family group. They and EGFR
are development cases. **IFNAR is held out**, with no outcome inspection or tuning
before the fixed rules are applied. IFNAR is only one unseen family: the minimum
two-family promotion requirement cannot be satisfied by this panel alone. This
expanded screen can reject regressions and quantify hypotheses; it cannot prove
broad generalization. No assay-labeled panel is available in the project.

Score every candidate, including rejects, with ESMFold and two single-sequence
Boltz repeats, with at most two independent scorer processes concurrently. Couple Boltz seeds by case, trajectory seed, MPNN draw and repeat,
excluding arm. Record checkpoint hashes and exact output selection. The primary
independent proxy is ESMFold confidence >=70, ESMFold/design RMSD <=3.5 Å and mean
Boltz ipTM >=0.6. Report component metrics, template RMSD, intended epitope coverage,
clashes, diversity and every missing score separately. OpenMM energy is not a
primary endpoint and is not needed to interpret this temperature/ranking test.

Average sibling outcomes within each case/seed before estimating temperature
changes. Report development and holdout separately, paired deltas and resampling
uncertainty; do not treat 96 sequences as 96 independent replicates. Family-level
estimates are descriptive with only three grouped families. Equal draws are the
primary search comparison; no equal-GPU-time search-quality claim is made.

## Candidate ranking and promotion

Compare quota-one selection on the same four-candidate case/seed pool:
1. Baseline: highest AF2 ipTM.
2. Candidate: fewest violations of template RMSD <=3.5 Å, interchain clashes <=5
   and epitope coverage >=0.1; break ties by AF2 ipTM.

These ranking rules never use ESMFold or Boltz outcomes. Also report ordinary
single-model and two-model confidence-gate selections and their coverage.

Scientific default promotion requires held-out primary-proxy improvement >=10
percentage points, paired 95% bootstrap lower bound >0, template RMSD regression
<=0.5 Å, clash regression <=1, and at least two unseen target families. Actual
binding-accuracy claims additionally require assay labels. Otherwise retain the
existing scientific defaults, even when individual proxy metrics improve.

All rules and input/code hashes are written to `protocol.json` before generation.
Budgets: five hours generation, four hours evaluation, stop on invalid artifacts.
No failed normalized-loss arm is included and no post-hoc threshold rescue is
permitted. Performance storage promotion is evaluated separately from scientific
defaults.
