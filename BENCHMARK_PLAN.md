# Corrected baseline and paired structural pilot

Prespecified before execution, 27 September 2026. Baseline tag: `corrected-baseline-2026-09-27` (`d29ba3b60f41a4c6e192c6716fbb9a7f788b6a07`). This pilot follows the scorer
correctness work; historical June results are not used as the corrected baseline.

## Frozen protocol

- PD-L1 / Top7 and EGFR / Top7, with repository structures and existing target
  hotspot selections. Two target families and one scaffold are a deliberately
  narrow initial screen, not a generalization benchmark.
- Seeds 20260927 and 20260928, one trajectory per case/seed/arm, default 100/100/20
  stages, two MPNN draws, soluble weights, non-interface redesign.
- Three generation arms: unchanged legacy objective and MPNN temperature 0.1;
  observed-pair-normalized fold/interface objective at temperature 0.1; unchanged
  legacy objective at MPNN temperature 0.2. Each variant changes one factor.
- Deterministically randomized arm order within each case/seed block. The MPNN
  temperature comparison must produce identical pre-MPNN backbone PDB hashes.
- Every candidate receives separate model_1_ptm and model_2_ptm validation with
  three recycles. Compare single-model versus all-models-pass selection **on this
  same frozen pool**. Record the extra model's inference time separately.
- Twelve jobs, 24 candidates, 48 per-model validations. Stop on any invalid run,
  with a 20-minute per-job timeout and a 90-minute total generation budget.
- Retain rejected candidates. Run independent scoring on the entire pool with
  fixed checkpoints, seeds and local/single-sequence inputs; no MSA service.

The baseline is the unchanged model_1_ptm decision and legacy design objective.
Running model_2_ptm for all arms supplies a paired selection experiment and is
not an implicit change to production defaults. Both generation variants remain
opt-in. Their custom loss values have different units/scales and must not be
compared directly as a quality metric.

## Evaluation and interpretation

Report per candidate: intended-template CA RMSD, target-epitope contact coverage,
interchain heavy-atom clashes, ipSAE, ESMFold monomer confidence and agreement,
Boltz interface confidence, and converged OpenMM interaction energy as a secondary
geometry diagnostic. The stock Boltz runtime lacks interface PAE; leave it unknown.

Aggregate MPNN siblings within their generating trajectory before comparing arms.
Show all four paired case/seed deltas and per-case outcomes. With only four paired
blocks, do not use a nominal p-value or a narrow bootstrap interval to claim
superiority. Report missing measurements and failures explicitly; never convert
unknown outcomes into rejection labels. Preserve candidate ancestry and input
hashes when joining scorers.

Timing includes reference preparation, generation and validation, with a shared
persistent compilation cache. Report cache/compilation effects and the second
validation model's cost. Equal attempts here are not equal-compute evidence; a
subsequent matched GPU-hour experiment is required for a cost-efficiency claim.

**No arm is promoted from this pilot.** No assay-labeled data has been provided,
so these are structural consistency and computational selection comparisons,
not measured binding accuracy. Use pilot variability and failure modes to choose
the next experiment, then freeze practical effect margins, collect more target
families/scaffolds, and reserve a held-out assay panel before an accuracy claim.

## Reproduction

`scripts/paired_pilot.py --output-root NEW --data-dir WEIGHTS` prints the complete
protocol without inference. Add `--execute` to run it on an available GPU.
`scripts/analyze_pilot.py NEW --build-pool` verifies completed source runs and
creates `NEW/pool`; score that pool with the corrected scorers and rerun the
analysis without `--build-pool`. The script reports paired trajectory results,
metric completeness and selection differences in JSON, CSV and Markdown.
