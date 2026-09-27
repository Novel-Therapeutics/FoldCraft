# FoldCraft change plan: correctness before improvements

Prepared 26 September 2026 for `codex/correctness-and-validation`, reviewed at
`e485f3244e47464ae6083c193f234e91c099e0c5`. This is an implementation plan, not a
record of completed fixes. See [the deep review](DEEP_REVIEW.md) for evidence.

**Implementation update, 27 September:** the CPU and GPU correctness work,
scorer validation and corrected-baseline tag have been tested. This does not
close every original review item: see [the closure status](CLOSURE_STATUS.md). See
[CPU fixes](../../CPU_FIXES.md), [GPU checks](../../GPU_FIXES.md),
[scorer fixes](../../SCORING_FIXES.md), [inference provenance](../../INFERENCE_PROVENANCE.md)
[closure validation](../../CLOSURE_VALIDATION.md) and [pilot results](../../PILOT_RESULTS.md).
The four-item follow-up is complete: freeze the validated baseline, profile and
promote a lossless storage improvement, execute an expanded paired structural
benchmark, and apply prespecified promotion criteria. See
[performance results](../../PERFORMANCE_RESULTS.md) and
[expanded results](../../EXPANDED_BENCHMARK_RESULTS.md). This completes that
bounded study, not every future hypothesis in I1–I5 or assay validation.
The plan below preserves the original ordering; its historical “Immediate next PR”
section is superseded by the current closure status and benchmark reports.

The objective is reliable fold-conditioned binder design and defensible candidate
ranking. FoldCraft uses pretrained predictors; there is no foundation-model
training pipeline here. Correct conditioning, evaluation and search come first.

## Working rules and dependencies

Preserve `integration-baseline-2026-09-26` and the recorded CSVs. Correctness fixes
will change some outputs, so establish a separately versioned **corrected
baseline** after the engineering gate. Compare scientific improvements with that
baseline, and separately report the effect of repairing the original pipeline.
Do not overwrite or relabel historical one-model results as two-model results.

Keep each numbered item below as a focused PR (split large items along the
stated interfaces). Add regression tests with each fix. Introduce only the small
pure functions and inference adapters needed to test those contracts; a broad
rewrite is unnecessary. Keep `main` current by merging reviewed steps before
starting dependent work.

The main dependency chain is:

`F1 input mapping → F2 model/sequence contracts → F3 run records → F4 scheduling
→ F5 scoring → F6 benchmark integrity → engineering gate → improvements`.

F7 entry-point repairs can proceed alongside F3–F6 once the shared interfaces
are stable. F5 residue alignment depends on F1; multi-model score handling depends
on F2. Physical/fold acceptance changes must wait for F5's trustworthy scores.

## Bug fixes, in implementation order

| Step | Scope and reason | Required result before merge |
|---|---|---|
| **F1 — Input and residue contracts** | Define one mapping from original chain/residue/insertion code to model position. Make target missing-residue handling consistent across map preparation, design and validation. Validate ranges, chain IDs, polymer/backbone availability, map shapes/finiteness, stage counts and positive budgets before loading weights. Version VHH conditioning conventions. | Offset numbering selects the same physical residue; gaps/missing atoms either map consistently or fail clearly before GPU work; zero/reversed/out-of-chain hotspots cannot write wrong quadrants. Golden fixtures explicitly distinguish historical VHH maps from current conventions. |
| **F2 — MPNN and model outputs** | Include the terminal residue; correctly handle an empty mutable set without unfreezing the target. Explicitly select validation models/recycles and record actual executed models. Save sequence, structure, PAE and metrics under a candidate-and-model identity; define ensemble aggregation. | Tests assert exact mutable indices and target immutability. Empty selection follows a documented no-redesign path. Two requested models actually run, and each model's score refers to its own structure/PAE. No mean-log/model-1-structure mismatch. |
| **F3 — Bounded runs and durable records** | Add maximum trajectories/time to success sampling. Introduce manifests, deterministic stage seeds, candidate/parent IDs and explicit running/failed/complete states. Make checkpoints atomic through a second temporary file; refuse accidental reuse of incompatible output. Separate committed reference tables from new runtime outputs. | Fault injection at every write boundary leaves either the last valid checkpoint or a recoverable incomplete run. Old final files cannot make a new attempt look complete. Zero-success jobs stop with a machine-readable reason and preserve attempted/rejected candidates. |
| **F4 — Scheduler and runner correctness** | Resume only compatible completed manifests with required artifacts. Avoid treating archived CSVs as executable runs. Make merge publication atomic; retain successful chunks after interruption. Enforce exclusive retry reservations, explicit GPU process ownership/cleanup, and nonzero campaign failure exits. | A fresh checkout schedules all requested work; changing target/hotspots/count/chunking invalidates incompatible results. A partial merge is never complete. A retry marked exclusive cannot share its GPU. Failed shell jobs yield a failing campaign exit. |
| **F5 — Trustworthy structural scoring** | Key scores by candidate, sequence/structure hash, model/checkpoint and scorer settings. Use isolated Boltz output directories and exact output manifests; preserve selected structure identity. Give each scorer independent result files with atomic publication and a validated join. Align RMSD by residue identity/sequence with coverage checks; retain directional ipSAE and validate PAE shape/split. Fix OpenMM zero-iteration semantics and record convergence/finite-energy QC. | Reused workdirs cannot select stale predictions. Concurrent scorers cannot erase one another's columns. Changing settings triggers rescoring. Missing/internal residues cannot earn a misleading low RMSD. `--min-iters 0` bypasses minimization. Invalid numerical scores remain failed/unknown, with a reason. |
| **F6 — Experimental protocol and statistics** | Share fold/target configs between production, examples and A/B scripts. Pair all applicable random streams, randomize arm order and record effective settings. Require metric completeness or report unknown outcomes explicitly. Group uncertainty by trajectory and target family; generate report tables from the same analysis code. | Non-Top7 A/B configs match the selected protocol. Current data reports 155 known consensus passes plus 27 unresolved cases, not an exact 155/1,000 result. Recomputed summaries agree with reports. Seeded smoke experiments are repeatable; no confidence gain is described as demonstrated binding accuracy. |
| **F7 — Supported entry points and installation** | Repair notebook undefined variables/indexing/interface membership, experimental imports/modes/length bounds, and shell path handling. Use explicit weight paths and pinned dependency contracts; validate the intended GPU platform before a large download. Make notebooks thin clients of the tested helpers where practical. | Fresh ordered notebook execution reaches the inference adapter without unrelated NameErrors; interface residues remain fixed when requested. Experimental default behavior is explicit, length bounds are inclusive, and paths with spaces/metacharacters are safe. CPU help/preflight can run without installing GPU weights. |

### F1 design decisions

Use original PDB identifiers as the canonical biological reference and explicit
model-position maps internally. Preserve simple historical numeric inputs through
an explicit documented convention; reject ambiguous inputs rather than silently
reinterpret them. A quick strict-preflight guard can ship before full mapping
support, but it must reject unsupported inputs consistently across all stages.

Unknown/masked contacts and explicit observed negatives need distinct data fields.
Adding that representation is engineering; changing the loss applied to it is an
experiment under I3 below. Do not treat every zero in the VHH map as a negative.

### F2 and F5 must preserve model identity

Adding only `num_models=2` creates a second inconsistency: the dependency averages
logs while current RMSD/ipSAE readers select model zero. Either store separate
files per model or an explicit indexed collection consumed consistently by every
scorer. Record model-specific pass criteria and disagreement. Define whether an
ensemble decision requires both models, an aggregate, or a calibrated score; do
not silently inherit a policy from averaged arrays.

### F3/F4 artifact contract

Use a run ID plus a configuration hash including inputs, mappings, maps, source
commit, dependency/checkpoint IDs and effective settings. A completion manifest
must identify the expected trajectory/candidate records and required files.
Historical CSV-only material must remain clearly labeled as imported reference
measurements. A new output root should be the default for new campaigns.

For interrupted runs, distinguish completed candidates, failed candidates,
unscored candidates and pending work. Do not infer state from a CSV's existence.
Runtime limits should terminate owned child processes and preserve the last
consistent checkpoint, without affecting other GPU users.

## Engineering gate: no accuracy tuning before this passes

1. Run the current 113 tests plus adversarial regression tests for each fix.
   Replace the invalid differential-map fixture with valid fixtures and explicit
   rejection tests; matching two copies of an indexing bug is not correctness.
2. Exercise orchestration with fake subprocesses/GPU inventories and injected
   failures: interrupted writes, partial merges, changed configs, stale files,
   concurrent scorers and exclusive retries. No GPU is needed for these tests.
3. On a supported GPU, run one short trajectory with two MPNN children through
   fixed-count and bounded-success modes, plus a VHH case and a longer target.
   Validate actual model IDs, chain mappings, per-model artifacts, acceptance
   reasons, restart behavior and exit status. Use a failing acceptance fixture
   to verify budget termination. Record cold/warm timings and peak RAM/VRAM.
4. Recalculate scores from saved artifacts without rerunning inference. Confirm
   score provenance and sufficient alignment coverage. Missing artifacts must
   fail explicitly rather than produce a completed run or a negative label.
5. Pin and tag the corrected baseline. Publish its configuration and small
   reproducible smoke artifacts; keep historical benchmark results separate.

Passing CPU tests does not replace GPU integration checks. This review did not
perform GPU inference, rescore raw structures, or run binding assays.

## Improvements after the engineering gate

### I1 — Reduce resource use without changing scientific behavior

Measure first, then remove unused trajectory histories and serialize only the
fields needed for replay/scoring. Store invariant maps once per run. Separate
compact candidate artifacts from optional debug tensors. Then investigate reuse
across same-shape trajectories and persistent compilation caching; avoid unsafe
interaction with `clear_mem()` and device RNG buffers.

**Benchmark:** paired replay of short, medium and long complexes under pinned
versions/seeds; cold and warm wall time, peak host/device memory, bytes per
candidate, and exact/tolerance-bounded structural/metric parity. Include
compilation and failed runs in total GPU-hours. Do not count the already-present
per-trajectory validation-model reuse as a new gain. Promote only when required
artifacts remain complete and parity passes.

### I2 — Improve acceptance and candidate ranking

Compute intended-fold/core alignment, intended-epitope coverage, clashes and
independent binder refolding. Compare current confidence gates with augmented
filters on the **same frozen candidate pool**. Compare ipSAE variants and one
cross-family predictor as additional features, not ground-truth labels. Evaluate
full target context when cropped constructs could conceal steric conflicts.

**Benchmark:** use labeled binders/nonbinders with assay and construct provenance
where available; split by target family, group related sequences and reserve a
final holdout. Report precision at the synthesis quota, recall, macro average
precision and calibration, plus fold/epitope/clash outcomes. Audit a randomized
subset of rejected candidates with the expensive evaluator to measure losses
from early filters. Without labels, report structural consistency and selection
changes only; do not call this improved binding accuracy.

### I3 — Normalize and redesign the objective incrementally

First separate fold and interface terms and normalize by meaningful observed-pair
counts. Check padding/length invariance and gradient scale before tuning weights.
Next compare dense all-pairs restraints with sparse/coverage-based interface
restraints at explicit geometric cutoffs. After these controls, revisit a small
iPAE weight sweep. Test observed-negative penalties only with explicit masks and
appropriate normalization.

**Benchmark:** matched seeds, backbones/inputs and validation settings; one change
at a time. Measure fold retention, actual epitope contacts, clashes, diversity and
independent ranking/experimental outcomes. Report both equal attempts and equal
GPU time. Count reference preparation and validation costs. A lower custom loss
or higher same-family confidence is not the promotion criterion.

**Evidence-based priorities:** best-versus-last is low priority after the current
negative pilot; iPAE remains an experiment; the tested off-mask penalty at 0.1/0.3
should not ship. Its failure does not rule out a different normalized,
observation-aware formulation.

### I4 — Improve MPNN search and diversity

On shared archived backbones, compare corrected non-interface redesign with a
controlled interface-adjacent mutable shell, temperatures and weight choices.
Measure additional samples separately from changes to the mutation policy.
Deduplicate exact candidates before expensive validation, retaining original draw
counts and sequence ancestry. Compare diversity-aware shortlisting with the
existing order on the same candidate pool.

**Benchmark:** fixed draw count, fixed unique-validation quota and equal total
compute are three separate comparisons. Measure independent yield, interface
retention, intended fold, unique sequence/structure clusters and expression
where available. Include resampling/top-up costs and avoid selecting only the
best descendant as if it were an independent trajectory.

### I5 — Model, recycle and reference choices

Compare ptm versus multimer generation, then 0/1/3 design recycles, and then
experimental-coordinate versus predicted reference maps. Keep validation fixed
while testing generation changes. Treat longer refinement and iterative redesign
as later experiments: the iteration scorer becomes an optimization target and
cannot also be the untouched final evaluator.

**Benchmark:** paired inputs/seeds, multiple folds/target families, independent
final evaluation, structural drift/diversity and quality per GPU-hour. Only
advance costly iterations if they beat spending the same budget on independent
one-pass trajectories. VHH framework/disulfide/CDR changes and biological-context
variants require separate stratified studies rather than universal defaults.

## Common experiment protocol and promotion rules

- **Data split:** use development targets to tune thresholds and weights, then
  freeze the method before examining held-out target families. All sibling
  sequences and closely related targets stay in the same split.
- **Exploratory screen:** start with 12 prespecified valid target/scaffold cases
  spanning several folds and at least four target families (new fixtures are
  required), with 20 matched trajectories and five MPNN children per case: 240
  trajectories and up to 1,200 candidates per arm. This is a planning example,
  not a guaranteed powered experiment. Restrict to one or two promising arms
  rather than a large Cartesian hyperparameter sweep.
- **Uncertainty:** resample trajectories within target/family groups; pair runs
  when seeds/inputs support pairing. Show per-target outcomes and a family-level
  summary. Account for selection and multiple exploratory comparisons. Choose
  confirmatory sample size from pilot variance and a predeclared practical gain.
- **Budget:** profile the corrected baseline before setting GPU-hour limits.
  For each finalist report equal-attempt and equal-compute outcomes. Include
  compilation, failures, preprocessing, extra models and all scoring passes.
- **Promotion:** require the primary held-out improvement to exceed a predefined
  practical margin with appropriate uncertainty, no unacceptable loss of fold
  retention/diversity, and a documented cost tradeoff. Define margins before
  seeing results. Confirm claimed binding gains on a prospective assay panel.
- **Artifacts:** immutable manifest; explicit target/binder sequences and maps;
  candidate/trajectory/model IDs; all attempted and rejected records; per-model
  structures/PAE/metrics; scorer statuses; alignment coverage; timings, peak
  memory and disk footprint; analysis code and a concise decision record.

A calibrated, simple ranker can follow I2 when sufficient assay-defined data
exists. Predictor fine-tuning should wait until leakage-controlled data and
simpler pipeline improvements justify that additional complexity.

## Immediate next PR

Start with **F1: input validation and residue mapping**, including a minimal
pure preparation interface and adversarial tests. Do not alter loss weights,
checkpoint choice, ranking thresholds or model families in that PR. Follow with
F2's coupled MPNN/model-artifact fixes. New large campaigns should wait until
F3–F6 also pass the engineering gate.
