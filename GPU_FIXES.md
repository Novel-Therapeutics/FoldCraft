# GPU fixes and validation — 27 September 2026

Status: the real GPU smoke suites passed on Bizon after explicit user authorization.
Both zero- and three-recycle validation passed all five cases. This establishes
runtime and artifact correctness for the tested cases; it does not establish
better prediction accuracy. Full-schedule results are recorded below.

## First fix: preserve AF2 model identity

The pinned ColabDesign source averages floating auxiliary values across models,
while using the first model's coordinates in some views. Merely setting
`num_models=2` would combine ensemble scores with model-specific structures.

`model_validation.py` instead calls each requested model separately with
`num_models=1, sample_models=False`. It checks the actual numeric dispatch IDs
against the loaded model names, validates the single-model PAE and finite scores,
and publishes a receipt only after all requested models finish.

The current default is still `model_1_ptm` with three recycles. Experimental
multi-model validation is selected explicitly:

```sh
python FoldCraft.py ... --validation_models model_1_ptm,model_2_ptm
```

- The first requested model is the **primary model**. Existing CSV metric columns
  and `<candidate>.pdb/.pickle` refer to that exact model, never an average.
- Additional files use `<candidate>.<model>.pdb/.pickle`.
- `<candidate>.validation.json` records sequence, ordered model IDs, seeds, actual
  recycles, individual scores, artifact hashes and model disagreement.
- `validation_pass` requires **every requested model** to pass the existing
  pLDDT/ipTM/iPAE thresholds. This conservative policy is explicit; it is not a
  demonstrated accuracy improvement. In fixed-count mode all candidates remain
  saved; bounded-success sampling counts candidates passing that policy.
- Rejected predictions retain per-model artifacts. A failure halfway through
  validation cannot publish a receipt or a completed run.
- Completion and merging preserve/check secondary-model files. Consensus and
  AF2-pass-only scoring honor the ensemble decision. Missing decisions are unknown.
- RMSD/ipSAE exports retain primary-model CSV columns and write separate
  `rmsd.models.json` / `ipsae.models.json` records for all models.

The A/B experiments and independent AF2 oracle still use their explicitly recorded
single-model protocol. Changing those protocols belongs to a matched experiment,
not an automatic relabeling of existing results.

## GPU checks

`scripts/gpu_smoke.py` only prints commands unless `--execute` is present. It does
not SSH, install software, download weights or start background work. After GPU
access is confirmed, use a fresh isolated checkout/environment on Bizon.

```sh
# Planning only; no GPU access, imports of JAX, or output directory creation.
python scripts/gpu_smoke.py --output-root runs/gpu-smoke-01 --data-dir /path/to/weights

# Run only after user confirmation, from the approved GPU environment.
python scripts/gpu_smoke.py --output-root runs/gpu-smoke-01 --data-dir /path/to/weights --gpu 0 --execute
```

The runtime probe rejects CPU fallback, checks float32/bfloat16 JIT kernels and
autodiff, and records device/driver/package details plus hashes of both AF2
checkpoints. Passing these kernels does not certify ColabDesign compatibility;
the subsequent pipeline cases must also pass.

The serial cases are fixed-count design, same-seed replay, one-trajectory bounded
sampling, VHH and a longer EGFR target. All request both validation models and two
MPNN samples, with 1/1/1 design stages. Initial validation uses zero recycles to
exercise interfaces cheaply; repeat with `--recycles 3` before calling the normal
validation path tested. Binder-reference preparation still uses three recycles.
A sampling run may complete or explicitly exhaust its one-trajectory budget.

Each case has a 30-minute timeout (configurable). Timeout/interruption terminates
only the process group started by the smoke runner. Logs, timings, run states,
runtime identity and checkpoint hashes are preserved. Replay checks candidate
sequence/model identity, pass decisions, scores, PAE and unrounded atom coordinates
(absolute tolerance 1e-5). Failures before run-manifest creation are reported as
`not_started`, preserving the actual process failure instead of masking it with a
missing-file exception.
The small stages are an integration test, not a quality benchmark.

## Local verification and remaining work

The full CPU suite now passes 174 tests locally and on Bizon, including single-/two-model driver
execution with inference doubles, distinct per-model outputs, wrong dispatch,
missing weights, mid-ensemble failure, merge/integrity checks, per-model ipSAE,
unknown/disagreeing decisions, and GPU-plan/timeout behavior.

OpenMM convergence/force QC, validation of external GPU scorers, and complete
model-checkpoint provenance for those scorers remain subsequent work. The
all-models-pass policy still needs a matched accuracy benchmark before promotion.


## Bizon evidence

Evidence: [GPU test reports](docs/review/evidence/gpu-2026-09-27/).
The isolated remote directory is `/home/bizon/projects/foldcraft-gpu-20260927`;
full structures and prediction pickles remain there. Local evidence includes
logs, commands, runtime/weight hashes, per-model receipts, artifact audits,
package versions and GPU telemetry. Runs tested the uncommitted GPU changes on
CPU-fix commit `6b77a430004fccb500b98f9b3c66bdaae23bcd84`; each `run.json`
records the actual execution source hashes. Documentation and help-text updates
followed execution.

Tested runtime: RTX PRO 6000 Blackwell, driver 580.65.06, Python 3.12.3,
JAX/jaxlib 0.6.2, CUDA 12 pip libraries, NumPy 1.26.4, and ColabDesign commit
`e31a56fe1d9b4de25c8697f3a28b75892941cc72`. Float32/bfloat16 JIT matrix
multiplication and autodiff passed on the CUDA device. The full package lock is
[environment.freeze.txt](docs/review/evidence/gpu-2026-09-27/environment.freeze.txt).
It can recreate this isolated Linux runtime using `python3 -m venv` and
`venv/bin/pip install -r` with that file. This is separate from the older generic
installer stack, which was not certified by these tests.

| Case | 0 validation recycles | 3 validation recycles | Result |
| --- | ---: | ---: | --- |
| Fixed PD-L1 / Top7 | 125.8 s | 29.6 s | Complete, both models and two MPNN samples |
| Same-seed replay | 26.9 s | 28.2 s | Complete; zero PAE and coordinate differences |
| Bounded sampling | 20.4 s | 20.5 s | Correct exhaustion after one rejected trajectory |
| PD-L1 / VHH | 102.7 s | 25.6 s | Complete, both models and two MPNN samples |
| EGFR / VHH | 108.0 s | 29.4 s | Complete, 301-residue complex |

A shared persistent compilation cache was enabled. The initial fixed/VHH/EGFR
runs include compilation for new shapes; subsequent cases reuse it. These are
observed integration timings, not a controlled runtime comparison. Peak observed
GPU memory in both suites was 6,732 MiB (6.57 GiB), sampled every two seconds and
including graphics/background allocations; this is not an exact allocator peak.

Independent audits passed for all 32 per-model predictions across the two suites:
PDB sequence and chain lengths, unchanged target and fixed MPNN interface residues,
finite PAE and coordinates, PDB/auxiliary-coordinate agreement within PDB rounding,
requested/actual recycles, hashes, per-model ipSAE, and template RMSD where a
binder template exists. Replay
matched exactly, including unrounded coordinates, in both recycle settings.
No inference failure or out-of-memory failure occurred in either suite.

The tiny smoke schedules intentionally do not measure design quality. None of
those candidates passed the acceptance gates, and bounded sampling exercised
exhaustion rather than successful early stopping; successful stopping and
model-disagreement branches remain covered by CPU regression tests.


The additional default-path run completed two consecutive 100/100/20 trajectories
(440 optimization steps total), two MPNN candidates per trajectory, and default
single-model validation with three recycles. It finished in 214.6 seconds with a
warm compilation cache. Peak observed GPU memory was 5,252 MiB (5.13 GiB), including
the second trajectory after `clear_mem()`. All four prediction artifact audits
passed, including fixed interface residues and target sequence preservation.

Overall: **11 successful integration invocations**, **36 audited per-model
predictions**, and **174 passing CPU tests** on both the local and Bizon runtimes.
The two bounded-sampling invocations intentionally returned exhausted status;
that is the expected passing test outcome. The GPU was confirmed free of compute
processes after completion. No deployment, commit or push was performed for these
GPU changes.
