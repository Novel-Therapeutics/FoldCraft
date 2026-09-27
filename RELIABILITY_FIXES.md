# Runtime reliability and review closure — 27 September 2026

This pass closes the production timeout and validated-installation gaps. It also
fixes two bookkeeping/reporting defects found during the closure audit. The
[item-by-item review status](docs/review/CLOSURE_STATUS.md) identifies remaining
work; this is not a claim that every review item or possible bug is closed.

## Production wall-time supervision

- `FoldCraft.py --timeout_minutes N` defaults to **360 minutes per invocation**.
  CPU preflight and exclusive run creation happen first. A separate supervisor
  runs inference, including model loading, in its own process group. GPU/native
  calls cannot block the supervisor's deadline.
- On timeout the supervisor sends TERM, allows five seconds for group cleanup,
  then kills remaining owned group members. It returns **124** and publishes
  `timed_out`. SIGINT/SIGTERM record `interrupted` and return 130/143. A failed or
  timed-out worker cannot publish a valid completed run; existing partial files
  are preserved. The timeout is an inference budget plus bounded cleanup, not an
  exact upper bound on preflight, filesystem I/O or uninterruptible kernel I/O.
- Scheduler `--timeout-minutes N` applies per chunk attempt, including its one
  existing retry. Different worker sessions are cleaned through the CLI's signal
  handler before the scheduler's cleanup grace period expires. Changing only the
  time budget does not invalidate a scientifically complete chunk; the effective
  budget is retained in each actual run manifest.
- The CLI always supervises. `main(supervised=False)` is an explicit internal
  adapter for CPU inference doubles, not a CLI option. Mid-trajectory recovery,
  Windows supervision and surviving a hard-killed supervisor/host are outside
  this implementation. Standalone scorers/experimental scripts need their own
  external limits.

## Reproducible core installer

```bash
bash install_foldcraft.sh --env /new/environment --python python3.12 --data-dir /existing/af2-weights
```

The installer reproduces `requirements/bizon-linux-py312.txt`: Linux x86_64,
Python 3.12, JAX/jaxlib 0.6.2, the pinned CUDA 12 libraries, NumPy 1.26.4, and
ColabDesign commit `e31a56fe1d9b4de25c8697f3a28b75892941cc72`.
It refuses existing environment paths, verifies the official model_1_ptm and
model_2_ptm checkpoint hashes, checks package dependencies and imports, runs the
GPU kernel/autodiff probe, and saves an installation status, runtime report and
package freeze. Failed environments are retained for diagnosis and never labeled
successful. It installs neither weights nor the separate experimental BindCraft
or independent scorer environments. The old conda/CUDA flags were replaced;
README commands now match the supported installer.

## Additional audited fixes

Fixed-count design retains rejected candidates, but its `attempts.json` previously
marked every retained candidate accepted. It now uses the actual all-models-pass
decision. CSV metrics and validation receipts already had the correct decision;
the paired pilot consumed those receipts, so its results are unchanged.

`baseline/score.py` now reports descriptive candidate counts without the old
binomial confidence interval that treated sibling sequences as independent. It
also rejects missing/nonfinite acceptance metrics instead of treating unknown
measurements as rejection. Historical CSVs and reports remain intact. Clustered
uncertainty and assay calibration still require the expanded experimental design.

## Validation

- **207 CPU tests passed locally and in the freshly installed Bizon environment.**
  New tests cover real hanging processes, a TERM-resistant descendant, unrelated
  process preservation, supervisor cancellation, checkpoint retention, false
  completion, timeout propagation, installer overwrite refusal, rejected
  fixed-count candidates and correlated/missing-data reporting.
- The new installer created `/home/bizon/projects/foldcraft-gpu-20260927/runtime-clean-01`
  from scratch, with no broken requirements. GPU float32/bfloat16 kernels and
  autodiff passed on the RTX PRO 6000 Blackwell.
- All five supervised three-recycle smoke cases passed: fixed count, replay,
  bounded exhaustion, VHH and long target. The final bookkeeping change was then
  tested in another fixed-count GPU run; attempt decisions matched all receipts.
- Predictions from the fresh environment/supervised driver matched the previous
  validated runtime exactly: maximum PAE and unrounded coordinate deltas **0**.
- A real GPU run with a **12-second** inference limit returned **124** in **17.21
  seconds**, including the five-second cleanup grace. GPU telemetry confirmed an
  active worker before timeout; the manifest was `timed_out`, completion was
  rejected, and no GPU compute process remained afterward.

Evidence: [reliability-2026-09-27](docs/review/evidence/reliability-2026-09-27/).
The complete isolated runtime and raw outputs remain on Bizon. No new accuracy
campaign was started and no scientific defaults were changed.

## Next correctness task

Finish R6: fingerprint the complete inference weight/conditioning bundle in
ordinary design manifests and scheduler resume signatures, and invalidate reuse
after in-place replacement. Controlled benchmarks and scorer caches already
record hashes, but that protection is not yet general to every design invocation.
Address this before expanding the accuracy benchmark.
