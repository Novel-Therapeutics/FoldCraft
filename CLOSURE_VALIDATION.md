# Remaining correctness work: closure validation

27 September 2026. This pass follows the checkpoint-provenance fix `d95f95a`.
It closes the VHH convention mismatch and the missing main-path optimization
history, and repairs additional experimental-entry-point defects. It does not
claim that all possible bugs have been eliminated.

## VHH convention migration (C4)

The three differing archived maps are explained exactly by binder positions
`27–35,56–60,103–117`, versus current `26–35,55–59,102–116`. EGFR already uses the
current convention. `--vhh_convention` now selects the convention explicitly;
preflight resolves the positions once and runtime consumes that selection. The
manifest records both the convention and positions. Current remains the default.

All archived arrays remain byte-identical. Golden fixtures verify their file
hashes, reproduce every map/mask element-for-element, and freeze hashes of every
current map/mask. The GPU historical PD-L1 run also reproduced both archived
arrays exactly. See [VHH_CONVENTIONS.md](examples/VHH_CONVENTIONS.md).

## Optimization history (engineering portion of E8)

Each completed trajectory writes `optimization/traj_N.json` with all optimization
iterations, stage boundaries, stage iteration, metrics, actual model names,
recycles and the design RNG seed. The existing SHA-256 seed protocol derives
validation/MPNN streams independently; this change does not consume extra model
random numbers or change optimization. Missing/nonfinite metrics or incomplete
histories cannot publish a valid completed run.

Run manifests hash these histories. Scheduler merges copy and hash histories
from every chunk, including trajectories whose candidates were rejected. CPU
regressions verify corruption invalidates completion. A real default trajectory
recorded all **100 + 100 + 20 iterations**, and its merged output retained the
complete history and resumed successfully.

Histories publish when a trajectory completes. A worker killed mid-trajectory
may only have its text log; step-by-step restart and transactional logging of
unfinished optimization are not implemented. This is distinct from the complete
history required of successful runs. Historical outputs are not backfilled.

## Experimental binder defects and safeguards

`test/FoldCraft_binder.py` had a reproducible `UnboundLocalError`: it read `os`
before a function-local import. Existing tests had exercised help/control parsing,
not the execution loop. New CPU integration tests run that loop through success,
physical rejection and budget exhaustion.

The repaired entry point also:

- rejects VHH/template/binder-conditioning options that it previously ignored;
- offers CPU preflight, explicit AF2 paths and recorded Python/AF2/MPNN/PyRosetta
  seeds; selects AF2 models explicitly, including the monomer validation;
- uses the shared process-group supervisor and records timeout/failure/exhaustion
  separately from completion;
- publishes CSVs atomically and rejects nonfinite physical metrics;
- verifies exact accepted count, finite result metrics and predicted/relaxed
  artifacts before recording completion, saving their hashes;
- rejects a worker's successful exit when verified results are absent.

This remains a separate experimental workflow. Its manifest is not a supported
main-pipeline scheduler-resume record. Installing BindCraft alone does not install
or license PyRosetta. Neither dependency exists in Bizon's tested FoldCraft
runtime, so numerical relaxation/interface-scoring validation is **blocked on an
appropriate existing licensed environment**. CPU doubles cannot substitute for
that validation. The user has been asked for its location.

## Validation evidence

- Final CPU suite: **242 passing tests**, locally and in the Bizon runtime.
- Main GPU smoke suite: all five cases passed (fixed, replay, bounded exhaustion,
  current VHH and long target).
- Fixed/current-VHH/long-target predictions match the preceding validated outputs
  exactly: maximum coordinate and PAE differences **0**.
- Historical VHH conditioning reproduces the archived PD-L1 map and mask exactly.
- A real full scheduler trajectory and merge preserve all 220 iteration records;
  verified resume recognizes the completed chunk.
- Real experimental CLI tests record the missing-dependency failure and terminate
  a deliberately hung import through the actual supervisor, preserving timeout
  status and returning 124. No physical-scoring result is fabricated by this test.
- Undefined-name/local-variable and selected high-confidence static correctness
  checks pass. Reviewed loop-closure warnings refer to callbacks consumed
  synchronously or intentionally reading a live success count, not deferred jobs.

Compact receipts, logs, iteration records and reproduction scripts are in
[closure-2026-09-27](docs/review/evidence/closure-2026-09-27/). Large raw predictions
remain on Bizon. No accuracy experiment or scientific-default change was made.

## Remaining boundary

The original reproduced main-path defects now have fixes or explicit early
rejection of unsupported inputs. Full experimental PyRosetta validation remains
open. General partial-chain repair/alignment, other GPU platforms, mid-trajectory
restart, performance optimizations, family-level uncertainty and held-out binding
assays are capability/research work; this pass does not claim those are complete.
The current confidence gates remain computational proxies for binding.
