# Inference provenance and safe resume

27 September 2026. This closes review item R6 for the supported main CLI and
scheduler. Scientific defaults and prediction algorithms are unchanged.

## Recorded identity

Every production design run records an `inference_bundle` in `run.json`:

- SHA-256, size, absolute lookup path and resolved path of each available
  template-capable AF2 checkpoint (`model_1_ptm` and `model_2_ptm`). Discovery
  follows the installed loader's four-path precedence. Requested validation
  models must exist; adding an optional checkpoint changes the identity too.
- The actual `v_48_010.pkl` used by the selected MPNN variant, resolved in the
  worker's ColabDesign installation, plus the relevant loader source files.
- `framework/vhh.npy` when VHH conditioning is enabled. Non-VHH runs do not
  consume this file. Input PDB hashes, canonical inputs and generated contact-map
  artifact hashes remain recorded by the existing run protocol.
- Selected package versions, repository and package locations, and effective
  model/conditioning options. AF2 constructors receive the fingerprinted model
  set explicitly, and the returned model set must match.

The supported layout is the pinned ColabDesign runtime described in
[RELIABILITY_FIXES.md](RELIABILITY_FIXES.md). This is checkpoint and conditioning
provenance, not a complete operating-system/container image fingerprint or a
claim that arbitrary future dependency layouts are supported.

## Checks and compatibility

Each invocation hashes file contents afresh. A scheduler campaign discovers the
bundle with its selected `--python`, without importing ColabDesign or accessing a
GPU, and includes that identity in every chunk signature. The worker independently
captures and compares its identity before creating a run directory. Merged runs
retain the identity of each contributing chunk.

Within a captured campaign, fast checks verify lookup resolution, file set,
symlink target, device, inode, size, nanosecond mtime and ctime. Model construction
is checked before and after loading. Full hashes are also checked at worker entry
and before completion. A detected mutation prevents a valid completion record.
These guards avoid repeatedly hashing large checkpoints during scheduler polling.
They assume normal filesystem change tracking, not an adversarial filesystem.

Stat observations are separate from scientific identity: touching an unchanged
file between invocations permits reuse after a fresh hash. Replacing bytes at the
same path, even with the same size and restored mtime, invalidates reuse. An
incompatible completed output is preserved and requires a fresh output root.

Historical runs remain readable and artifact-verifiable, but runs without a
matching bundle cannot satisfy scheduler resume checks. The internal
`main(supervised=False)` adapter is for CPU inference doubles; it does not certify
production provenance. Experimental drivers and standalone scorers retain their
separate protocols.

## Scheduler usage

```bash
python baseline/scheduler.py config.tsv \
  --repo /path/to/FoldCraft --repro /path/to/new-runs \
  --python /path/to/runtime/bin/python --data-dir /path/to/af2-weights \
  --mpnn-weight soluble --gpus 0
```

Relative `--data-dir` paths resolve against `--repo`. MPNN weights come from the
selected Python environment, not the AF2 data directory. For a CPU-only resume
check, add `--dry-run --verify-runtime`. Ordinary `--dry-run` remains usable
without weights/GPU dependencies and labels the runtime unverified; its chunks
all require verification before they can be considered complete.

## Validation

- **222 CPU tests passed locally and on Bizon.** New regressions exercise each
  checkpoint/conditioning file, same-size/same-mtime replacement, unchanged-byte
  touches, lookup shadowing, optional models, selected-interpreter discovery,
  legacy manifests, scheduler/worker mismatch and mutation during loading/hashing.
- Five supervised GPU smoke cases passed: fixed count, replay, bounded exhaustion,
  VHH and long target. Replay produced zero PAE and coordinate differences.

- A real scheduler campaign completed one default 100/100/20-stage trajectory
  with five MPNN candidates in 136.73 seconds. Worker and scheduler identities
  matched, and the merged run retained chunk provenance.
- Unchanged weights resumed successfully. Replacing the first byte of a private
  MPNN copy, retaining size and mtime, made verified planning report zero completed
  chunks and actual execution refuse the existing output. Saved run manifests and
  results stayed byte-identical. Restoring the checkpoint allowed resume again.
- The smoke outputs also exactly matched the prior validated runtime (zero PAE
  and coordinate deltas). No GPU compute processes remained after verification.

Compact receipts, CPU logs and the reproducible scheduler test are recorded under
[provenance-2026-09-27](docs/review/evidence/provenance-2026-09-27/).
