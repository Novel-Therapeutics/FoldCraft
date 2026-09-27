# CPU correctness fixes — 26 September 2026

Implemented on `codex/correctness-and-validation`, based on `e485f324`.
Historical benchmark CSVs and the integration-baseline tag are unchanged.
These changes establish engineering contracts; they do not demonstrate improved
binding accuracy or finish the GPU integration gate in `docs/review/CHANGE_PLAN.md`.

## Implemented

- **Input/mapping:** CPU preflight validates controls, selections, chain backbone
  completeness and contact maps. Original PDB residue IDs are mapped explicitly
  to canonical model positions, including numbering offsets. Gaps, insertion
  codes, multiple models and incomplete backbones fail early. MSE and first-record
  alternate locations are supported. Float32 probability roundoff in the bundled
  VHH map is retained; its current conditioning convention is recorded.
- **MPNN/model contract:** terminal residues are included. An empty mutable set
  returns one unchanged sequence without calling MPNN. Validation explicitly uses
  `model_1_ptm`, one model, three recycles, matching actual historical execution.
- **Run durability:** exclusive output directories, atomic partial checkpoints,
  running/failed/exhausted/complete manifests, source/input hashes, candidate
  identities and artifact verification. Success sampling stops at
  `--max_trajectories` (default 1000), preserves partial results and exits nonzero
  when exhausted. Attempted trajectories and evaluated candidates have metric
  records, including rejections. `--seed` controls recorded design/MPNN/prediction
  streams; omitting it generates and records a replay seed. `--data_dir` resolves
  weights explicitly (default: this checkout).
- **Scheduling:** live defaults are under ignored `runs/`. CSV-only archives never
  count as completed work. Resume requires matching input/config/code/seed
  signatures and intact artifacts; incompatible completed output is refused.
  Incomplete attempts are retained under `.failed-*` before retry. Merges publish
  by directory rename; a scheduler lock prevents competing launchers. Solo retries
  reserve their GPU until exit. Cleanup terminates only owned process groups.
  Shell campaigns return nonzero on failure.
- **Scoring:** strict complete-chain RMSD replaces prefix truncation; unsupported
  correspondence fails instead of returning a flattering score. Clash counting
  uses a selected model rather than pooling alternative conformations. PAE must be
  finite, nonnegative and square with a valid split; multiple model arrays cannot
  silently lose all but model zero. OpenMM skips minimization for zero iterations
  and rejects nonfinite energies and extra-chain complexes. Each Boltz invocation
  gets an isolated directory and a selected-confidence receipt. GPU scorer
  sidecars key reuse by inputs, source/helper code, settings and package versions;
  changed settings invalidate old values. Publication merges by candidate identity
  under locks and atomically replaces the CSV. Legacy CSV-only scores are
  recomputed when a scorer is explicitly invoked on them.
- **Benchmark/entry points:** consensus reports **155 known passes + 27 unknown**,
  or 155–182/1000, rather than an exact 155/1000 rate. A/B fold hotspots come from
  the production reproduction config; new experiments share stage seeds, randomize
  arm order and record their protocol. Both notebooks use the shared CLI with CPU
  preflight. Experimental binder help/mode errors occur before model imports;
  length bounds are inclusive, output paths use filesystem APIs, best/last
  artifacts agree and sampling has an attempt budget. Installation pins the
  inspected ColabDesign commit and runs a JAX GPU check before weight downloads.

## Validation

The baseline had 113 passing CPU tests. The updated suite has 156 passing tests;
old tests that treated empty CSVs/template copies as completed jobs were replaced
with the manifest contract tests. Coverage includes:

- Offset PDBs, malformed ranges/maps, missing backbone atoms and valid VHH roundoff.
- Full driver execution with CPU inference doubles: fixed count, success quota,
  zero-success exhaustion, saved model identity and failed-run status.
- Interrupted serialization, stale scorer snapshots, cache invalidation, partial
  merge cleanup, scheduler child cleanup and full-lifetime solo reservations.
- Truncated/internal-missing CA coordinates, rectangular/invalid PAE, stale Boltz
  confidence, OpenMM zero iterations and cross-model clash counting.
- Ordered notebook execution through the shared inference adapter and all three
  A/B drivers with mock predictors, including paired seeds and saved PAE artifacts.
- Shell campaign failure propagation using a fake failing Python executable.

Python syntax checks and shell syntax checks pass. No GPU job, model installation,
weight download or new experiment was run. Bizon was not used by this patch.

Run tests in a CPU environment containing numpy, scipy, pandas, biopython,
pytest and gemmi:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
python FoldCraft.py --help
python baseline/scheduler.py baseline/repro_config.tsv --dry-run
```

## GPU gate and remaining plan items

- Test the real ColabDesign/MPNN adapters and seeded replay on the intended GPU;
  mocks establish call/control-flow contracts, not numerical equivalence. The old
  CUDA/JAX installation stack is not yet certified for Bizon's Blackwell GPU.
- Enable two-model validation only with coupled per-model structures/PAE/metrics
  and an explicit ensemble decision. This patch does not relabel historical
  single-model results as an ensemble.
- Pin/checksum the complete runtime and model checkpoints, beyond the source
  commit/package versions recorded here. Scorer cache identity does not yet detect
  manual replacement of weight files under unchanged package/model identifiers.
- Add OpenMM convergence/force QC and calibrated physical acceptance criteria;
  finite but enormous energy values are not certified as meaningful by this patch.
- Strict RMSD guards support complete continuous chains. General sequence-based
  alignment with explicit partial-coverage reporting remains future work.
- A/B summaries remain exploratory. Add paired trajectory/target-family confidence
  intervals and independent holdout evaluation before accuracy claims. Historical
  runs do not acquire paired seeds retroactively.
- Mid-trajectory restart, a wall-time watchdog and saving every rejected prediction
  structure remain separate work. Candidate metric records are retained; failed
  scheduler attempts restart in a fresh directory while completed chunks resume.
