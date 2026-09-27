# Experimental

Code here is **experimental** and **not part of the validated FoldCraft pipeline**.
It is not covered by the paper's benchmarks and may change or break without notice.
For published results, use `FoldCraft.py` in the repository root.

## Contents

- `FoldCraft_binder.py` — length-variable de novo binder design (exploratory
  linear / miniprotein binder runs, e.g. Nipah, KEAP1).
- `bindcraft_deps.py` — locates a BindCraft checkout and exposes the PyRosetta
  helpers (`pr_relax`, `score_interface`) and `DAlphaBall.gcc` that the binder
  pipeline reuses.
- `install_foldcraft_binder.sh` — clones BindCraft into `test/BindCraft` and
  verifies it.

## Dependencies

`FoldCraft_binder.py` reuses helpers from
[BindCraft](https://github.com/martinpacesa/BindCraft), which is not
pip-installable. The main `install_foldcraft.sh` does **not** install it (the
validated `FoldCraft.py` doesn't need it). Install it once for this pipeline:

```bash
bash test/install_foldcraft_binder.sh
# optionally pin a revision:
BINDCRAFT_COMMIT=<sha> bash test/install_foldcraft_binder.sh
```

This clones BindCraft to `test/BindCraft` (git-ignored). Alternatively, point
`bindcraft_deps.py` at an existing checkout with `export BINDCRAFT_PATH=/path/to/BindCraft`.

## Usage

These scripts import `biopython_utils` from the repository root, so run them
**from the repo root**:

```bash
python test/FoldCraft_binder.py --help
```

## Execution controls and validation limits

The supported experimental mode requires `--sample`. VHH/template/binder-map
options are rejected rather than silently ignored; use the main driver for those
workflows. `--preflight_only` validates inputs without importing GPU/PyRosetta
packages. `--data_dir` selects AF2 weights, `--seed` records reproducible Python,
AF2, MPNN and PyRosetta seeds, and `--timeout_minutes` (default 360) supervises the
worker process group. Timeouts return 124 and preserve partial artifacts.

`experimental_run.json` distinguishes running, failed, exhausted, timed-out and
complete runs. Completion requires the exact accepted count, finite metrics and
all predicted/relaxed artifacts, whose hashes are saved. Result CSVs publish
atomically. This manifest is separate from the main pipeline's completion and
scheduler protocol; experimental runs cannot be used as main-pipeline resume
chunks. Interrupted trajectories restart in a fresh output directory.

CPU doubles exercise the complete selection/artifact flow. Real-process tests
exercise dependency failure and timeout cleanup. Full numerical PyRosetta/BindCraft
validation still requires an installed licensed environment; the main GPU suite
cannot certify this separate physical-scoring path. See
[CLOSURE_VALIDATION.md](../CLOSURE_VALIDATION.md) for current evidence.
