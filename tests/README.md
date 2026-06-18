# FoldCraft test suite

This suite was added to give the project a regression safety net before we
refactor or tune the design pipeline. Upstream FoldCraft shipped **no tests**.

## What is covered

These are **characterization tests**: they assert the *current* behavior of the
GPU-free helper functions so that refactors and performance/accuracy work do not
silently change residue selection, interface detection, or scoring.

Covered today (all CPU-only, run on a laptop in seconds):

- `biopython_utils.set_range` — hotspot/mask range parsing
- `biopython_utils.validate_design_sequence` — composition notes
- `biopython_utils.calculate_clash_score` — clash counting (real PDB fixtures)
- `biopython_utils.hotspot_residues` — interface detection (synthetic complex)
- `biopython_utils.target_pdb_rmsd` — CA RMSD after superposition
- `cmap_utils` — fold-conditioned cmap assembly + binder_mask (differential test vs an independent re-implementation)
- `bindcraft_deps` — locating the BindCraft checkout used by the binder pipeline
- `baseline/ipsae.py` — ipSAE interface score (differential test vs the reference DunbrackLab/IPSAE algorithm)
- `baseline/score.py` — the baseline scorer, run as a subprocess (reports ipSAE, fails loud on missing/inconsistent columns)
- portability — the committed baseline scripts contain no machine-specific paths

## Fixed behaviors the tests now lock

- **`set_range` is inclusive of both endpoints.** `"39-45"` -> `39..45`,
  `"80-81"` -> `[80, 81]`, `"80-80"` -> `[80]`. (Previously the upper bound was
  exclusive, silently narrowing every hotspot/mask window by one residue; that
  was accuracy-relevant since it decides which residues the cmap loss conditions
  on. The fix's effect on design success is measured via a baseline re-run.)
- **`binder_mask` indexing is consistent with hotspots** (residue R -> position
  R-1, with out-of-range residues raising `ValueError`). Adversarial tests cover
  terminal ranges (which used to crash after the inclusive fix) and out-of-range
  residues.

## What is NOT covered yet

The two design drivers (`FoldCraft.py`, `FoldCraft_binder.py`) still need a GPU,
AlphaFold2 weights, and `colabdesign` to run end-to-end, so their `main()` flow
isn't exercised here. Their pure pieces have been extracted and are covered: the
fold-conditioned cmap assembly (`cmap_utils.py`) and the helper functions
(`biopython_utils.py`). `FoldCraft_binder.py`'s BindCraft dependency is now
declared and located via `bindcraft_deps.py` (installed by `install_foldcraft.sh`),
not a side-loaded `from BindCraft.functions import *`. Further extraction of the
design loop is tracked separately.

## Running

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python -m pytest
```
