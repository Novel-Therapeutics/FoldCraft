# FoldCraft fold-conditioning baseline

A reproducible benchmark of FoldCraft's fold-conditioned binder design: 6 folds
designed against PD-L1, scored by AF2 confidence + fold fidelity. This is the
**frozen reference** we measure changes (e.g. the `set_range` fix) against.

> **Caveat — this is OUR frozen config, not a bit-faithful reproduction of the
> paper.** The paper's exact template numbering and hotspot frame aren't
> recoverable from the repo (Supp.-table hotspots are PDB residue numbers, but
> `FoldCraft.py` consumes chain *positions*). We map them consistently and freeze
> the result. Numbers land in the paper's Fig. 3C regime (~0–27%), which
> validates the harness, but absolute values are not claimed to match the paper.

## Layout
- `prep_templates.py` — fetch-free template prep: trims each source PDB to its
  chain-A range, renumbers 1..N, maps Supp.-table hotspots to positions, writes
  `templates/<fold>.pdb` and `config.tsv`. (Source PDBs `templates/<PDBID>.pdb`
  are fetched from RCSB.)
- `config.tsv` — per-fold `template`, `binder_hotspots`, `binder_len`.
- `run_campaign.sh STAGES NDES NSAMP OUTROOT` — runs `FoldCraft.py` for all 6
  folds against PD-L1 (`target_hotspots=34-39,43-49,11-17`).
- `score.py [runs_dir]` — per-criterion + combined success rate with Wilson 95%
  CIs. Computes RMSD-to-template itself (binder chain B vs `templates/<fold>.pdb`)
  since `results.csv` doesn't include it.
- `runs/<fold>/results.csv` — committed per-fold metrics (the reference numbers).
  Raw designs (PDBs/pickles, ~GB) are **not** committed; they live on the GPU box
  and are backed up separately.

## Run
```bash
# from repo root, in the FoldCraft env
python baseline/prep_templates.py                          # (re)build templates + config
bash   baseline/run_campaign.sh 100,100,20 10 10 baseline/runs   # 6 folds x 100 designs
python baseline/score.py baseline/runs
```

## Frozen baseline results (100 designs/fold; success = pLDDT>0.8, ipTM>0.5, iPAE<0.35, RMSD-to-template<3.5)

Per-criterion pass counts out of 100, plus combined success rate (Wilson 95% CI):

| fold | pLDDT>.8 | ipTM>.5 | iPAE<.35 | RMSD<3.5 | **ALL** | success% [95% CI] |
|---|---|---|---|---|---|---|
| top7 | 79 | 52 | 41 | 86 | **40** | 40% [30.9–49.8] |
| barrel | 49 | 61 | 40 | 92 | **34** | 34% [25.5–43.7] |
| solenoid | 46 | 28 | 18 | 53 | **17** | 17% [10.9–25.5] |
| iglike | 20 | 32 | 20 | 83 | **14** | 14% [8.5–22.1] |
| ankyrin | 17 | 32 | 18 | 13 | **2** | 2% [0.6–7.0] |
| tim | 41 | 6 | 1 | 52 | **1** | 1% [0.2–5.4] |

**Reading:** fold fidelity (RMSD<3.5) is generally *not* the bottleneck — the
interface (iPAE especially) is. Fold-conditioning works (most folds hit RMSD<3.5
at high rates); the weak link is AF2 interface confidence, worst on the large
binders (tim 180 aa, ankyrin 125 aa). ankyrin is the exception — weak on both.
