# FoldCraft fold-conditioning baseline

A reproducible benchmark of FoldCraft's fold-conditioned binder design: 6 folds
designed against PD-L1, scored by AF2 confidence + fold fidelity. This is the
**frozen reference** we measure changes (e.g. the `set_range` fix) against.

> **Caveat — this is OUR frozen config, not a bit-faithful reproduction of the
> paper.** The paper's exact template numbering and hotspot frame aren't
> recoverable from the repo (Supp.-table hotspots are PDB residue numbers, but
> `FoldCraft.py` consumes chain *positions*). We map them consistently and freeze
> the result. Our per-fold success spans 1–40% — a plausible regime for this task
> and broadly comparable to the paper's Fig. 3C, though our easy folds run higher
> (top7 40%, barrel 34%) than the paper's per-fold AF2 success (up to ~27%). That
> gap, with the hotspot-frame ambiguity above, is what we'd like to reconcile.

## Layout
- `prep_templates.py` — fetch-free template prep: trims each source PDB to its
  chain-A range, renumbers 1..N, maps Supp.-table hotspots to positions, writes
  `templates/<fold>.pdb` and `config.tsv`. (Source PDBs `templates/<PDBID>.pdb`
  are fetched from RCSB.)
- `config.tsv` — per-fold `template`, `binder_hotspots`, `binder_len`.
- `run_campaign.sh STAGES NDES NSAMP OUTROOT` — runs `FoldCraft.py` for all 6
  folds against PD-L1 (`target_hotspots=34-39,43-49,11-17`).
- `add_rmsd.py [runs_dir]` — computes RMSD-to-template (binder chain B vs
  `templates/<fold>.pdb`) for each design and writes it back into `results.csv`
  as an `rmsd` column. This is the only step that needs the raw design PDBs;
  run it where the designs live. Fails loudly if a referenced PDB is missing.
- `ipsae.py` / `add_ipsae.py [runs_dir] [pae_cutoff]` — ipSAE (Dunbrack 2025), a
  PAE-derived interface score that fixes ipTM's size bias. `ipsae.py` is the
  pure-numpy core (validated bit-for-bit against DunbrackLab/IPSAE); `add_ipsae.py`
  computes it offline from each design's saved PAE matrix
  (`aux['all']['pae']`, in the pickle — no GPU / no re-prediction) and writes an
  `ipsae` column. Default pae_cutoff 10.
- `score.py [runs_dir]` — per-criterion + combined success rate with Wilson 95%
  CIs. Reads the metrics **and the `rmsd` column** from `results.csv`, so it
  works from the committed CSVs alone, from any working directory.
- `runs/<fold>/results.csv` — committed per-fold metrics including `rmsd` (the
  reference numbers; scorable as-is). Raw designs (PDBs/pickles, ~GB) are **not**
  committed; they live on the GPU box and are backed up separately.

## Run
```bash
# from repo root, in the FoldCraft env
python baseline/prep_templates.py                                # (re)build templates + config
bash   baseline/run_campaign.sh 100,100,20 10 10 baseline/runs   # 6 folds x 100 designs (GPU)
python baseline/add_rmsd.py  baseline/runs                       # rmsd-to-template -> results.csv (needs raw designs)
python baseline/add_ipsae.py baseline/runs 10                    # ipSAE -> results.csv (needs design pickles)
python baseline/score.py     baseline/runs                       # success rates (works from committed CSVs)
```

## Frozen baseline results (100 designs/fold; success = pLDDT>0.8, ipTM>0.5, iPAE<0.35, RMSD-to-template<3.5)

Per-criterion pass counts out of 100, plus combined success rate (Wilson 95%
CI). ipSAE>.3 is *reported* (a less size-biased interface metric), not part of
the success gate:

| fold | pLDDT>.8 | ipTM>.5 | iPAE<.35 | ipSAE>.3 | RMSD<3.5 | **ALL** | success% [95% CI] |
|---|---|---|---|---|---|---|---|
| top7 | 79 | 52 | 41 | 38 | 86 | **40** | 40% [30.9–49.8] |
| barrel | 49 | 61 | 40 | 39 | 92 | **34** | 34% [25.5–43.7] |
| solenoid | 46 | 28 | 18 | 14 | 53 | **17** | 17% [10.9–25.5] |
| iglike | 20 | 32 | 20 | 12 | 83 | **14** | 14% [8.5–22.1] |
| ankyrin | 17 | 32 | 18 | 20 | 13 | **2** | 2% [0.6–7.0] |
| tim | 41 | 6 | 1 | 1 | 52 | **1** | 1% [0.2–5.4] |

**Reading:** fold fidelity (RMSD<3.5) is generally *not* the bottleneck — the
interface is. Fold-conditioning works (most folds hit RMSD<3.5 at high rates);
the weak link is interface confidence, worst on the large binders. ipSAE
sharpens it: tim's interface is genuinely hopeless (1/100 above 0.3), while
ankyrin is the reverse — ~20/100 have a confident interface by ipSAE, so its
failure is fold fidelity (RMSD<3.5 only 13/100), not binding.
