# Paired structural pilot — 27 September 2026

The correctness changes and corrected baseline are complete. This pilot does
**not** support promoting either experimental setting or changing the default
acceptance policy. It tests structural consistency, not experimental binding
accuracy.

## What ran

- Baseline tag `corrected-baseline-2026-09-27` at `d29ba3b`; pilot code at
  `3322ec6`. Exact legacy replay against the earlier GPU fixture passed before
  testing the opt-in objective.
- Twelve full 100/100/20-stage runs: PD-L1 and EGFR, Top7 scaffold, two seeds,
  three arms, two MPNN children per trajectory. All 24 candidates were retained.
- Separate AF2 model_1_ptm/model_2_ptm predictions with three recycles for each
  candidate, followed by ESMFold, single-sequence Boltz-2, and OpenMM scoring of
  the entire pool. No MSA service or assay labels were used.
- All 48 AF2 artifacts passed the independent sequence/chain, PAE, coordinate,
  hash, recycle-count and fixed-interface-residue audit. Source/input hashes
  stayed unchanged. All four baseline/temperature backbone pairs replayed
  byte-for-byte.

The [prespecified protocol](BENCHMARK_PLAN.md) defines the design and analysis.
MPNN siblings are averaged first: there are **four paired trajectory blocks**,
not 24 independent replicates. All arms produced eight distinct binder
sequences; two draws per block cannot characterize diversity well.

## Results

| Arm | AF2 pass, either policy | Template RMSD ↓ | ESMFold/design RMSD ↓ | Boltz ipTM ↑ | Epitope coverage ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2/8 | 4.244 Å | 3.344 Å | 0.523 | 6.88% |
| Normalized pairs | 0/8 | 27.295 Å | 11.925 Å | 0.793 | 23.36% |
| MPNN temperature 0.2 | 2/8 | 3.759 Å | 2.336 Å | 0.608 | 8.27% |

These are means of the four paired blocks, with equal weight per block. Arrows
indicate the desired metric direction, not established binding accuracy. Epitope
coverage is the fraction of specified hotspot residues with a cross-chain
heavy-atom contact within 4 Å. Clashes below use heavy-atom pairs within 2 Å.

### Normalizing contact terms needs a different formulation

The tested equal-sum, observed-pair-normalized fold/interface objective damaged
fold retention in all four blocks. Template RMSD increased by 14.825–28.490 Å,
despite increased hotspot contact coverage. This is a substantial regression for
fold-conditioned design; higher contact coverage alone would give the wrong
decision. Mean independent Boltz ipTM also rose, from 0.523 to 0.793, while
the intended fold was lost. This illustrates why interface confidence alone is
an insufficient success criterion. Normalization changes the relative gradient weights as well as the
units. This result rejects this particular setting, not all possible normalized
objectives. Keep the legacy objective as the production default.

### Higher MPNN temperature has an uncertain tradeoff

Temperature 0.2 reduced mean ESMFold/design RMSD from 3.344 to 2.336 Å, but
about 92% of that mean improvement came from one PD-L1 seed. Two blocks changed
by only 0.01 Å. Template RMSD improved in two blocks and worsened slightly in
two; mean interchain clash count increased from 2.375 to 3.375. There was no
change in AF2 acceptance. Boltz ipTM improved in three of four blocks, with
paired deltas +0.111, +0.035, +0.203 and −0.010 (EGFR seeds first, then PD-L1).
Its mean increased by 0.085. These are encouraging interface-consistency results,
but need more cases and repeated independent prediction seeds. Keep 0.1 as the default pending a broader comparison.

### Two-model acceptance did not change selection here

Both models passed the same four candidates and disagreed on none. Requiring
both therefore selected exactly the same pool as model_1 alone, giving no
evidence of better selection in this pilot. Keep the explicit per-model artifacts
and optional policy, but do not describe this as an accuracy improvement.

## Cost and limits

Generation, reference preparation and two-model validation took 2,118.2 seconds
(35.3 minutes). Peak sampled GPU memory was 7,304 MiB (7.13 GiB); telemetry was
sampled every two seconds. Warm PD-L1 jobs took roughly 107–117 seconds and warm
EGFR jobs roughly 198–205 seconds. First-use compilation for each input shape
and objective raised some job times; the shared persistent cache and randomized
order prevent interpreting total arm time as an intrinsic speed comparison.

The additional validation model's prediction calls totaled 15.16 seconds for
24 candidates. This excludes separately attributing model setup/compilation and
is not a cold-start overhead estimate. ESMFold evaluation took 25.35 seconds.
Boltz took 783.39 seconds (13.1 minutes), and OpenMM took 115.54 seconds.
Including pool construction and analysis, recorded stages total about 50.8
minutes. Evaluation memory was not sampled; the 7.13 GiB peak above applies to
generation and AF2 validation only.

All 24 candidates have ESMFold and Boltz ipTM measurements. All 24 OpenMM
minimizations converged within the fixed 500-iteration budget at RMS force ≤10
kJ/mol/nm. Mean interaction energies were −42.54, −39.37 and −29.51 kcal/mol for
baseline, higher temperature and normalized pairs, respectively. Neither variant
improved this diagnostic mean. Stock Boltz interface PAE is unavailable for all
24 candidates and remains unknown; no score was imputed. The final audit verified
all scorer joins and the hashes of all 24 selected Boltz confidence/structure
pairs.

These are two target families and one scaffold, with no held-out assay panel.
Boltz is a computational confidence proxy, not a binding label. Its per-candidate
seed is reproducible but includes the arm-specific candidate name; predictor
noise is therefore not coupled across arms, and this pilot has no Boltz repeat
samples. OpenMM interaction energy is a geometry diagnostic, not binding free
energy. No significance or generalization claim is warranted.

## Next experiments

1. Expand the frozen pool across target families, scaffolds and trajectories,
   keeping both AF2 model outputs. Compare selection policies and independent
   fold/interface consistency on the same candidates. Assay-labeled data is
   needed to test precision, recall and calibration for actual binding.
2. Rework loss normalization on a development subset: measure fold/interface
   gradient magnitudes, preserve a minimum fold-retention constraint, then freeze
   a small weight comparison before evaluating new seeds and held-out cases.
   Do not rescue this failed arm by selecting a favorable metric post hoc.
3. Retest MPNN temperature 0.1 versus 0.2 on that broader panel. Report fixed
   draws, unique-validation quotas and equal GPU-time comparisons separately,
   including clashes, fold retention, sequence diversity and independent scores.

Choose sample size and practical effect margins before the expanded run using
this pilot's variability and measured costs. No larger campaign was started.

## Evidence

The [evidence directory](docs/review/evidence/pilot-2026-09-27/) contains the
protocol, execution record, paired table, all candidate measurements, scorer
provenance, model receipts, artifact audit and GPU telemetry. The full raw run
remains on Bizon at `/home/bizon/projects/foldcraft-gpu-20260927/pilot-01`.

The code suite passed 190 CPU tests before GPU execution. The generation and
evaluation outcome records provide the real-GPU checks for this pilot.
