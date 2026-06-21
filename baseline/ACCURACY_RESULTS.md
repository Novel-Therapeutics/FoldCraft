# Accuracy A/B results — get_best, i_pae, false-positive penalty (top7, 2026-06-21)

All run on a Lambda A100, top7 fold, n=15 trajectories/arm (5 designs each = 75
designs/arm). Pass gate = AF2 plddt>0.8, iptm>0.5, ipae<0.35. "fold-fidelity" =
the design-trajectory recall cmap_loss (lower = better fold-conditioning).
`openmm_dE` = the family-neutral interface energy (more negative = better), the
independent check against AF2 self-flattery. Wilson 95% CIs in brackets.

## Accuracy #2 — get_best=True : NEUTRAL (don't ship)
Paired (same trajectory, save last vs best stage-3 iterate). AF2 passes: `last`
24/75 (32%) vs `best` 21/75 (28%), paired net -3. The best vs last iterate differ
by only ~0.015 cmap_loss (~1%) — the hard stage is near-discrete, so there is
nothing to propagate. Matches the analysis prediction.

## Accuracy #3 — i_pae loss term : INCONCLUSIVE (underpowered)
| arm | pass | fold-fid cmap_loss | openmm_dE median | dE<0 |
|-----|------|--------------------|------------------|------|
| cmap (baseline) | 18.7% [11-29] | 1.612 | -34.6 | 93% |
| ipae0.05 | 30.7% [21-42] | 1.710 | -36.8 | 97% |
| ipae0.1  | 21.3% [14-32] | 1.688 | -32.1 | 91% |
| ipae0.2  | 36.0% [26-47] | 1.762 | -42.8 | 91% |

There is a *suggestive* trend (more i_pae → higher pass rate and, for 0.2, a more
favourable OpenMM ΔE), at a modest fold-fidelity cost (+6-9% cmap_loss). BUT the
order is non-monotonic (0.1 < 0.05) and — decisively — **the experiment is
underpowered** (see below). The effect sizes are inside the run-to-run noise, so
this does NOT establish that i_pae helps. Promising enough to warrant a properly
powered follow-up; not shippable on this evidence.

## Accuracy #4 — false-positive penalty : CLEAR FAILURE
| arm | pass | fold-fid cmap_loss | openmm_dE median | dE<0 |
|-----|------|--------------------|------------------|------|
| fp0 (baseline) | 48.0% [37-59] | 1.596 | -40.3 | 100% |
| fp0.1 | 0.0% [0-5] | 2.286 | -0.8 | 56% |
| fp0.3 | 0.0% [0-5] | 2.365 | 1.1 | 44% |

The penalty (as scaled) **collapses the interface**: 0% pass, fold fidelity blows
up (+43-48% cmap_loss), and the OpenMM ΔE goes to ~0 / positive (no interface).
Exactly the "mis-scaled penalty fights recall and collapses the interface" risk
flagged up front. Rule out at these weights; a ~100x smaller weight is unexplored
and speculative.

## THE KEY FINDING: the A/Bs are underpowered
The two **recall-only baselines** — `cmap` (i_pae run) and `fp0` (fp run), the
SAME loss — gave **18.7% vs 48.0%** pass rates across two runs. That 2.5x swing is
pure run-to-run (trajectory-level) variance, and it is **larger than any i_pae
effect**. So at n=15 trajectories on one fold the pass-rate metric is too noisy to
detect modest changes. The only effect that clears the noise is the fp collapse
(huge), which is why that verdict is solid and the i_pae verdict is not.

**Implication for future accuracy work:** need n≥40-60 trajectories AND/OR multiple
folds, ideally a paired design (same trajectory seed across arms where the change
allows it). Single-fold n=15 only resolves large effects.

## Net
- Ship: nothing from these three (get_best neutral, i_pae unproven, fp fails).
- The real wins this round were on the **performance** side (Perf #1, shipped,
  ~24% campaign speedup) and **robustness/bug fixes** (PR #16).
- Worth a powered follow-up: i_pae (esp. ~0.1-0.2), with adequate n + multi-fold.
