# Accuracy A/B results — get_best, i_pae, false-positive penalty (top7, 2026-06-21)

**Interpretation updated at integration:** these are exploratory June 2026
measurements, not validated accuracy improvements. The five sequences from each
trajectory are related observations; the sequence-level Wilson intervals below
do not account for this dependence. OpenMM interaction energy is an additional
proxy, not a ground-truth binding label. See [INTEGRATION.md](INTEGRATION.md).

All run on a Lambda A100, top7 fold, n=15 trajectories/arm (5 designs each = 75
designs/arm). Pass gate = AF2 plddt>0.8, iptm>0.5, ipae<0.35. "fold-fidelity" =
the design-trajectory recall cmap_loss (lower = better fold-conditioning).
`openmm_dE` = the OpenMM interface interaction energy (more negative means
lower energy under this scoring protocol), an additional molecular-mechanics proxy. Wilson 95% CIs in brackets.

## Accuracy #2 — get_best=True: no benefit demonstrated in this pilot
Paired (same trajectory, save last vs best stage-3 iterate). AF2 passes: `last`
24/75 (32%) vs `best` 21/75 (28%), paired net -3. The best vs last iterate differ
by only ~0.015 cmap_loss (~1%). This does not establish equivalence across folds
or targets. The arms share the design trajectory, but downstream MPNN sampling
and validation randomness are not explicitly paired.

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
evidence of poor repeatability, and it is **larger than any observed i_pae
effect**. Without complete seed/configuration manifests, variance and other run
differences cannot be separated. This single-fold pilot does not reliably
resolve modest changes. The only effect that clears the noise is the fp collapse
(huge), which is why that verdict is solid and the i_pae verdict is not.

**Implication for future accuracy work:** use multiple folds/targets, matched
seeds, and trajectory-level uncertainty. The earlier suggestion of 40–60
trajectories is a planning estimate, not a power calculation. Choose sample size
from a prespecified effect size and pilot trajectory-level variance.

## Net
- Ship: nothing from these three (get_best has no demonstrated benefit, i_pae is
  unproven, and the tested fp formulation/weights fail).
- The fp implementation penalizes all off-mask pairs, including unconstrained
  regions. Its failure does not rule out a normalized, observation-aware penalty.
- The real wins this round were on the **performance** side (Perf #1, shipped,
  ~24% campaign speedup) and **robustness/bug fixes** (PR #16).
- Worth a powered follow-up: i_pae (esp. ~0.1-0.2), with adequate n + multi-fold.
