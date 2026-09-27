# Expanded structural benchmark results — 27 September 2026

**Scientific defaults remain unchanged.** The holdout evidence does not satisfy the predeclared promotion criteria.

Completed 24 paired case/seed blocks, 96 candidates, 48 full design trajectories, 192 AF2 model predictions, 96 ESMFold evaluations and 192 Boltz predictions (two repeats per candidate).

Four targets, three scaffolds and two seeds expand the original pilot. PD-L1/PD-1 are grouped as one immune-checkpoint family. IFNAR was held out from outcome-driven method selection; no ranking threshold or temperature was tuned against it. Only one unseen family is available. There are no assay labels, so the endpoints measure structural consistency, not binding accuracy. Evaluation uses the bundled PDB constructs; full-length target context was not separately tested.

| Split | Arm | Blocks | Independent proxy pass | Template RMSD Å | ESMFold RMSD Å | Boltz ipTM | Epitope coverage | Clashes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development | baseline | 18 | 22.2% | 3.480 | 2.493 | 0.472 | 6.6% | 2.14 |
| development | mpnn_temp_02 | 18 | 19.4% | 3.450 | 2.731 | 0.461 | 7.2% | 1.97 |
| holdout | baseline | 6 | 25.0% | 3.644 | 2.634 | 0.387 | 24.3% | 1.75 |
| holdout | mpnn_temp_02 | 6 | 25.0% | 3.346 | 2.336 | 0.385 | 21.5% | 1.08 |

The independent proxy requires ESMFold confidence ≥70, ESMFold/design RMSD ≤3.5 Å and mean repeated Boltz ipTM ≥0.6. It is not a biological binding label. Siblings are averaged within each trajectory before arm means.

## Held-out paired changes (temperature 0.2 minus 0.1)

| Metric | Mean change | Conditional 95% interval |
| --- | ---: | --- |
| proxy_pass | 0.0000 | [0.0000, 0.0000] |
| rmsd | -0.2983 | [-0.9225, 0.0750] |
| esmfold_rmsd | -0.2983 | [-0.9700, 0.1500] |
| esmfold_plddt | 0.4333 | [-0.0250, 1.0500] |
| boltz2_iptm | -0.0019 | [-0.0512, 0.0277] |
| epitope_coverage | -0.0278 | [-0.1458, 0.0625] |
| interchain_clashes | -0.6667 | [-1.7500, 0.7500] |

These intervals resample case-level paired means after averaging the two seeds. They condition on the observed scaffold/target panel; one unseen family cannot support population-level generalization. See `family_means` for grouped outcomes.

## Fixed-quota ranking

| Split | Policy | Selected | Independent proxy pass | Template RMSD Å |
| --- | --- | ---: | ---: | ---: |
| development | af2_rank | 18 | 27.8% | 3.117 |
| development | fold_contact_rank | 18 | 27.8% | 3.083 |
| holdout | af2_rank | 6 | 16.7% | 3.570 |
| holdout | fold_contact_rank | 6 | 16.7% | 3.348 |

Both policies select one candidate per case/seed from the same four-candidate pool. The baseline ranks by AF2 ipTM. The candidate rule first minimizes violations of template RMSD, clashes and epitope coverage, then ranks by ipTM. Neither consumes ESMFold/Boltz outcomes. Coverage of ordinary single-/two-model confidence gates appears below and in JSON.

## Promotion decision

- temperature_promotion: retain existing default. Failed gates: enough_unseen_families, primary_gain, interval_positive.
- ranking_promotion: retain existing default. Failed gates: enough_unseen_families, primary_gain, interval_positive.

## Interpretation

Temperature 0.2 left the held-out primary proxy unchanged (25.0% versus 25.0%). Template and ESMFold RMSD means each improved by about 0.30 Å, but their conditional intervals include no improvement. Mean Boltz ipTM and epitope coverage did not improve. Both arms produced 48 unique binder sequences in this panel, so the higher temperature did not increase the observed unique-sequence count.

The ranking rule also left held-out proxy yield unchanged (16.7% versus 16.7%). It reduced mean clashes from 2.50 to 1.33, while ESMFold/design RMSD worsened from 2.54 to 3.13 Å. Better values for the features used to rank candidates did not translate into better independent proxy yield.

Zero-width primary-proxy intervals reflect zero changes in the observed case-level means; they do not establish zero uncertainty for new targets or binding assays.

## Confidence-gate coverage (descriptive)

| Split | Gate | Selected / pool | Proxy-positive fraction among selected |
| --- | --- | ---: | ---: |
| development | single_model_pass | 11 / 72 | 63.6% |
| development | two_model_pass | 10 / 72 | 70.0% |
| holdout | single_model_pass | 8 / 24 | 50.0% |
| holdout | two_model_pass | 1 / 24 | 0.0% |

On the held-out panel the two-model gate narrowed selection from eight candidates to one, and the surviving candidate did not pass the independent proxy. This supports keeping the two-model acceptance policy optional rather than promoting it from its development-set behavior. These small descriptive counts do not establish binding precision or population-level recall.

No post-hoc threshold change was used to rescue either method. Before independent scoring, the analyzer was corrected to enforce the fold/clash ranking limits already specified in the frozen plan. The original source, both hashes and the timing are retained in `analysis_implementation_correction.json`. The normalized-loss arm rejected by the earlier pilot was not rerun.

## Cost

| Stage | Allocated GPU/runner wall minutes |
| --- | ---: |
| generation | 132.96 |
| pool | 0.06 |
| esmfold | 0.91 |
| boltz | 60.65 |
| analysis | 0.01 |

Generation used up to two concurrent owned jobs; Boltz used two independent scorer processes. These are campaign wall times, not additive per-candidate GPU costs. Equal-draw quality is measured; equal-GPU-time search efficiency is not claimed.

All candidates, including rejects, were independently scored. Paired Boltz seeds exclude the arm identity. Exact paired backbone hashes, per-model artifact checks, scorer-signature checks and raw Boltz selection hashes passed independent audits. Evaluator package/source fingerprints match before and after the campaign.

Evidence: [improvements-2026-09-27](docs/review/evidence/improvements-2026-09-27/). AF2/Boltz structures and full optimization histories remain on Bizon. ESMFold retains scores and checkpoint/input provenance, but not monomer coordinate files; reproducing its RMSDs requires rerunning the recorded evaluator. More untouched families and assay-defined labels are needed before a scientific default or binding-accuracy claim can be promoted.
