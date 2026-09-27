# Model objectives and evaluation review — integrated FoldCraft baseline

Reviewed branch: `codex/correctness-and-validation`, commit `e485f3244e47464ae6083c193f234e91c099e0c5`, September 26, 2026.

Repository root for the path:line references below: `/Users/andreivolgin/PyCharmProjects/FoldCraft-Novel-Therapeutics`.

Evidence: source inspection; lightweight CPU probes in `current_model_probes.py` and `current_model_probes.json`; committed-table audit and trajectory bootstrap independently performed by the root reviewer in `current_data_audit.json` and `current_ab_summary.json`. No GPU model inference or molecular-mechanics scoring was rerun. The ColabDesign behavior cited below comes from the source snapshot at `/private/tmp/foldcraft-review-colabdesign`, commit `e31a56fe1d9b4de25c8697f3a28b75892941cc72`; pin this dependency before relying on that behavior in a new run.

Severity: P1 = fix before expensive accuracy campaigns or using a reported comparison to choose defaults; P2 = important experimental or robustness limitation; P3 = lower-priority validation hardening. A hypothesis is explicitly labeled as such; no proposed objective change has demonstrated a general accuracy gain.

## Findings by domain and severity

### Validation correctness

**M01 — P1, confirmed: intended two-model validation runs one model, and a simple model-count change would create inconsistent score/structure provenance.**

- Locations: `FoldCraft.py:363`, `FoldCraft.py:452`, `baseline/score_af2.py:35-40`, `baseline/ab_get_best.py:119`; downstream `baseline/add_ipsae.py:69`, `baseline/add_rmsd.py:44-50`, `baseline/score_esmfold.py:69`.
- The fresh validation models inherit `opt.num_models=1`. Listing model_1_ptm and model_2_ptm is only an allowed-model list; `_get_model_nums` selects its first member when prediction uses `sample_models=False`. Source: ColabDesign `af/model.py:48`, `af/design.py:62-78,276-302`.
- With two models enabled, ColabDesign averages floating aux/log fields (`af/design.py:97-106`) and writes all model structures into the PDB (`af/utils.py:71-86`). Current ipSAE selects PAE[0]; structural scorers select PDB model[0]. Enabling two models without updating this would combine mean confidence with first-model fold/interface scores.
- Fix: explicitly define model identities/count, output per-model metrics and structures/PAE with a shared candidate/model ID, then implement a documented ensemble aggregation/acceptance policy. Record actual models run, not just requested names. Unit-test selection and aggregation using two deliberately disagreeing synthetic models; perform a small GPU parity check later. Do not relabel historic tables as two-model results.

**M02 — P1, confirmed missing check: production acceptance does not test the central fold/geometry requirements.**

- Locations: `FoldCraft.py:414,454`; offline benchmark gate `baseline/score.py:21-25`.
- Main sampling accepts pLDDT/iPAE/ipTM only. No explicit intended-template fold retention, epitope contact coverage, steric clash, target conformation, or polymer integrity check enters acceptance. Offline scoring adds template RMSD, but that does not control actual sample output.
- Qualification: the existing iPAE *does* depend on the requested hotspot, so this is not a wholly epitope-independent score. ColabDesign `af/loss.py:42-57,251-259` symmetrizes PAE, divides by 31, then averages binder-to-hotspot pairs. It is still confidence, not measured contact geometry. `baseline/score_af2.py:12` incorrectly describes its scoring as epitope-agnostic.
- Fix: one candidate-evaluation function shared by production, A/B drivers, and reports. Keep a separately named historical confidence gate. Report fold, interface, and structural-integrity criteria separately, then apply a versioned combined gate. Store accepted and rejected attempts plus reason codes. Before imposing new thresholds as defaults, characterize retention/yield on a complete frozen candidate panel and calibration controls.
- Corrected evidence: Ig-like reproduction has 187/200 RMSD passes and 66/200 combined passes; the old 9/200 statement was wrong. This cohort does not establish widespread Ig-like fold loss. The missing production check remains a real omission independent of that old erroneous count.

**M03 — P1 for measurement, confirmed: RMSD can report an apparently excellent partial match while ignoring missing or mismatched residues.**

- Locations: `baseline/add_rmsd.py:30-50`; `baseline/score_esmfold.py:62-85`; the old `biopython_utils.py` RMSD helper should be unified with the eventual fix.
- Both scorers use `min(len(a),len(b))` and align prefixes. Internal missing CA atoms shift correspondence; terminal missing residues are silently discarded. Template extraction chooses its first chain rather than a saved explicit chain mapping. This can produce an erroneous score instead of failing or reporting low coverage.
- CPU probe: a five-residue reference scored against only its first three CA coordinates reports RMSD 0.0. This is a demonstrated edge case, not proof the missing raw historical structures were mis-scored.
- Fix: use the run's residue map, explicitly validate model/chain/sequence/length identity, mask only matched atom pairs, record coverage and excluded residues, and require coverage for a fold decision. For designed sequences, sequence identity to the template is not expected; map by the construction provenance rather than a naive sequence-identity threshold. Add robust global fold similarity/topology diagnostics alongside RMSD, especially when loops are intentionally free.

### Scientific data integrity and benchmark implementation

**M04 — P1, confirmed in code and committed data: missing independent scores are silently counted as failed consensus, and the resulting count is labeled exact.**

- Locations: `baseline/consensus.py:11-13,43,53-60`; data `baseline/repro/top7/results.csv`.
- NaN Boltz scores compare false at `d['boltz2_iptm'] > 0.5`. Code then reports `(AF2 & Boltz).sum()/all_designs` as an exact consensus success rate.
- Root audit: 188/1,000 FoldCraft rows pass AF2, but 27 of those rows lack Boltz scoring. The known consensus count is 155; without those scores, the possible count is 155–182, not an exact 15.5%. All 27 missing AF2 passers are Top7. Only 225/1,000 FoldCraft rows have Boltz scores and 188 have ESMFold scores. BoltzProt has 100/200 of each. Scored-subset agreement is not a population-wide rate when selection depends on AF2 scores.
- Fix: represent pending, failed, and valid scores separately; refuse an exact conjunctive count when an otherwise-eligible candidate lacks the second score; show bounds and score coverage. Complete all eligible second-stage scores for an exact conjunction. Use prespecified stratified random scoring of both AF2 passers and failures when estimating model agreement/calibration, with sampling weights if needed. Selective scoring is valid for cost-saving conjunctive gating only when completeness and the estimand are explicit.

**M05 — P1, confirmed by CPU probe: Boltz rescoring can select a stale confidence file from a previous run.**

- Locations: `baseline/score_boltz2.py:75-90,140-150`.
- Each candidate has a reusable work directory. The script recursively globs every `confidence_*.json` under that directory, selecting the highest confidence rather than files belonging to the current invocation. Changing diffusion-sample count, input, or retaining old nested output can leave stale files that win.
- CPU probe: fake the new subprocess output with ipTM 0.2 while a previous output has 0.99; `run_boltz2` returns 0.99. No GPU is needed to reproduce the wrong file selection.
- Fix: immutable invocation directory keyed by input sequence/target and scorer manifest, or explicit returned output manifest; select only files produced by the current invocation. Record chosen sample ID and sample budget. Top-of-K scoring must use a fixed K across comparator arms or report its additional budget.

**M06 — P2, confirmed API mismatch: `--min-iters 0` performs unlimited minimization instead of raw scoring.**

- Locations: `baseline/score_openmm.py:107,131-132`.
- CLI help promises raw structure scoring at zero, but unconditionally calls `LocalEnergyMinimizer.minimize(..., maxIterations=0)`. OpenMM defines zero as convergence without an iteration limit. This changes the scientific protocol and can unexpectedly increase runtime.
- Fix: skip the minimizer when raw scoring is requested; expose convergence/unlimited minimization as a separate explicit option. Record input/post-minimization energies, displacement, force/convergence diagnostics, iteration cap, platform, force-field version, and success status.
- Root audit found one Top7 `openmm_dE ≈ 1.3313e19` entry. The raw structure is absent, so its cause cannot be diagnosed locally. Treat such a result as a QC failure requiring investigation; do not silently use it as evidence of nonbinding or drop it without accounting for failed measurements. The median may be numerically robust to an outlier, but that does not validate the underlying energy calculation.
- Primary source checked: [OpenMM LocalEnergyMinimizer API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html).

**M07 — P1 before new comparisons, confirmed: A/B provenance, blocking, and uncertainty are inadequate for small effects.**

- Locations: `baseline/ab_get_best.py:156-180`; `baseline/ab_loss.py:8-9,82-101`; `baseline/ab_loss_fp.py:105-122`; `baseline/score.py:31-37,72-75`; `baseline/consensus.py:22-29,57`.
- Best/last share a design trajectory, but use sequential unpaired downstream MPNN and validation RNG streams. Loss A/Bs use unrecorded random seeds and complete one arm before the next. Different objectives do not prevent matching initialization seeds and execution blocks; the claim that they must be unpaired is a methodological choice, not a necessity.
- Five MPNN children share one trajectory. Sequence-level Wilson intervals and pooled per-sequence tests ignore that dependence. For cross-target generalization, targets/families add another sampling level. The two nominally identical baselines' 18.7% vs 48% difference cannot be assigned to variance alone without full manifests.
- Fix: record independent seeds for template prediction, optimization, MPNN, and each evaluation model; share these across comparable arms where appropriate. Randomize/interleave execution blocks; pair trajectory IDs and repeat stochastic downstream draws if required. Store configuration/environment/checkpoint/template hashes and all attempts. Analyze per-trajectory yield and bootstrap paired trajectories; resample targets at the top level for generalization. Prespecify sweep selection and hold out targets for confirmation. Report equal trajectory budget and equal GPU-hour efficiency, since extra recycles/models change cost.

**M08 — P2, confirmed: extending A/Bs to other advertised folds silently changes the reference hotspot configuration.**

- Locations: `baseline/ab_get_best.py:42-49` versus `baseline/repro_config.tsv:3-5` and corresponding example scripts.
- Barrel A/B uses `30-44,90-104` instead of `12-15,25-31,40-43`; Ig-like `1-9,40-54` instead of `17-24,39-46`; solenoid `28-42,70-84` instead of `7-13,26-33,46-53,64-71,83-88`. The comment claims example provenance.
- CPU PDB inspection found all these positions in range (including polymer MSE: lengths 111, 77, 98). This is an objective/configuration mismatch, not an out-of-bounds crash. The published Top7 experiments use the matching Top7 configuration and are not invalidated by this particular discrepancy.
- Fix: one validated manifest source for examples, scheduler, and A/B arms; print/store expanded mapped hotspot IDs and config hashes. Do not describe a new-fold baseline as author reproduction unless it actually uses the author configuration.

### Loss design and opportunities for accuracy

**M09 — P2, confirmed mathematical behavior; benefit of changing it is unmeasured: contact-map loss is size- and mask-density-dependent and conflates observed negatives with unknown pairs.**

- Locations: `FoldCraft.py:232-240`, `cmap_utils.py:95-112`, `baseline/ab_loss_fp.py:53-65`.
- Existing loss is `sqrt(sum_ij(error_ij^2) / total_length)`, not RMS over constrained pairs. Padding with unconstrained target residues changes its scale. CPU probe with the identical two nonzero errors: length 4 gives 0.7071, length 8 gives 0.5. Changing hotspot count or binder length also changes the relative intra/inter contribution.
- `binarize_cmap` turns every positive entry into observed support. Zero may mean intentionally unconstrained, an excluded loop, or a true noncontact. Conversely, a very small *positive* target probability remains supervised and can penalize a predicted false contact. Describing all binder-map supervision as strictly recall-only is imprecise.
- FP A/B penalizes all off-mask pairs, including unknown/masked regions, and sums these with different normalization from the original loss. Its failure at weights 0.1/0.3 therefore does not refute a sparse, observation-aware negative restraint.
- Improvement: separate observation masks from target probabilities; split intra-binder, requested interface, and explicit forbidden-contact terms; normalize each by valid pair/row counts and log values/gradient norms. Check invariance to unconstrained padding and mask changes. Preserve a legacy mode for exact baseline comparisons. This is a hypothesis requiring A/Bs, not a safe silent numerical cleanup.

**M10 — P2, confirmed objective mismatch; proposed remedy is a hypothesis: every selected hotspot pair is forced toward contact within a broad default distance.**

- Locations: `cmap_utils.py:97-100`, `FoldCraft.py:232-240,315-317`.
- The interface target is the full Cartesian product of target and binder hotspot sets. This requests every binder hotspot near every target hotspot, including spatially dispersed choices. With the reviewed dependency, contact probabilities are integrated below default interface cutoff 21.6875 Å, intra-binder cutoff 14 Å (`ColabDesign af/model.py:50-51`, `af/loss.py:215-227`). No interface confidence term is weighted in production.
- A broad proximity objective is not direct evidence of a compact, specific, physically viable interface. Simply lowering the cutoff while retaining impossible all-to-all requirements could make optimization worse.
- Improvement order: first test feasible sparse/per-hotspot coverage (e.g. smooth top-k/contact-at-least-one) while retaining the old distance scale; then test physically tighter contact definitions and coarse-to-fine schedules. Change these factors separately and log actual heavy-atom contacts, requested-epitope coverage, off-epitope contacts, steric clashes, binder fold fidelity, and independent predictor agreement. Thresholds need calibration and predefined atom/distance conventions.

**M11 — P2, confirmed measurement limitation: A/B 'fold fidelity' is not actually an isolated fold-retention metric.**

- Locations: `baseline/ab_loss.py:19-22,91,110-116`, `baseline/ab_loss_fp.py:14,59,114`; `baseline/ACCURACY_RESULTS.md:10-13`.
- The cmap loss combines intra-binder and inter-chain terms. It may change because interface placement changes even if the binder fold is unchanged. Design-trajectory loss is only printed/kept in memory; the committed `cmap_loss` column records subsequent AF2 validation loss from `ab_get_best.py:124-127`. Those columns cannot directly reproduce all claimed trajectory-loss summaries.
- Fix: persist stage/checkpoint losses with names and provenance; report template RMSD with residue coverage, global fold similarity/topology, and separate intra/interface map errors. Do not use a decrease in one combined loss as proof of preserved fold.

**M12 — P2, scientific limitation: independent predictors, ipSAE, and OpenMM remain uncalibrated proxy measurements.**

- Locations: `baseline/score_esmfold.py:1-12`; `baseline/ipsae.py:8-12,39-55`; `baseline/score_openmm.py:82-116`; `baseline/consensus.py:4-9`.
- ESMFold RMSD compares its monomer to the *designed* binder chain, not to the requested template or experimentally observed structure. Agreement is useful but not proof of folding. ipSAE is derived from the same AF2 PAE and is not an independent model. The max-of-directions score is consistent with the reference definition; it is not itself a bug. Alternative directional/minimum scores should be distinctly named and calibrated.
- OpenMM calculates interaction energy of one minimized predicted geometry. It does not include unbound-state reorganization or entropy and is not affinity. AF2/Boltz disagreement cannot identify which prediction is correct without external labels. Requiring both can reduce some single-predictor failures but does not mathematically remove bias. Integration docs acknowledge this, but several script docstrings still say 'neutral oracle', 'validated loss', or consensus 'removes bias'.
- Fix/improvement: retain raw per-predictor outputs; calibrate a selector on independently labeled positives/negatives with family/target-disjoint validation. Until labels exist, report yield under specified computational criteria, not binding accuracy. Evaluate precision-at-fixed-selection-budget, calibration, and target-level hit rate once labels are available. Include fold-conditioned and unconstrained tasks as different questions rather than pooling them into a purportedly fair single benchmark.
- Primary reference checked: [DunbrackLab ipSAE definition](https://github.com/DunbrackLab/IPSAE).

**M13 — P3, CPU-confirmed input-validation omission: ipSAE accepts malformed nonsquare matrices.**

- Location: `baseline/ipsae.py:45-51`. It checks only the first dimension against chain lengths. A 4×1 all-ones array with lengths 2/2 broadcasts into calculations and returns 0.5. Validate square rank-2 shape, finite nonnegative PAE, positive chain lengths, and valid cutoff. This should be included with measurement-contract fixes; it is not evidence the committed valid-shaped tables were wrong.

## What the existing A/B results change

The exploratory trajectory bootstrap from the root reviewer uses 100,000 percentile resamples, seed 20260926, and 15 trajectories per arm. These intervals are conditional on one fold/target and are not adjusted for selecting among weights; they are a diagnostic, not a confirmatory trial or a formal power analysis.

| Change | Observed gate-rate difference | Exploratory trajectory-bootstrap 95% interval | Decision |
|---|---:|---:|---|
| Best versus last checkpoint | −4.0 percentage points | −14.67 to +6.67 | Deprioritize a default switch; no demonstrated benefit. Keep checkpoint capture for diagnosis. |
| iPAE weight 0.05 | +12.0 pp | −14.67 to +40.0 | Inconclusive; retest only after experimental controls and proper fold metrics. |
| iPAE weight 0.1 | +2.67 pp | −18.67 to +24.0 | Inconclusive. |
| iPAE weight 0.2 | +17.33 pp | −6.67 to +40.0 | Most suggestive pilot arm; still not a default recommendation. |
| Existing FP weight 0.1 or 0.3 | −48.0 pp | −66.67 to −29.33 | Retire these settings; any new negative-contact experiment must use a new formulation/scale and explicit observation masks. |

The ~24% reported performance gain predates this branch and is not evidence of new prediction accuracy. The 187/200 corrected Ig-like RMSD result supersedes the earlier incorrect 9/200 claim.

## Logical implementation and experimental order

1. **Correct input/residue mapping and deterministic pipeline behavior first.** Root/other-agent findings cover these. Include MPNN terminal/empty selections, target/binder missing residues, hotspot validation, and example-map consistency. Unit fixtures must cover native numbering, insertion codes, MSE, chain identities, masked regions, and mixed input lengths. Preserve legacy reference artifacts; do not compare objective changes against silently changing input semantics.
2. **Repair measurement contracts.** Implement M01/M03/M04/M05/M06/M08/M13 together with immutable scorer provenance and failure states. Define per-model outputs and aggregation before enabling two models. Add targeted CPU fixtures and a small GPU integration smoke test only when compute is available. Historical data remain historical; no backfilled claim of models/protocols that did not run.
3. **Create a reproducible corrected baseline.** Pin dependency/model/config/template versions; hierarchical seeds; stage-by-stage metrics; all attempts/failures; paired IDs; atomic immutable artifacts. Re-run the unmodified scientific objective through the corrected pipeline. Report old-versus-corrected output changes separately from accuracy experiments. Restore raw historical structures if available, otherwise start a new reproducible cohort.
4. **Calibrate selection and establish a target/fold benchmark.** Re-score a fixed complete candidate set; compare historical confidence-only selection with explicit fold/coverage/geometry and independently calibrated confidence ranking. Use a small calibration set of experimental positive/negative examples where available and clearly marked synthetic decoys as diagnostic controls. Reserve targets/families and scaffold families for confirmation. Do not pick thresholds on the final test set.
5. **Test model changes individually in the experiment sequence below.** Only combine independently promising changes after factorial/ablation tests establish their interaction. Keep performance changes separate so quality effects and GPU savings can be attributed.

## Accuracy experiment sequence and promotion criteria

Common design: paired target/fold/initialization blocks, independently recorded stage seeds, randomized execution order, fixed MPNN count and validation budget; all candidate/failure records retained. Discovery can start with 3 varied public target/epitope systems × 4 folds × roughly 12–20 paired trajectories/arm to estimate feasibility and variance; these are planning numbers, not a claim of adequate power. Choose confirmatory sample size from a prespecified meaningful effect and trajectory/target-level pilot variance, expanding to held-out targets and folds. Use paired trajectory bootstrap within target and target/family resampling for cross-target claims. Fix the primary metric and correction/selection procedure before sweeping weights.

Common endpoints: (a) unique candidates satisfying explicitly defined fold/coverage/interface/integrity criteria per trajectory, (b) that yield per GPU-hour and per fixed number synthesized, (c) each gate's retention/rejection rate and scorer completeness, (d) continuous structural/confidence metrics, diversity, and failure rates. With experimental labels, make precision-at-fixed-budget and target-level experimental hit rate primary; without them, call the outcome computational yield. Normalization choices must not let a changed optimization loss define its own victory.

| Order / experiment | Arms and controls | Specific metrics and promotion rule |
|---|---|---|
| E0: Corrected baseline and repeatability | Frozen integrated baseline versus correctness-fixed pipeline; repeated identical seeds plus new-seed repeats. No objective change. | Exact CPU invariants; saved mapped inputs identical where valid; GPU reproducibility tolerance established; all result rows linked to artifacts. Explain every changed candidate/metric before running model A/Bs. |
| E1: Validation and selection | Same candidate panel, explicit AF2 model1 versus models1+2, mean versus conservative/disagreement-aware selection; compare confidence-only with fold/geometry gate. | Per-model disagreement; fold/epitope violation rate among selected candidates; calibrated precision/recall and selected-candidate diversity; cost. Promote a stricter gate for its declared requirement, but do not claim accuracy gain without independent labels. |
| E2: Contact observation masks and normalized objectives | Legacy objective; split/normalized intra and interface loss; then explicit known-negative term as a separate ablation. Keep geometry/sequence schedules fixed. | Padding invariance and gradient scaling on CPU; true fold metrics, epitope contacts, clashes, unique passing yield. Promote only if held-out yield improves and fold fidelity/integrity meet prespecified noninferiority margins; do not reuse failed 0.1/0.3 FP weights. |
| E3: Feasible interface geometry | Cartesian restraints versus sparse/per-hotspot coverage at same broad distance; then tighter-distance or coarse-to-fine arms using the winning coverage formulation. | Contact precision/coverage on requested epitope; spatial feasibility, clashes, fold retention, independent predictor agreement. Promote if gains persist on held-out non-PD-L1 targets at matched budget. |
| E4: Confidence regularization | Weight 0 versus a small prespecified iPAE sweep, with pLDDT or other losses separately rather than bundled. Use normalized loss components from E2 or hold legacy scale constant. | Real fold and geometry metrics; independent validation/experimental precision in addition to self-confidence. Promote only after a new-seed held-out confirmation excludes a practically important loss in fold retention and improves the primary endpoint. |
| E5: Template conditioning | Predicted native-monomer map versus template-coordinate distance/contact representation and a confidence-weighted prediction map. Retain identical residue maps/observation masks. | Fidelity to intended topology and mask/core regions; generalization across folds; sequence diversity and target binding criteria. No assumption that the direct or predicted template is inherently better. |
| E6: Sampling and predictor compute allocation | MPNN non-interface versus full or controlled interface resampling, limited temperature sweep; separately design recycles0/1 and validation3/6; AF2-multimer as a separate protocol arm. | Unique qualifying candidates per GPU-hour, per trajectory, and fixed selection budget; fold/interface quality and redundancy. Compare equal compute as well as equal trajectory count. Pin protocol and calibrate score thresholds rather than reusing AF2-ptm cutoffs blindly. |
| E7: Independent ranking and external confirmation | Rank the same frozen panel with AF2-only, directional/max ipSAE variants, AF2+Boltz, ESM/template consistency, and QC-passing molecular-mechanics descriptors. Calibrate on training targets only. | Precision-at-k, recall, calibration, target-level holdout hit rate, robustness to sequence length/composition, missing-scorer handling. Use matched synthesis/assay budget for a final blinded experimental panel. Promote only after superiority/noninferiority criteria chosen in advance are met. |

For a concrete default decision rule, preregister a meaningful gain such as at least five percentage points of computational yield (or a prespecified relative gain when the base rate is very low), require a confirmatory trajectory/target-aware confidence interval excluding zero for that primary endpoint, and set separate fold/integrity noninferiority margins and an acceptable GPU-cost increase. These are choices to agree before running, not thresholds justified by the current pilot. Experimental binding-accuracy claims need external binding measurements; improvement in a model's own confidence is insufficient.

Best-checkpoint selection is a low-priority optional ablation after these steps. Large cross-generator leaderboard claims should wait for equal generation/selection budgets, explicit task matching, complete rescoring, and independent labels.
