# FoldCraft: deep review of the integrated working branch

Reviewed 26 September 2026. Branch: `codex/correctness-and-validation`.
Commit: `e485f3244e47464ae6083c193f234e91c099e0c5`, also tagged
`integration-baseline-2026-09-26`. No production implementation was changed for
this review. The next actions are ordered in [CHANGE_PLAN.md](CHANGE_PLAN.md).

**Fix conditioning, run state and evaluation integrity before optimizing model
scores.** The branch has useful experimental tooling and 113 passing CPU tests,
but important contracts remain untested. New probes reproduce silent reuse of
incompatible/incomplete outputs, missing scores counted as failures, and stale
Boltz predictions selected as current results. The previously identified residue
mapping and model-count problems also remain.

No P0/critical issue is established. P1/high means a realistic failure can
invalidate conditioning/evaluation, lose campaign results, or defeat a compute
budget. P2/medium means a localized defect or important reliability limitation.
An opportunity is a hypothesis to benchmark, not a demonstrated accuracy gain.

## Scope and evidence

Reviewed the main CLI, map/structure helpers, both notebooks, experimental driver,
installer/entry points, scheduler/runners, added scorers, A/B harnesses, tests and
committed results. Dependency-dependent findings refer to inspected ColabDesign
`e31a56fe1d9b4de25c8697f3a28b75892941cc72`; the current installer follows a moving
source, so pinning the executed version is part of the plan.

| Check | Result and limit |
|---|---|
| Existing CPU suite | **113 passed** on this branch, Python 3.12.7; tests cover helpers, scoring and planning, not full GPU execution. |
| Deterministic correctness probes | Reproduced residue/shape errors, wrong model dispatch, invalid-input acceptance, MPNN/notebook issues, and unsafe checkpoint assumptions. Dependency parsing/dispatch was exercised without neural inference. |
| Eight operational probes | Reproduced false campaign completion, incompatible resume, partial merge completion, retry overlap, scorer lost updates, ignored score configuration, stale Boltz selection and a successful shell exit after all jobs failed. |
| Committed-data audit | Recounted metrics, missing evaluations and A/B outcomes; performed exploratory trajectory bootstrap. No duplicate names or exact sequences within the audited result tables. |
| GPU and structural re-evaluation | **Not performed.** No prediction accuracy/speed measurements, weights installation, raw-structure rescoring or binding assays. Most original structures are not in Git. |

Reproduction scripts and captured outputs are in [evidence/](evidence/). They
assert/reveal current failures; they are review probes, not a replacement for
regression tests that assert corrected behavior.

## 1. Input, conditioning and sequence contracts

| ID | Severity/evidence | Finding and consequence | Change |
|---|---|---|---|
| C1 | **P1; CPU reproduced** | Native PDB residue IDs are interpreted as array positions in the contact map but mapped as residue IDs by the predictor. For a target A10–A29, hotspot A15 should map to index 5; the loss conditions index 14, corresponding to A24. | One authoritative residue-to-model mapping; F1. |
| C2 | **P1; CPU reproduced** | Map preparation and binder-mode preparation disagree on missing target residues. Residues 1,2,4,5 produce lengths 5 versus 4. Even contiguous numbering can fail: a missing backbone N at residue 2 produced lengths 3 versus 2 in the inspected dependency. | Shared target preparation/atom-coverage policy and shape validation; F1. |
| C3 | **P2; CPU reproduced** | Invalid hotspots, malformed maps and invalid budgets are accepted. Binder position 0 can touch the target block; reversed ranges remove restraints; a 1×1 map broadcasts to 3×3 and NaNs survive. CLI accepts zero MPNN batch size and malformed stage counts. | Fail before loading weights; validate all ranges, shapes, values and budgets; F1. |
| C4 | **P2; array comparison** | Three VHH example maps differ from current runtime conditioning: PD-L1 180 cells, PD-1 170, IFNAR 120; EGFR agrees. These represent different scientific inputs, not numerical noise. | Version the historical/current conventions and verify golden fixtures; F1. |
| C5 | **P2; source/CPU contract** | `range(1, binder_len)` excludes the last residue from MPNN redesign. An empty non-interface mutable selection becomes `fix_pos=''`, which fails in the inspected dependency. | Exact mutable-set construction, target immutability and explicit zero-redesign behavior; F2. |
| C6 | **P1; dependency dispatch reproduced** | Validation lists two AF2 models without `num_models=2`; the reviewed defaults execute only the first. This occurs in the main driver and new experiment/scoring paths. | Request/log actual model execution, coupled to per-model artifacts; F2/F5. |

Sources: [map construction](../../cmap_utils.py#L80),
[target preparation](../../FoldCraft.py#L117),
[binder-mode preparation](../../FoldCraft.py#L308),
[MPNN positions](../../FoldCraft.py#L326),
[validation call](../../FoldCraft.py#L363),
[new AF2 scorer](../../baseline/score_af2.py#L35).

**The two-model fix must be atomic with an output-contract fix.** The inspected
ColabDesign code averages logged metrics across models and can save multi-model
PDBs. Current ipSAE/RMSD readers select model zero. Merely changing one argument
would compare ensemble-average confidence with a single model's structure and
PAE. Persist model identities and align every score with its own artifact before
choosing an ensemble acceptance policy. See [PAE selection](../../baseline/add_ipsae.py#L69)
and [RMSD model selection](../../baseline/add_rmsd.py#L44).

The main acceptance gate uses confidence alone. It does not require the intended
fold, requested epitope contacts or absence of clashes. This is a high-priority
**evaluation limitation**, not evidence that every accepted design is defective.
Adding calibrated physical/fold acceptance belongs after the underlying scoring
bugs are repaired (I2). [Acceptance](../../FoldCraft.py#L454)

## 2. Run integrity and orchestration

| ID | Severity/evidence | Finding and consequence | Change |
|---|---|---|---|
| R1 | **P1; CPU reproduced on this checkout** | The scheduler treats committed `baseline/repro` summary CSVs as completed campaigns. Its dry-run marks **20/24 chunks done**, although all five corresponding design directories are absent; copying the template is enough to satisfy completion. | Separate reference data from live output; validate manifests and required artifacts; F3/F4. |
| R2 | **P1; CPU reproduced** | Resume checks ignore configuration and artifact integrity. Changing a chunk from one to ten trajectories and changing hotspots still schedules no work. A deliberately interrupted merge with a header-only CSV is considered finished. | Configuration-aware completion and atomic merged publication; F3/F4. |
| R3 | **P1; CPU reproduced/source** | Checkpoint progress is rewritten in place, so interruption can destroy the previous partial checkpoint. An old final CSV survives a new incomplete run. Success sampling has no maximum attempts/time. | Transactional checkpoints, safe output reuse and bounded sampling; F3. |
| R4 | **P2; CPU reproduced** | A retry labeled solo can start alone, then admit another job on the same GPU. This defeats the intended OOM-recovery isolation. | Reserve the GPU for the retry's full lifetime; F4. |
| R5 | **P2; CPU reproduced** | `run_campaign.sh` returns success and prints `CAMPAIGN_DONE` even when all six design commands fail. Supervising automation cannot rely on its exit status. | Aggregate failures and return nonzero; F4. |
| R6 | **P1 for reproducible evaluation; source** | Runs lack complete versioned manifests, effective configurations and replayable stage seeds. Scorer completion is inferred from populated columns rather than settings/input identity. | Immutable provenance and distinct generated/evaluated/failed/unknown states; F3/F5/F6. |

Sources: [scheduler completion](../../baseline/scheduler.py#L95),
[template repair](../../baseline/scheduler.py#L118),
[merge publication](../../baseline/scheduler.py#L181),
[scheduler launching](../../baseline/scheduler.py#L334),
[campaign exit handling](../../baseline/run_campaign.sh#L14),
[checkpoint writer](../../biopython_utils.py#L50),
[sampling loop](../../FoldCraft.py#L375).

A fresh checkout with a new empty output root plans all 24 chunks; the problematic
behavior is the default historical-output root. This distinction explains why a
basic empty-directory dry-run passed during integration while default resume is
still wrong. New large campaigns should wait for F3–F6.

## 3. Scoring and scientific interpretation

| ID | Severity/evidence | Finding and consequence | Change |
|---|---|---|---|
| E1 | **P1; committed-data reproduction** | Consensus treats missing Boltz scores as failures and claims exact coverage. **27 of 44 AF2-passing Top7 rows lack Boltz scores.** Current pooled 155/1,000 is a known-pass lower bound, not a fully observed success rate. | Fail on missing required scores or report known/unknown counts and bounds; F6. |
| E2 | **P1; CPU reproduced** | Boltz scoring recursively collects confidence files from a reused workdir and chooses the maximum. A stale ipTM 0.99 wins over a fresh 0.10–0.20 result. Existing score columns also bypass changed target/hotspot/scorer settings. | Isolated, provenance-keyed outputs and exact candidate/model selection; F5. |
| E3 | **P1 when scorers overlap; CPU reproduced** | Independent scorers read and rewrite the entire shared CSV. Interleaved successful AF2 and ESM writes delete the ESM columns. Direct rewrites also risk partial files on interruption. | Per-scorer atomic artifacts plus a validated join, or robust locking/transactions; F5. |
| E4 | **P2; CPU reproduced** | RMSD helpers truncate to the shorter atom list and pair prefixes rather than validating residue correspondence. A 5-residue versus 3-residue example returns 0 Å despite incomplete coverage. | Sequence/residue-aware alignment, model identity and coverage thresholds; F5. |
| E5 | **P2; API contract/source** | OpenMM help advertises `--min-iters 0` as raw scoring, but the minimizer receives 0, which means no iteration limit. It changes both the measurement and runtime. | Bypass minimization for raw scoring; separate bounded convergence controls; F5. |
| E6 | **P2; data/source/CPU** | Energy and ipSAE inputs lack adequate validity/status checks. One committed Top7 energy is ~1.33×10¹⁹ kcal/mol; a malformed rectangular 4×1 PAE array is accepted and scores 0.5. | Record physical/numerical QC and failure reasons; validate square PAE, finite values and chain splits; F5. |
| E7 | **P2; config comparison** | Non-Top7 A/B fold hotspots disagree with author/example configurations. They are in bounds but define different objectives. The existing Top7 experiments are unaffected. | One shared versioned configuration source before multi-fold experiments; F6. |
| E8 | **P1 for accuracy claims; source/data** | Sibling MPNN sequences are counted as independent trials. Loss experiments are unpaired; best/last shares a trajectory but not all downstream randomness. The combined intra/interface cmap score is labeled fold fidelity; design-stage loss summaries are not fully persisted. | Clustered/paired analysis, all-stage seeds and separate fold/interface measurements; F6. |

Sources: [missing-score comparison](../../baseline/consensus.py#L54),
[Boltz file selection](../../baseline/score_boltz2.py#L85),
[Boltz CSV writes](../../baseline/score_boltz2.py#L151),
[AF2 cached columns](../../baseline/score_af2.py#L60),
[ESM RMSD](../../baseline/score_esmfold.py#L76),
[OpenMM minimizer](../../baseline/score_openmm.py#L107),
[OpenMM CLI](../../baseline/score_openmm.py#L131),
[official minimizer API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html),
[ipSAE input](../../baseline/ipsae.py#L39),
[A/B configs](../../baseline/ab_get_best.py#L45),
[A/B arm loop](../../baseline/ab_loss.py#L82),
[sequence-level intervals](../../baseline/score.py#L31).

The extreme energy should be treated as a QC failure needing structural
inspection, not proof of a repulsive biological interface. The review cannot
identify its cause without the original structure and minimization record.

### What the committed data actually covers

| FoldCraft reproduction fold | AF2 passes | Missing Boltz among AF2 passes | Known consensus passes | RMSD<3.5 Å |
|---|---:|---:|---:|---:|
| Ankyrin | 2/200 | 0 | 2 | 22/200 |
| Barrel | 56/200 | 0 | 55 | 187/200 |
| Ig-like | 66/200 | 0 | 66 | 187/200 |
| Solenoid | 20/200 | 0 | 18 | 41/200 |
| Top7 | 44/200 | **27** | 14 | 166/200 |
| Total | 188/1,000 | **27** | **155** | 603/1,000 |

There are 155 known consensus passes and 27 unresolved candidates: completion
could yield **155–182 passes**, without changing any observed result. This is a
missing-data bound, not a confidence interval. Only 225/1,000 FoldCraft rows have
Boltz scores and 188 have ESM scores. The BoltzProt table has those scores for
100/200 designs. Selected subsets must not be described as full-population
validation or compared without explicit denominators/sampling rules.

The integrated report correctly replaced the historical Ig-like 9/200 claim
with **187/200** passing the template RMSD threshold; 66/200 pass all confidence
and RMSD criteria. Therefore, prioritizing Ig-like solely because of the old
9/200 report is unjustified. Ankyrin/solenoid have much weaker recorded fold
retention in this configuration, but raw structures still need auditing.

### Updated interpretation of the A/B experiments

Exploratory bootstrap below uses 100,000 resamples, seed 20260926, and trajectories
rather than individual sequences. Best/last uses matched trajectory differences;
loss arms are unpaired because the original runs do not share controlled seeds.
These are single-target, single-fold, 15-trajectory pilots; intervals are neither
multiplicity-adjusted nor a guarantee about new targets.

| Variant | Confidence-gate difference | Exploratory 95% trajectory-bootstrap interval | Decision |
|---|---:|---:|---|
| Best versus last checkpoint | −4.0 percentage points | −14.7 to +6.7 | No demonstrated benefit; lower priority. |
| iPAE weight 0.05 | +12.0 pp | −14.7 to +40.0 | Uncertain; do not adopt as default. |
| iPAE weight 0.10 | +2.7 pp | −18.7 to +24.0 | Uncertain. |
| iPAE weight 0.20 | +17.3 pp | −6.7 to +40.0 | Worth a controlled follow-up, not proof of improvement. |
| Off-mask penalty 0.10 or 0.30 | −48.0 pp | −66.7 to −29.3 | Reject these tested formulations/settings. |

The two nominally cmap-only baselines achieved 18.7% and 48.0%, confirming poor
repeatability but not proving that all of the difference is random variation.
Without complete manifests, unrecorded differences cannot be ruled out. The
failed penalty treats unobserved/off-mask regions as negatives; it does not refute
an explicitly masked, normalized negative-restraint objective.

## 4. Performance and software structure

Already implemented: invariant map caching, hoisted MPNN construction with key
reset, and validation-model reuse within each trajectory. Preserve these. The
historical ~24% speedup is not an incremental gain from this review.

Remaining opportunities, requiring measurement:

- Full auxiliary pickle trees include internal recycling tensors. Compact
  artifacts can retain sequence/structure/PAE/confidence/provenance without every
  intermediate representation. Audit readers before pruning fields.
- Per-trajectory AF model construction and large unused histories may repeat
  compilation and retain host memory. Profile cold/warm behavior by complex size;
  lifecycle changes must preserve seeds, clean state and outputs.
- Cross-trajectory reuse and persistent JAX compilation caching could help, but
  `clear_mem()` deletes device buffers. Reuse must not retain invalid RNG/state.
- Exact-sequence deduplication can save validation work in future larger searches.
  No within-table exact duplicates were found in the committed data, so no
  existing savings are demonstrated here.

The current test suite has important blind spots: driver orchestration is not
importable without GPU dependencies, scheduler subprocess/GPU orchestration is
explicitly untested, and a differential-map test passes hotspot 20 into a target
of length 10. It verifies agreement with another implementation, not validity.
Add pure contracts and adversarial integration tests with the fixes; avoid a
large refactor solely for test coverage.

Sources: [test scope](../../tests/test_scheduler.py#L1),
[invalid map fixture](../../tests/test_cmap_utils.py#L67),
[model construction](../../FoldCraft.py#L302),
[output serialization](../../FoldCraft.py#L365).

## 5. Notebooks, experimental code and packaging

These remain distinct supported-surface decisions rather than prerequisites for
a new scientific method:

- **P1 for notebook users:** the general notebook uses inconsistent artifact paths and an
  incorrect string-versus-integer interface membership test. The reproduced
  non-interface selector includes residues 2 and 3 despite their being protected.
  Both notebooks retain exclusive range parsing; the VHH notebook also diverges
  from the CLI's indexing/input behavior. Repair them or explicitly narrow the
  supported entry points; F7.
- **P2:** the experimental driver sets import paths after importing a root helper,
  its default mode writes an empty result without designing, its random length
  upper bound can be exceeded, and its selected PDB can represent a best snapshot
  while stored metrics/pickle represent the last snapshot. Paths are used in
  unquoted shell commands. Repair defaults and artifact consistency before use.
- **P2:** structure helpers can mix atoms across PDB models when calculating
  clashes. A synthetic two-model structure showed two cross-model clashes even
  though neither model had one. Use per-model geometry, particularly before
  enabling multiple prediction models.
- Explicit weight paths, pinned dependency versions/checkpoints and an installer
  platform check are required for reproducible GPU execution. macOS CPU helper
  tests passing does not validate the Linux/CUDA installation workflow.

See the notebook cell/line references and exact reproducible entry-point cases
in [the correctness appendix](evidence/current_correctness.md).

## Conclusion and next action

The integrated branch is a useful starting point, not yet a reliable accuracy
benchmark. Its strongest contribution is evaluation infrastructure; the strongest
current evidence is for **fixing contracts**, not for changing the optimization
objective immediately.

Start with F1 input mapping/preflight, then F2's coupled MPNN/model-artifact
corrections. Finish run/scorer integrity before spending on new campaigns.
After the engineering gate, prioritize inexpensive resource reductions and
trustworthy acceptance/ranking, followed by normalized/sparse conditioning and
controlled MPNN search. Model-family/recycle sweeps and iterative redesign follow
only if their extra cost is justified. The full sequence, tests, experiment
budgets and promotion criteria are in [CHANGE_PLAN.md](CHANGE_PLAN.md).

Detailed appendices: [correctness](evidence/current_correctness.md), [operations](evidence/current_operations.md), [model/evaluation](evidence/current_model_eval.md). The consolidated IDs and ordering in this review and plan take precedence over appendix-local IDs.
