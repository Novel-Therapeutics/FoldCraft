# Current-branch correctness review

Reviewed September 26, 2026. Repository: `/Users/andreivolgin/PyCharmProjects/FoldCraft-Novel-Therapeutics`; branch `codex/correctness-and-validation`; HEAD `e485f3244e47464ae6083c193f234e91c099e0c5`. Dependency contract checked against the source-only ColabDesign checkout `/private/tmp/foldcraft-review-colabdesign`, commit `e31a56fe1d9b4de25c8697f3a28b75892941cc72`. This does not establish which dependency revision generated historical GPU results: the installer still installs an unpinned branch.

No implementation changes were made; the review deliverables were added afterward. CPU reproductions are in `review_evidence/current_correctness_probes.py`; they execute the current repository's pure helpers, extracted unchanged upstream parsing/dispatch methods, notebook AST snippets, and experimental-driver no-inference dispatch. They do not simulate successful neural predictions. Run with:

```
PYTHONDONTWRITEBYTECODE=1 /private/tmp/foldcraft-integration-venv/bin/python /Users/andreivolgin/Documents/ChatGPT/FoldCraft/review_evidence/current_correctness_probes.py
```

All assertions completed successfully. They assert the observed defective behavior so failures can be converted into regression tests during implementation. No GPU inference was attempted.

Severity: P1 invalidates scientific conditioning/validation or a documented workflow; P2 material localized correctness/data-integrity issue; P3 helper issue outside current production acceptance. No P0 found.

## Main CLI and scientific input contract

### C01 — P1, reproduced: native target residue identifiers select the wrong epitope

Locations: `cmap_utils.py:80,97–100`; `FoldCraft.py:184–186,308–312`.

`assemble_fold_conditioned_cmap` treats numeric hotspot R as array position R−1. The same string is passed to ColabDesign, whose `prep_pos` searches the actual PDB residue identifiers. For target A10–A29, requested A15 resolves to index 5 in upstream preparation; FoldCraft writes index 14, corresponding to A24. Both fit, so there is no exception and the loss optimizes a different target site. The binder-count guard does not catch offset-only numbering.

Fix: one authoritative original `(chain, resSeq, insertion code)` → processed model index map. Use it for target/binder selections, crops, conditioning, MPNN mutable sets, and output attribution. Define whether CLI numeric binder selections are template identifiers or generated ordinals; do not conflate them. Renumbering may be a preprocessing implementation, provided the mapping survives and requested selections are translated. Until supported, reject ambiguous or unsupported numbering before model/weight loading.

Acceptance: fixture A10–A29 with A15 must constrain index5; identity fixture1..N remains bit-identical; all valid generated hotspots lie exclusively in the cross-chain blocks; insertion codes/multiple chains either translate explicitly or produce a helpful early error; exported manifest records original and processed IDs.

### C02 — P1, reproduced: target preparation uses incompatible missing-residue policies

Locations: `FoldCraft.py:116–125,174–185` versus `FoldCraft.py:308–312,356–360`; upstream `colabdesign/af/prep.py:211–214,429–449`.

The setup `fixbb` length includes missing positions (`ignore_missing=False`); binder preparation hardcodes `ignore_missing=True` for target chains. Target residues1,2,4,5 produce setup length5 versus binder-model target length4, leading to incompatible conditioned loss tensors. A newly checked case proves contiguous numbering alone is insufficient: a three-residue target with N missing at residue2 but CA present produces lengths3 versus2. In the inspected upstream code the filtering mask is atom index0 (N), despite its docstring describing missing CA.

Fix: derive target sequence, lengths, atom masks, conditioning and downstream inputs from the same processed target artifact and explicit atom-coverage policy. Validate aligned dimensions before inference. Passing `ignore_missing=False` to the binder constructor is insufficient in this dependency because the target policy is hardcoded.

Acceptance: numbering gaps and separately missing N/CA/backbone atoms; no silent removal without mapping; all prepared target/cmap dimensions match, or fail before expensive model work; preserve target hotspot identity after compression.

### C03 — P2, reproduced: selection and contact-map validation is incomplete

Locations: `biopython_utils.py:12–24`; `cmap_utils.py:78–100,105–113`; `tests/test_cmap_utils.py:67`.

Binder hotspot0 writes a target-target entry; target hotspot0 wraps into binder positions; reversed ranges return an empty selection and silently remove interface conditioning. The mapper accepts a `(1,1)` binder cmap for a length3 binder, broadcasting its single value into a `(3,3)` block. Nonfinite values propagate into the objective. The current differential tests duplicate assumptions rather than validating chain boundaries: one VHH test even passes target hotspot20 with target_len10.

Fix: validate grammar, endpoints, emptiness, existence in the selected chain, duplicates policy, exact square shape, finite numeric values and probability range. Validate the mapped indices, not just raw numeric bounds, once C01 is implemented. Treat shape/mask semantics as an explicit data contract.

Acceptance: reject0, negative, reversed, missing/out-of-chain and malformed selections; reject `(1,1)` broadcasting and NaN/Inf/non-square maps; verify contact-block invariants; retain valid explicit and all-binder-default parity; all errors precede GPU work.

### C04 — P1, dependency-dispatch reproduced: nominal two-model validation executes one model

Locations: `FoldCraft.py:363,452`; upstream `colabdesign/af/model.py:47–48`, `colabdesign/af/design.py:62–77,276–302`.

Fresh model opt.num_models is1. Supplying two names without `num_models=2`, with predict's sample_models=False, selects `[0]`. Extracting the real upstream dispatcher yields `[0]` for current call and `[0,1]` with explicit2. Other agents cover the new benchmark scripts that reuse this pattern.

Fix: explicit model selection/count plus logged actual model identities and per-model scores. Define ensemble aggregation and which structure is selected. Simply adding2 produces multi-model PDB files; downstream scoring/relaxation must be made model-aware, so include artifact handling in the same change.

Acceptance: CPU API-contract tests assert exact requested models; GPU smoke emits two outputs, two identities, correct aggregation and deterministic seed manifest. Preserve and label the one-model historical baseline; do not rewrite its labels as an ensemble.

### C05 — P2, reproduced: MPNN mutable set excludes the terminal binder residue and can be empty

Locations: `FoldCraft.py:326–337,416–427`; upstream `colabdesign/mpnn/model.py:80–85`, `colabdesign/shared/prep.py:19`.

`range(1,binder_len)` leaves B_N fixed even for `--redesign_method full`. Length3 currently freezes A1,A2,B3. A fully interfacial binder yields `fix_pos=''`, and upstream `prep_pos` raises IndexError; fixing the endpoint alone will not solve the empty-set case. One new benchmark helper already uses `binder_len+1`, but the production paths do not.

Fix: construct integer mutable positions once using the mapped structure. Use the entire inclusive binder set for full redesign. If empty, explicitly skip MPNN and retain the designed sequence or record a structured rejection. Do not solve this by passing `fix_pos=None`: that would make the target mutable.

Acceptance: length1, full redesign, only terminal residue mutable, all residues interfacial, none interfacial; exact immutable target/interface identities; both finite and sampling modes share one helper. End-to-end CPU driver stub verifies no empty string reaches upstream.

### C06 — P1 acceptance gap, source-confirmed: success means confidence alone

Location: `FoldCraft.py:414,454`.

Production `--sample` requires only pLDDT, iPAE and iPTM; it neither verifies intended fold retention after MPNN nor requested epitope contact nor physical clashes. Offline `baseline/score.py` adds RMSD, so the production success count and benchmark acceptance contract differ. This proves an unchecked invariant, not that a particular accepted candidate fails it. The previously quoted Ig-like9/200 fold-retention result was wrong and has been corrected during integration; do not use it as evidence here.

Fix: first record structurally mapped fold/core retention, epitope contact, per-model uncertainty and physical checks for every candidate; then introduce calibrated acceptance/ranking in an explicit versioned policy. Check per model, not pooled coordinates. Numeric thresholds need controlled benchmarking and should not be presented as a purely mechanical bug fix.

Acceptance: deliberate high-confidence wrong-fold/wrong-epitope/clashing synthetic records cannot pass strict policy; missing/NaN scores never count as success; masked VHH loops do not invalidate legitimate core matching; historical confidence-only reporting remains recoverable.

## Execution and output integrity

### C07 — P2, reproduced: progress checkpoints are not atomic, and reruns retain stale completion

Locations: `biopython_utils.py:57–79`; `FoldCraft.py:102,244–268,290–298,467–469`.

Final promotion is atomic, but each progress update overwrites/truncates `results.csv.partial` before serialization completes. A simulated exception changes the previously recoverable checkpoint into `TRUNCATED`. A rerun in a completed folder resets trajectory names to1, overwrites artifacts, and leaves the old `results.csv` visible until the new run completes. The reproduction confirms old final plus new partial coexist, invalidating the scheduler's existence-as-completion assumption after a rerun/crash.

Fix: unique temporary file → successful write/flush → atomic replace of progress checkpoint. Give runs immutable identity/manifests, with explicit compatible resume semantics and refusal of accidental reuse. A completion record must identify the run/config and certify all expected artifacts; a filename alone is insufficient. Retrying a partial run must not accidentally combine old and new structures.

Acceptance: inject exceptions at serialization, flush and rename boundaries; last good checkpoint survives; old completed directory is refused by default; changed input/config cannot resume; same-config retry preserves consistent row/PDB identifiers. Include scheduler integration tests without launching models.

### C08 — P2, reproduced/static: invalid CLI controls fail late or silently perform no work

Locations: `FoldCraft.py:39–60,95,290,319,339,386`; experimental dispatch separately below.

The parser accepts `mpnn_samples=0`, negative num_designs/target_success, negative temperature/backbone noise, and `design_stages='1,2'`. Two stages fail by IndexError only after preparation; negative finite quotas run no trajectory and still finalize empty results. In sampling mode there is no maximum attempts/time/no-progress budget, so valid but unproductive jobs can consume unlimited compute. Zero batches make progress impossible (or fail inside the sampler, depending on dependency behavior).

Fix: one validated configuration object before loading weights: exact three-stage policy with supported zero-stage semantics, nonnegative/positive constraints, valid paths/chains/selections, finite temperature/noise. Add trajectory/time budgets and structured complete/exhausted/failed status; persist rejected candidates or at least counts/reasons. Preserve the already-correct exact accepted-quota cap.

Acceptance: malformed controls fail immediately and leave no fake completion marker; fake always-rejecting and empty-sampler runs terminate at budget; exact target count retained for final passing batch; successful and exhausted outcomes distinguished.

## Notebook workflows

### C09 — P1, source-confirmed: general notebook reads different files from the ones it writes

Locations: `FoldCraft.ipynb:203–207,224–225,326,336` (JSON file line numbers).

Map preparation writes `folder_name/fold_cond_cmap*.npy`; design loads CWD `fold_cond_cmap*.npy`. Trajectory output goes under folder_name but interface detection reads CWD `<name>.pdb`. Clean execution fails; stale files in CWD can silently supply another target's map or trajectory.

Fix: notebooks become thin callers of the shared run/artifact API; until then use one output-path object throughout.

Acceptance: execute preparation/design handoff with model stubs in a clean temporary directory and with contradictory stale root files. Only the intended current-run artifacts are read.

### C10 — P1, reproduced: notebook non-interface mode redesigns interface residues

Locations: `FoldCraft.ipynb:333–337`.

Mutable positions are strings while hotspot keys are integers. For binder_len5 and interface residues[2,3], executing the actual assignment AST yields `B1,B2,B3,B4`: both protected interface positions remain mutable. It also omits B5.

Fix/acceptance: reuse C05's integer/mapped mutable-set helper; assert full/non-interface parity between CLI and notebook, including endpoint and empty-set cases.

### C11 — P1 default failure/P2 parity, reproduced/static: notebook coordinate conventions and inputs drift from CLI

Locations: `FoldCraft_VHH.ipynb:109,126,130–138,160–163`; `FoldCraft.ipynb:142–150,163,179–190`; both prediction cells.

VHH defaults crop target A19–132 but request11–17. Subtracting19 yields negative indices that wrap into the binder block; the later upstream hotspot parser rejects the absent native residues. Both notebook `set_range('2-3')` implementations return[2], excluding the endpoint. Explicit binder hotspot values are used directly as0-based positions whereas default binders are0-based; a terminal explicit hotspot can overrun the array. The standard notebook still derives sequence from every PDB residue's raw resname and trims it, bypassing the current CLI's polymer/modified-residue fixes. Crop code also hardcodes the AlphaFold-v3 filename rather than using the downloaded/uploaded `pdb_target_path`, breaking the advertised four-letter PDB/upload crop routes.

Fix: package shared preparation, configuration, paths, sequence extraction and acceptance; use the package from notebooks. Replace invalid defaults, select target chains during cropping, retain mappings and original downloaded artifacts. Pin the same supported dependency/framework revision; VHH currently retrieves an external Poly-P repository and a different dependency tag.

Acceptance: CLI/notebook produce identical processed inputs and maps for standard/VHH cases; inclusive endpoints; crop/uncropped residue equivalence; terminal hotspot; no negative wrapping; uploaded PDB/four-letter PDB/AlphaFold inputs resolve through the same path contract. Full GPU notebook smoke follows CPU frontend parity.

## Historical artifacts, experimental driver and helpers

### C12 — P2, array comparison reproduced: three archived VHH maps differ from current example commands

Locations: `FoldCraft.py:104–125`; `examples/scripts/design_vhh_pd_l1.sh`, `design_vhh_pd_1.sh`, `design_vhh_ifnar.sh`; `examples/cmaps/vhh_*.npy`.

Current VHH hotspot hardcode is26–35,55–59,102–116. Reconstructing provided examples differs from archived PD-L1,PD-1,IFNAR maps by180,170,120 cells respectively; EGFR matches. This is a reproducibility/versioning discrepancy, not evidence the old or new scientific convention is superior.

Fix: version historical conditioning rules and maps; offer validated explicit map input or appropriate VHH override, retaining a clearly named historical reproduction mode. Do not silently rewrite reference arrays.

Acceptance: golden array equality for each historical example; explicit current-method fixtures; manifest says which convention generated results.

### C13 — P2, experimental-only, reproduced/static: default no-op, early import failure, length overshoot and unsafe path interpolation

Locations: `test/FoldCraft_binder.py:43,53–55,69,132–160,164,174,294,311`.

The root-helper import precedes sys.path setup; `python test/FoldCraft_binder.py` does not generally put the repository root on sys.path just because shell CWD is root. The default sample=False has no design branch; executing the actual main AST with actual default arguments and side-effect-free shell stubs creates an empty CSV without a model call. Parsed num_designs is unused. Python's inclusive randint(lo,hi+1) can exceed requested maximum binder length. Shell mkdir/rm interpolate paths literally, so whitespace/metacharacters break file operations or execute unintended local commands.

Fix: proper package/module entry point; implement bounded finite mode or reject it explicitly; inclusive randint(lo,hi); pathlib filesystem APIs; shared validated run config. Treat this driver as experimental or deprecate it until tested.

Acceptance: invoke documented command in subprocess from clean directory/root with dependency stubs; default either designs requested finite count or errors; boundaries including lo=hi; paths with spaces/metacharacters treated literally; no shell invocations for files.

### C14 — P2, newly noted experimental-only, source-confirmed: saved best backbone and recorded last metrics disagree

Locations: `test/FoldCraft_binder.py:202–205`; upstream `colabdesign/af/utils.py:64–71`.

Experimental driver saves PDB with get_best=True but writes `af_model.aux['all']` and gates `af_model.aux['log']` from the last step. The upstream saver selects `_tmp['best']['aux']` locally and does not replace `self.aux`. When best and last differ, MPNN receives a different backbone from the one described/gated by the pickle/log.

Fix: select one snapshot object and derive structure, sequence, metrics, gate and serialized artifact from it. This is a consistency bug independent of whether best or last has superior average design quality.

Acceptance: synthetic best/last objects with deliberately different IDs/coordinates/metrics; all emitted fields and selection decisions must correspond to the chosen object; actual GPU snapshot parity smoke.

### C15 — P3 currently, reproduced: structural helpers cannot yet serve as reliable acceptance oracles

Locations: `biopython_utils.py:136–147,160–189`.

`target_pdb_rmsd` silently truncates chains to min length and drops missing CA atoms separately, so it can report0.0 with incomplete coverage or shift residue correspondence. Probe returns0.0 for a three-residue trajectory against four-residue target without coverage reporting. `calculate_clash_score` pools every MODEL's coordinates without model IDs. A two-model PDB with no clashes in either model yields2 cross-model clashes in the probe. This becomes consequential when C04 emits two models and these helpers are connected to production acceptance.

Fix: explicit model selection/per-model evaluation, shared residue/sequence alignment and paired atom masks, coverage reporting. Define heavy-atom versus CA and inter- versus intra-chain clash policies explicitly; current default ignores same-chain heavy-atom clashes.

Acceptance: mismatched lengths, internal missing atoms, modified residues, insertion codes, translated/rotated structures, multi-model PDBs; no cross-model clash pairs; deterministic coverage thresholds. Keep method calibration separate from mechanical mapping fixes.

## Changes to prior interpretation and implementation order

- Current core scientific/driver code retains the prior findings; the September integration corrected reporting and tests, not these production paths.
- Do not re-file already fixed old problems: CLI inclusive range endpoints, default all-binder hotspot indexing, binder-mask1-based indexing, polymer-aware sequence construction, exact accepted quota, final atomic promotion, MPNN construction hoisting, cached maps and within-trajectory AF validation reuse are present.
- Target missing-data mismatch is broader than numbering gaps: newly reproduced missing N demonstrates why contiguous renumbering alone is inadequate. Dependency docs' missing-CA wording is not the executable filter in this snapshot.
- Do not infer two-model validation from the list of names, or assume multi-model PDBs can be consumed by single-model structural helpers.
- Do not call get_best=False a proven accuracy bug. New pilot data did not show a gain. C14 is instead an actual artifact consistency bug in the experimental driver.
- Do not repeat the old report's9/200 Ig-like fold-retention figure: integration correctly records187/200 RMSD passes and66/200 combined passes.
- Archived-map mismatch is a historical convention issue; version it before benchmarking changes, rather than claiming current conventions are scientifically wrong.

Recommended correctness order: (1) validated shared config, version/provenance and input mapping C01–C03/C08; (2) MPNN mutable sets C05; (3) output/run integrity C07; (4) explicit model contract plus per-model artifacts C04/C15; (5) frontend parity and historical configuration versions C09–C12; (6) repair or isolate experimental driver C13–C14. These can be focused commits/PRs. Then establish reproducible candidate-level structural measurements and benchmark/calibrate acceptance improvements C06. Objective changes, additional predictors and performance reuse should follow this correctness baseline so altered results remain attributable.
