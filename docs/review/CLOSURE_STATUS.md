# Review closure status — 27 September 2026

This is an item-by-item audit of [DEEP_REVIEW.md](DEEP_REVIEW.md), including the
CPU, GPU, scorer and reliability patches. **The repository is not certified
bug-free, and the original plan is not wholly complete.** “Fixed” refers to the
specific reproduced defect. “Restricted” means unsupported cases fail explicitly
or are excluded from the supported workflow. “Partial” identifies remaining work.

## Input and model contracts

| ID | Status | Evidence and remaining scope |
| --- | --- | --- |
| C1 | Fixed | Canonical residue mapping in `input_validation.py`; offset-numbered inputs tested. |
| C2 | Restricted | Gaps, insertion codes and missing backbone atoms fail before inference. General repair/partial-chain support remains unimplemented. |
| C3 | Fixed | Hotspot/map/control validation, including finite positive runtime limits; adversarial CPU tests. |
| C4 | Fixed | Explicit current/historical VHH conventions; all four archived maps/masks reproduce exactly. Golden input/array hashes and a historical-mode GPU run verify migration without rewriting archives. See [VHH conventions](../../examples/VHH_CONVENTIONS.md). |
| C5 | Fixed | Terminal residue, exact mutable set, empty-redesign handling, fixed-interface preservation; CPU and GPU checks. |
| C6 | Fixed on supported main path | Explicit per-model execution, artifacts and receipts. The A/B drivers and independent AF2 scorer intentionally retain explicit single-model protocols. Two-model selection remains optional. |

## Run integrity and orchestration

| ID | Status | Evidence and remaining scope |
| --- | --- | --- |
| R1 | Fixed | Live output defaults and manifest/artifact completion; archived CSVs cannot masquerade as executable runs. |
| R2 | Fixed on supported main path | Signature-checked chunks, atomic merge, tamper tests and checkpoint-aware resume; see R6. |
| R3 | Fixed for supported CLI | Atomic checkpoints, exclusive output, attempt budgets and external wall-time limit. Timeout/cancellation terminate owned processes and cannot publish completion. Recovery retains artifacts but restarts the interrupted trajectory. |
| R4 | Fixed | Solo retry holds its reservation for its full lifetime; scheduling regression tests. |
| R5 | Fixed | Campaign failures return nonzero; shell subprocess tests. |
| R6 | Fixed on supported main path | Design manifests and scheduler signatures fingerprint actual AF2/MPNN checkpoints, relevant loader sources and VHH conditioning. Selected-worker discovery, load/completion guards and in-place replacement tests enforce compatibility. Historical runs remain readable but cannot resume without matching provenance. See [inference provenance](../../INFERENCE_PROVENANCE.md). |

## Scoring and interpretation

| ID | Status | Evidence and remaining scope |
| --- | --- | --- |
| E1 | Fixed reporting | Consensus reports known/unknown counts. Generic summaries now reject missing/nonfinite acceptance metrics. Historical unmeasured candidates remain unmeasured. |
| E2 | Fixed | Isolated Boltz invocations, exact output selection receipts, checkpoint-aware cache; real independent scorer tests. |
| E3 | Fixed | Per-scorer sidecars and locked atomic column publication; stale/concurrent writer tests. |
| E4 | Restricted | Complete-chain correspondence is required, instead of prefix truncation. Partial coverage/alignment remains future work. |
| E5 | Fixed | Zero iterations bypass minimization; raw and minimized protocols distinguished and GPU-tested. |
| E6 | Fixed numerical QC | Valid PAE dimensions/finiteness, finite energy/coordinates/forces and verified convergence. Energies remain geometry diagnostics; physical acceptance is not calibrated. |
| E7 | Fixed A/B configuration selection | A/B fold hotspots use the selected reproduction configuration; protocol is recorded. Historical data is not relabeled. |
| E8 | Partial scientific validation | New experiments pair applicable streams and aggregate siblings by trajectory. Generic candidate reports no longer print misleading independent-binomial intervals. The pilot reports separate fold/contact metrics. Complete main-path optimization histories now retain every iteration, actual model dispatch and stage seed, with hashed merged artifacts. Family-level uncertainty and a held-out assay panel remain open; no binding-accuracy claim is supported. Independent Boltz random streams in the pilot were not coupled across arms. |

## Other reviewed surfaces

- **Notebook correctness:** repaired by routing local notebooks through the shared
  CLI; ordered execution tested with CPU inference doubles. The README's upstream
  Colab links are external copies and do not inherit these repairs.
- **Experimental binder:** startup scope error, ignored conditioning controls,
  atomic output, numerical QC, seeded execution and supervised timeout are fixed
  and CPU/real-process tested. Full numerical validation remains blocked because
  the tested runtime lacks PyRosetta/BindCraft. An existing licensed environment
  is required; the main GPU suite cannot certify this separate scoring path.
- **Multi-model clash mixing:** fixed through model-specific geometry.
- **Installation:** fresh Linux x86_64/Python 3.12 environment recreated from the
  tested pinned stack. GPU kernels and supervised inference were exercised.
  Other GPU platforms and the independent scorer environments are separate profiles.
- **Performance:** artifact compaction, model reuse, host-memory profiling and
  deduplication remain measured optimization work, not closed bug fixes.
- **Runtime limits:** the main CLI, its scheduler chunks and the experimental
  binder are supervised. Standalone scorers and historical A/B scripts are not
  automatically covered. The benchmark runners retain their own external limits.
  Hard-killing a supervisor or a host failure can leave an interrupted record;
  automatic mid-trajectory recovery and containment of deliberately detached
  child sessions are not provided.

## Additional findings during closure

Fixed-count `attempts.json` used “accepted” to mean retained, rather than passing
validation. It now records the actual ensemble decision; rejected candidates
remain saved. The pilot used correct validation receipts, so its conclusions do
not change. Candidate-summary missing-metric handling and binomial intervals
were also corrected in this pass.

## Next gate

R6 provenance and C4 VHH convention migration are implemented and GPU-tested.
Complete experimental PyRosetta validation when its licensed runtime is available.
For accuracy work, freeze the expanded held-out protocol and use paired analysis.
Do not reopen historical data as if it had the new provenance or paired design.

Validation and commands: [RELIABILITY_FIXES.md](../../RELIABILITY_FIXES.md) and
[INFERENCE_PROVENANCE.md](../../INFERENCE_PROVENANCE.md) and
[CLOSURE_VALIDATION.md](../../CLOSURE_VALIDATION.md).
