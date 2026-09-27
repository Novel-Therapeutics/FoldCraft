# Validated improvement evidence — 27 September 2026

All four follow-up steps are complete: freeze the baseline, measure/promote lossless storage, execute the expanded paired structural benchmark, and apply the prespecified promotion criteria.

- Baseline: `validated-baseline-2026-09-27`, commit `c5e3f78`.
- Frozen benchmark implementation: `3d293b5`. The pre-scoring analyzer correction is recorded in `expanded/analysis_implementation_correction.json`; it implements the already-declared ranking noninferiority limits without changing thresholds.
- CPU regression suites: 249 passed on macOS and 249 in the validated Linux runtime.
- Performance: 12 cold/warm runs across three complex sizes; all nine non-reference outputs passed exact replay. Candidate pickle storage fell 98.7–98.9%.
- Expanded study: 48 full trajectories, 24 exact paired backbones, 96 candidates, 192 AF2 predictions, 96 ESMFold evaluations and 192 Boltz predictions. No missing evaluations.
- Final audits: prediction identities/arrays/interfaces, ranking inputs, pairing metadata, scorer signatures and raw Boltz selection hashes passed. Evaluator source/package fingerprints match before and after.
- Default-storage GPU replay: implicit compact mode matched the full-artifact reference exactly (PAE and coordinate maximum deltas zero).
- Scientific decision: retain existing temperature and ranking defaults. Neither improved the held-out primary proxy; only one untouched target family and no assay labels are available.

`performance/`, `expanded/` and `default-storage/` contain compact receipts, results and logs. The initial profiling setup failure (missing psutil before any GPU model launched) is retained under `performance/setup_attempt_01.*`. The subsequent benchmark used OS telemetry without installing dependencies.

Large arrays, AF2/Boltz structures, model weights and full optimization histories are not duplicated here. Raw AF2/Boltz outputs and histories remain on Bizon under `/home/bizon/projects/foldcraft-gpu-20260927/`; run manifests retain their hashes. ESMFold stores scores/provenance, not monomer coordinates. Replaying its recorded evaluator is required to reproduce those RMSDs.

The operational runner scripts preserve the exact machine-specific commands. Use the repository benchmark scripts with new output roots to reproduce elsewhere. Exported log trailing whitespace is normalized. `evidence_manifest.json` hashes every retained file except itself.
