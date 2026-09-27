# Pilot evidence

Generated on Bizon from execution code at `3322ec6`. The protocol and source/input
hashes were frozen before execution. See the root `PILOT_RESULTS.md` report.

- `protocol.json`, `execution.json`, `evaluation.json`: protocol, commands,
  return codes and measured stage times.
- `analysis.json`, `paired_blocks.csv`, `REPORT.md`: generated paired analysis.
- `pool/results.csv`, `pool/pool_manifest.json`, `pool/designs/`: all 24 candidate
  measurements, ancestry/input hashes and primary-model structures.
- `pool/score_*.scores.json`: checkpoint/version provenance, values and OpenMM QC.
- Per-run records and validation receipts identify both models and raw artifacts.
- `artifact-inspection.json`, `resource-audit.json`, `scoring-audit.json`:
  successful integrity and coverage audits.
- Boltz selection receipts and confidence JSONs identify the exact scored outputs.
- Generation GPU CSVs are two-second samples. Evaluation VRAM was not sampled.

Large inference pickles, full secondary-model structures, Boltz structures and
weights remain on Bizon under `/home/bizon/projects/foldcraft-gpu-20260927`.
They are identified by hashes in the retained records. The standalone evaluator
and exporter are included; they contain this run's absolute remote paths. The
artifact audit helper runs from the repository with `PYTHONPATH=.`. Logs were
exported with trailing whitespace removed; source structures were copied intact.
