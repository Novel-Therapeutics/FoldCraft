# Inference provenance evidence — 27 September 2026

See [the report](../../../../INFERENCE_PROVENANCE.md) for scope and interpretation.
JSON manifests/receipts and CSVs are copied without modification. Exported text
logs have trailing whitespace removed. Large PDB/pickle outputs, model weights
and the private worker package remain on Bizon under
`/home/bizon/projects/foldcraft-gpu-20260927/`.

`bundle-smoke-01/smoke.json` records all five cases and exact replay.
`bundle-campaign-01/verification.json` records unchanged/replaced/restored resume,
prior-runtime numerical parity, preserved historical artifacts and GPU cleanup.
`verify-bundle-campaign.py` reproduces the scheduler test in a fresh output folder;
its host paths must be adjusted before reuse. It only mutates its private copy of
MPNN weights and restores the original byte in a finally block.

The five-case smoke suite preceded a final absolute-data-directory normalization;
the scheduler campaign and final CPU suites used that final source. The smoke
suite already passed absolute weight paths. Source SHA-256 values in run manifests
identify the exact versions exercised.
