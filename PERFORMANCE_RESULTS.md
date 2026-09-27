# Performance results — 27 September 2026

Baseline: `validated-baseline-2026-09-27` (`c5e3f78`). Protocol/code: `3d293b5`.

Compact storage passed the prespecified promotion gate. Scientific predictions are unchanged.

| Complex | Full candidate MB | Compact candidate MB | Reduction | Full warm seconds | Compact warm seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 81.59 | 1.08 | 98.67% | 31.04 | 30.03 |
| medium | 121.07 | 1.48 | 98.77% | 34.20 | 34.24 |
| long | 272.14 | 2.92 | 98.93% | 43.53 | 42.47 |

Bytes cover four candidate predictions (two candidates × two models). Trajectory artifacts are also compacted. MB are decimal.

| Complex | Mode | Cold seconds | Warm host RSS GiB | Warm device used GiB |
| --- | --- | ---: | ---: | ---: |
| short | full | 126.55 | 2.35 | 5.13 |
| short | compact | 122.28 | 2.51 | 5.13 |
| medium | full | 130.67 | 2.82 | 7.13 |
| medium | compact | 128.58 | 2.79 | 7.13 |
| long | full | 150.47 | 3.15 | 11.13 |
| long | compact | 151.55 | 2.82 | 11.13 |

Every other mode/phase was compared with its cold full-artifact reference: sequences, decisions, metrics, PAE and unrounded coordinates agree exactly. PAE/coordinate maximum deltas are zero. All 249 CPU tests passed on macOS and the validated Linux runtime. They exercise downstream array access and model-specific receipts. A final GPU run without an explicit storage flag used compact output and passed exact replay against full artifacts.

The demonstrated gain is storage reduction. These single measurements do not establish a speedup or a reliable memory reduction. Host RSS is sampled process-tree RSS and can double-count shared pages; device memory includes its baseline. See the JSON for sample counts.

Full diagnostic tensors remain available with `--artifact_mode full`; compact arrays are not quantized. No model reuse or deduplication speedup is claimed.

Evidence: [improvements-2026-09-27](docs/review/evidence/improvements-2026-09-27/).
