# Paired structural pilot

Exploratory results; no production defaults were changed. No binding-accuracy claim is supported.

| Arm | Candidates | Single-model pass | Two-model pass | Template RMSD (Å) | Epitope coverage | ESMFold RMSD (Å) | Boltz ipTM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 8 | 0.250 | 0.250 | 4.244 | 0.069 | 3.344 | 0.523 |
| mpnn_temp_02 | 8 | 0.250 | 0.250 | 3.759 | 0.083 | 2.336 | 0.608 |
| normalized_pairs | 8 | 0.000 | 0.000 | 27.295 | 0.234 | 11.925 | 0.793 |

Means give each case/seed trajectory equal weight. Sibling MPNN samples are not independent replicates. Missing measurements remain unknown.

See `paired_blocks.csv` and `analysis.json` for paired changes, coverage counts and selection comparisons.

No assay labels; no binding-accuracy conclusion.
Two target families, one scaffold, four paired blocks; no promotion or significance claim.
Independent metrics are computational proxies. OpenMM is a geometry diagnostic.
Runtime includes compilation and all inference; persistent cache was shared.
