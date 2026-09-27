# Versioned VHH conditioning

The default `foldcraft-127-current-v1` uses 1-based inclusive binder positions
`26–35,55–59,102–116`. All four example scripts explicitly select this convention.
The unchanged `framework/vhh.npy` supplies the 127-residue intra-binder map.

The archived `examples/cmaps/vhh_*.npy` files contain two different conventions:

| Archived example | Convention | Binder positions | Cells differing from current |
| --- | --- | --- | ---: |
| PD-L1 | `foldcraft-127-historical-v0` | `27–35,56–60,103–117` | 180 |
| PD-1 | `foldcraft-127-historical-v0` | `27–35,56–60,103–117` | 170 |
| IFNAR2 | `foldcraft-127-historical-v0` | `27–35,56–60,103–117` | 120 |
| EGFR | `foldcraft-127-current-v1` | `26–35,55–59,102–116` | 0 |

To reproduce a historical map, run the example's target/hotspots with `--vhh
--vhh_convention foldcraft-127-historical-v0` and a fresh output directory. EGFR
uses the current convention even for archive reproduction. This reproduces the
conditioning arrays exactly; it does not reconstruct unrecorded historical
seeds, dependencies or predictions. No convention is claimed to improve accuracy.

The driver records the selected convention and resolved binder positions in
`run.json`. CPU preflight and runtime consume the same selection. Tests verify
all archived maps and masks element-for-element, preserve their SHA-256 hashes,
and verify independently frozen hashes for every current map and mask in
`tests/fixtures/vhh_conventions.json`. Array hashes use C-order little-endian
float64 bytes, independent of NumPy file header versions. No reference array is
rewritten or silently reinterpreted.
