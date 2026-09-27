# Closure validation evidence — 27 September 2026

See [the report](../../../../CLOSURE_VALIDATION.md) for scope and limitations.
JSON/CSV records are copied intact; exported text logs have trailing whitespace
removed. PDBs, pickles, weights and full runtime environments remain under
`/home/bizon/projects/foldcraft-gpu-20260927/` on Bizon.

- `closure-smoke-01/smoke.json`: five supervised GPU cases and exact replay.
- `closure-extra-01/verification.json`: current-runtime numerical parity,
  historical VHH map/mask reproduction, all 220 optimization iterations in
  chunk/merge, verified scheduler resume and GPU cleanup.
- `experimental-ops-02/verification.json`: actual CLI missing-dependency failure
  and a hung-import timeout (3-second limit plus 5-second process cleanup).
- `local-cpu-tests.log` and `bizon-cpu-tests.log`: final 242-test suites.

Reproduction scripts require fresh output directories and adjusted host paths.
For `verify-experimental-ops.py`, the final run used output root
`experimental-ops-02`; the first run was retained separately. Its injected import
sleep tests process supervision only; it does not simulate a successful physical
prediction. Full experimental PyRosetta/BindCraft numerical validation is pending.

Main GPU source did not change after these runs. Experimental source received a
final false-completion guard and was then validated by the final CPU suites and
`experimental-ops-02`. Manifest source hashes identify exact executed versions.
