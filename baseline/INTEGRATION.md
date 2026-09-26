# Integration baseline

The September 2026 integration brings `pure-tier-refactor` (original tip
`645fea86c71b084986889f302bbecb544ad0706f`) into the organization's `main`.
It preserves historical measurements and model behavior while clarifying
experimental status and interpretation. The annotated integration tag identifies
the exact baseline for subsequent correctness and accuracy work.

## Scope

- Includes helper fixes, model reuse, example assets, CPU regression tests,
  campaign scheduling, scoring tools, and historical June 2026 measurements.
- `baseline/ab_*.py` and the optional model scorers are experimental tools.
  Their changes to objectives/checkpoint selection are not production defaults.
- The reported ~24% performance gain is historical evidence for model reuse;
  it was not independently measured during this integration.
- CPU tests cover helpers and scheduling/scoring logic. Passing them does not
  validate GPU inference, prediction parity, or binding accuracy.
- No new GPU experiments or API design campaigns are part of this integration.

## Known issues for the correctness-and-validation branch

These are accepted baseline limitations, not claims that the integration fixes
them. Avoid using this baseline as a validated production release.

1. Target residue numbering versus array positions, missing residues, and invalid
   hotspot ranges need consistent mapping and validation.
2. MPNN terminal-residue coverage is fixed in one A/B helper but not the main
   driver; empty fixed-position selections also need handling.
3. Validation lists two AF2 model names without explicitly setting `num_models=2`.
   With the reviewed ColabDesign defaults this runs one model. Historical results
   must not be relabeled as a verified two-model ensemble.
4. Main candidate acceptance does not check intended-fold retention, the requested
   epitope, or clashes. Offline `score.py` additionally requires template RMSD.
5. Contact masks conflate unknown pairs with negative contacts, loss scaling
   depends on system size, and VHH example maps need consistency checks.
6. Sampling can run indefinitely without enough passing designs. Reusing an
   output directory can retain stale completion markers; partial checkpoint
   writes and scheduler merge writes need stronger interruption handling.
7. Full model outputs consume substantial memory/storage. Dependencies, seeds,
   and configuration manifests need recording before controlled comparisons.
8. RMSD helpers use positional truncation rather than validated residue mapping.
   Experimental driver and notebook issues remain outside CPU coverage.

## Benchmark interpretation

Read [REPORT.md](REPORT.md) and [ACCURACY_RESULTS.md](ACCURACY_RESULTS.md) with
these limits: one target; correlated sequences within trajectories; unequal
cross-method generation/selection budgets; scoring on predicted structures;
no experimental labels. Raw structures are not in the repository, so committed
CSV summaries alone are insufficient to repeat structural scoring.

Future comparisons should preserve this baseline, use matched seeds and equal
budgets, report fold and interface criteria separately, and estimate uncertainty
at the trajectory level across several folds and targets. Determine sample size
from the effect of interest and pilot variance. No tested loss/checkpoint change
has yet demonstrated a general accuracy improvement.

## Integration checks (September 26, 2026)

On macOS with Python 3.12.7 and the declared CPU dependencies:

- 113 tests passed with `python -m pytest -q -p no:cacheprovider`.
- Python AST parsing passed for 33 files; `bash -n` passed for 16 scripts.
- Scheduler dry-run planned 24 chunks across six folds without launching jobs.
- Baseline/reproduction summaries and the OpenMM comparison recalculated from
  the committed tables. The report's Ig-like RMSD count was corrected to 187/200
  (previously 9/200); combined confidence-plus-RMSD passes are 66/200.
- `git diff --check` passed. GPU inference and scoring were not rerun.

The portability guard permits Conda setup instructions in `lambda_setup.sh`,
while retaining machine-path checks everywhere and activation restrictions for
runtime scripts. Historical metric tables and inference code were not changed
by the integration corrections.
