# Review evidence

These scripts reproduce/audit behavior of the reviewed commit
`e485f3244e47464ae6083c193f234e91c099e0c5`. They are diagnostic evidence, not the
implementation test suite: several deliberately assert that current defects exist.
They should fail or need replacement once those defects are fixed.

Run from the repository root in a CPU environment with `requirements-dev.txt`:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/review/evidence/current_data_audit.py
PYTHONDONTWRITEBYTECODE=1 python docs/review/evidence/current_ab_summary.py
PYTHONDONTWRITEBYTECODE=1 python docs/review/evidence/current_ops_probes.py
PYTHONDONTWRITEBYTECODE=1 python docs/review/evidence/current_model_probes.py
```

The correctness probe additionally needs the source-only ColabDesign checkout at
`e31a56fe1d9b4de25c8697f3a28b75892941cc72`. It extracts unchanged dependency
parsing/dispatch functions without running neural inference. Set its location:

```sh
COLABDESIGN_REVIEW_SOURCE=/path/to/ColabDesign PYTHONDONTWRITEBYTECODE=1 \
  python docs/review/evidence/current_correctness_probes.py
```

The local review used `/private/tmp/foldcraft-review-colabdesign` (also the script's
fallback). This is a source-contract check against that revision, not evidence
of the dependency revision that generated the historical result tables.

The operational/model probes use temporary artifacts and fake prediction/process
boundaries; they do not launch GPU inference, cloud jobs or paid API requests.
Data summaries read the committed CSVs. Bootstrap results are exploratory:
100,000 trajectory resamples, seed 20260926, only 15 trajectories per arm and one
target/fold; no multiple-comparison adjustment or new experiments.

Captured JSON/text outputs reflect the reviewed commit. Appendix issue IDs are
local to each domain; DEEP_REVIEW.md and CHANGE_PLAN.md define the consolidated
priorities. No raw design structures or model weights are included here.
