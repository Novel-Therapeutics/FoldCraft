"""Guard: the committed baseline scripts must not hard-code one machine's layout.

The baseline runner/scorers are meant to run on any machine (and for upstream
reviewers), deriving paths from the script location and relying on the caller's
active conda env. This catches regressions like an absolute ``/root/FoldCraft``
or ``/opt/conda`` path or an in-script ``conda activate`` creeping back in.
"""
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(REPO, "baseline")

# absolute paths / activations that tie a script to one machine's layout
FORBIDDEN = re.compile(r"/root/FoldCraft|/opt/conda|/home/\w+/miniconda|conda activate")


def _scripts():
    return sorted(
        glob.glob(os.path.join(BASELINE, "*.sh"))
        + glob.glob(os.path.join(BASELINE, "*.py"))
    )


def test_no_machine_specific_paths_in_baseline_scripts():
    offenders = []
    for path in _scripts():
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                if FORBIDDEN.search(line):
                    rel = os.path.relpath(path, REPO)
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "machine-specific paths in baseline scripts:\n" + "\n".join(offenders)
