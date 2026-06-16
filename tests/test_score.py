"""Subprocess tests for baseline/score.py.

score.py is what makes the committed baseline report reproducible, so these run
the real script (not a mock) on small fixture run dirs and check it reports the
ipSAE column, computes the success gate correctly, and fails loud on
missing/inconsistent data.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE = os.path.join(REPO, "baseline", "score.py")

HEADER = "name,plddt,ipae,iptm,rmsd,ipsae\n"
# d0,d1,d3 pass all four gates; d2 fails; ipSAE>.3 holds for d0 and d3 only.
ROWS = (
    "d0,0.9,0.2,0.6,1.0,0.5\n"
    "d1,0.9,0.2,0.6,1.0,0.1\n"
    "d2,0.5,0.9,0.1,9.0,0.0\n"
    "d3,0.85,0.3,0.55,2.0,0.4\n"
)


def _write(runs, fold, csv):
    d = runs / fold
    d.mkdir(parents=True)
    (d / "results.csv").write_text(csv)


def _run(runs):
    return subprocess.run([sys.executable, SCORE, str(runs)], capture_output=True, text=True)


def test_reports_ipsae_and_success_gate(tmp_path):
    runs = tmp_path / "runs"
    _write(runs, "demo", HEADER + ROWS)
    r = _run(runs)
    assert r.returncode == 0, r.stderr
    assert "ipSAE>.3" in r.stdout
    row = next(l for l in r.stdout.splitlines() if l.startswith("demo")).split()
    # demo n=4 | pLDDT ipTM iPAE ipSAE RMSD = 3 3 3 2 3 | ALL = 3 (gate excludes ipSAE)
    assert row[1] == "4"
    assert row[2:7] == ["3", "3", "3", "2", "3"]
    assert row[7] == "3"


def test_optional_ipsae_column_omitted_when_absent(tmp_path):
    runs = tmp_path / "runs"
    _write(runs, "demo", "name,plddt,ipae,iptm,rmsd\nd0,0.9,0.2,0.6,1.0\nd2,0.5,0.9,0.1,9.0\n")
    r = _run(runs)
    assert r.returncode == 0, r.stderr
    assert "ipSAE>.3" not in r.stdout
    row = next(l for l in r.stdout.splitlines() if l.startswith("demo")).split()
    assert row[1] == "2" and row[6] == "1"  # n=2, ALL=1 (only d0 passes the gate)


def test_missing_rmsd_fails_loud(tmp_path):
    runs = tmp_path / "runs"
    _write(runs, "demo", "name,plddt,ipae,iptm,ipsae\nd0,0.9,0.2,0.6,0.5\n")
    r = _run(runs)
    assert r.returncode != 0 and "rmsd" in (r.stderr + r.stdout)


def test_inconsistent_ipsae_fails_loud(tmp_path):
    runs = tmp_path / "runs"
    _write(runs, "a_withipsae", HEADER + ROWS)
    _write(runs, "b_noipsae", "name,plddt,ipae,iptm,rmsd\nd0,0.9,0.2,0.6,1.0\n")
    r = _run(runs)
    assert r.returncode != 0 and "ipsae" in (r.stderr + r.stdout).lower()
