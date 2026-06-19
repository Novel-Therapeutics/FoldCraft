"""Tests for openmm_compare's self-ipTM column dispatch.

Regression guard for the bug where the FORWARDED comparison ranked BoltzProt's
"own top designs" by ``af2_iptm`` -- the *independent AF2 judge* -- instead of
``boltz_iptm``, BoltzProt's *self*-reported confidence. The cause was a ternary
(``"iptm" if "iptm" in cols else "af2_iptm"``) that silently fell back to a judge
column because BoltzProt's CSV names its self-score ``boltz_iptm``, not ``iptm``.
The adversarial case below (boltz_iptm present, bare iptm absent, judge columns
present) is exactly the shape the original code got wrong.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
import openmm_compare


def test_boltzprot_columns_pick_self_not_judge():
    """The exact BoltzProt CSV shape: self-score is boltz_iptm; af2_iptm /
    boltz2_iptm are external judges and must NOT be selected."""
    df = pd.DataFrame(columns=[
        "name", "sequence", "length", "boltz_iptm", "boltz2_iptm",
        "af2_iptm", "openmm_dE"])
    assert openmm_compare.self_iptm_column(df) == "boltz_iptm"


def test_foldcraft_columns_pick_iptm():
    """FoldCraft hallucinates against AF2, so its design-time `iptm` is its
    self-score."""
    df = pd.DataFrame(columns=["name", "sequence", "iptm", "af2_iptm", "openmm_dE"])
    assert openmm_compare.self_iptm_column(df) == "iptm"


def test_iptm_preferred_over_boltz_iptm():
    """Deterministic precedence when both exist (no ambiguity)."""
    df = pd.DataFrame(columns=["iptm", "boltz_iptm"])
    assert openmm_compare.self_iptm_column(df) == "iptm"


def test_only_judge_columns_fails_loud():
    """If only judge columns exist, fail loud rather than silently rank a method
    by another method's model (the failure mode the original ternary hid)."""
    df = pd.DataFrame(columns=["name", "af2_iptm", "boltz2_iptm", "openmm_dE"])
    with pytest.raises(SystemExit):
        openmm_compare.self_iptm_column(df)
