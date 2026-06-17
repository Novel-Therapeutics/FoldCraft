"""Tests for the target/binder split add_ipsae derives from a design PDB.

This is the logic the P1 fix turns on: ipSAE must split the PAE matrix at the
*actual* binder boundary of each design (chain A = target, chain B = binder),
not a binder length looked up from a global config -- which was wrong for runs
built from other templates (e.g. baseline/repro, whose folds reuse the same
names with different binder lengths).
"""
import os
import sys

import pytest
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
import add_ipsae


def test_chain_lens_from_two_chain_complex(two_chain_complex):
    # fixture: chain A has 1 residue (target), chain B has 2 (binder)
    tlen, blen = add_ipsae.complex_chain_lens(two_chain_complex)
    assert (tlen, blen) == (1, 2)


def test_chain_lens_ignores_hetero_residues(two_chain_complex, tmp_path):
    # add a water (HETATM) to chain B; it must NOT count toward the binder length
    s = PDBParser(QUIET=True).get_structure("c", two_chain_complex)
    from Bio.PDB.Residue import Residue
    from Bio.PDB.Atom import Atom
    hoh = Residue(("W", 3, " "), "HOH", "")
    hoh.add(Atom("O", (50.0, 0.0, 0.0), 0.0, 1.0, " ", "O", 99, element="O"))
    s[0]["B"].add(hoh)
    out = tmp_path / "withhet.pdb"
    io = PDBIO(); io.set_structure(s); io.save(str(out))

    tlen, blen = add_ipsae.complex_chain_lens(str(out))
    assert (tlen, blen) == (1, 2)  # HOH excluded


def test_chain_lens_fails_loud_when_binder_chain_absent(pdb_1qys):
    # 1qys1.pdb is a single-chain monomer (chain A only) -> no binder chain B
    with pytest.raises(ValueError, match="missing chain B"):
        add_ipsae.complex_chain_lens(pdb_1qys)
