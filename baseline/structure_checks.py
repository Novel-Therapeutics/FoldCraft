"""Strict, CPU-only structural correspondence and energy controls."""
from numbers import Integral
import numpy as np
from Bio.PDB.Polypeptide import is_aa


def chain_ca(chain):
    residues = [r for r in chain if is_aa(r)]
    if not residues or any('CA' not in r for r in residues):
        raise ValueError('Every polymer residue must have a CA atom')
    ids = [r.id[1] for r in residues]
    if any(r.id[2].strip() for r in residues) or ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError('RMSD requires continuous residue numbering without insertion codes')
    atoms = [r['CA'] for r in residues]
    if not np.isfinite([a.coord for a in atoms]).all():
        raise ValueError('CA coordinates must be finite')
    return atoms


def matching_ca(a, b):
    if len(a) != len(b) or len(a) < 3:
        raise ValueError('RMSD requires equal, complete chains with at least 3 CA atoms')


def minimize_if_requested(minimizer, context, iterations):
    if not isinstance(iterations, Integral) or iterations < 0:
        raise ValueError('Minimization iterations must be a nonnegative integer')
    if iterations:
        minimizer.minimize(context, maxIterations=iterations)
