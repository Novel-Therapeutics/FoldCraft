"""Strict, CPU-only structural correspondence and energy controls."""
from numbers import Integral
import math
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


class ConvergenceError(ValueError):
    def __init__(self, qc):
        self.qc = qc
        super().__init__(f"OpenMM minimization did not converge: RMS force {qc['rms_force_kj_mol_nm']:.3g} > {qc['tolerance_kj_mol_nm']}")


def force_qc(forces, positions, energy, tolerance, minimized):
    """CPU-testable QC in kJ/mol/nm, nm and kcal/mol."""
    forces, positions = np.asarray(forces), np.asarray(positions)
    if forces.ndim != 2 or forces.shape[1] != 3 or not forces.size or positions.shape != forces.shape:
        raise ValueError('Invalid OpenMM force/position shape')
    if not np.isfinite(forces).all() or not np.isfinite(positions).all() or not math.isfinite(energy):
        raise ValueError('Nonfinite OpenMM energy, force or coordinates')
    rms = float(np.sqrt(np.mean(forces**2)))
    return dict(status='converged' if minimized and rms <= tolerance else 'unconverged' if minimized else 'raw',
                converged=bool(rms <= tolerance) if minimized else None,
                rms_force_kj_mol_nm=rms, max_force_kj_mol_nm=float(np.linalg.norm(forces,axis=1).max()),
                tolerance_kj_mol_nm=tolerance)


def fractional_plddt_percent(values):
    """Hugging Face ESMFold categorical_lddt returns fractions, not percentages."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError('Expected finite ESMFold confidence fractions in [0, 1]')
    return float(100 * values.mean())
