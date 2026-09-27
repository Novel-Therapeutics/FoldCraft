"""Amber/GBN2 interaction-energy diagnostic, not a binding free-energy or accuracy oracle.

Minimized scores are published only after a finite Cartesian force convergence
check. Zero iterations explicitly scores the repaired, unminimized geometry.
QC and failure reasons are retained in score_openmm.scores.json. Historical
constrained, unconverged energies are invalidated by the new protocol identity.
"""
import math
import argparse
import os
import sys
import random
import numpy as np
from pathlib import Path

try:
    from .score_cache import ScoreSession
except ImportError:
    from score_cache import ScoreSession

import pandas as pd
try:
    from .gates import af2_mask
except ImportError:
    from gates import af2_mask
from openmm import (LangevinIntegrator, Context, Platform, OpenMMException,
                    LocalEnergyMinimizer, MinimizationReporter)
from openmm.app import ForceField, Modeller, NoCutoff, forcefield
from openmm.unit import kilocalorie_per_mole, kelvin, picosecond, picoseconds, kilojoule_per_mole, nanometer
from pdbfixer import PDBFixer
try:
    from .structure_checks import force_qc, ConvergenceError
except ImportError:
    from structure_checks import force_qc, ConvergenceError

# Amber ff14SB protein params + GBN2 implicit solvent (Onufriev-Bashford-Case);
# implicit solvent so the interaction energy includes a desolvation term without
# the cost/ambiguity of explicit waters. amber14/protein.ff14SB is the maintained
# OpenMM build of ff14SB.
FF = ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")
_PLATFORM = {"p": None}


def _platform(name):
    if name:
        return Platform.getPlatformByName(name)
    for cand in ("CUDA", "OpenCL", "CPU"):           # fastest available
        try:
            return Platform.getPlatformByName(cand)
        except Exception:
            continue
    return None


def _single_point(topology, positions, keep_chain=None):
    """Potential energy (kcal/mol) of `topology` at `positions`; if keep_chain is
    given, first delete every other chain (isolated-chain energy at bound coords)."""
    modeller = Modeller(topology, positions)
    if keep_chain is not None:
        modeller.delete([c for c in modeller.topology.chains()
                         if c.id != keep_chain])
    system = FF.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                             constraints=None)
    integ = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    ctx = Context(system, integ, _PLATFORM["p"]) if _PLATFORM["p"] else \
        Context(system, integ)
    ctx.setPositions(modeller.positions)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    del ctx, integ, system
    return e.value_in_unit(kilocalorie_per_mole)


class IterationCount(MinimizationReporter):
    def __init__(self):
        super().__init__()
        self.iterations = 0

    def report(self, iteration, x, grad, args):
        self.iterations = iteration + 1
        return False



def interface_dE(pdb, min_iters, target_chain="A", binder_chain="B", *, tolerance=10., seed=0, return_qc=False):
    """Rigid-body interface interaction energy E(AB) - E(A) - E(B) in kcal/mol.

    PDBFixer adds missing heavy atoms + hydrogens (AF2/Boltz outputs are
    heavy-atom only), then the *complex* is energy-minimized so the per-method
    clash level is normalised identically before the three single-point energies
    are read off the one minimized geometry.
    """
    if min_iters < 0 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Iterations must be nonnegative and tolerance must be finite and positive')
    random.seed(seed)
    fixer = PDBFixer(filename=pdb, platform=_PLATFORM['p'])
    fixer.findMissingResidues()
    if fixer.missingResidues:
        raise ValueError('Missing polymer residues: refusing to invent a scoring geometry')
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=seed)
    fixer.addMissingHydrogens(7.0)
    chains = {c.id for c in fixer.topology.chains()}
    if target_chain == binder_chain or chains != {target_chain, binder_chain}:
        raise ValueError(f"{pdb}: expected exactly two chains, found {sorted(chains)}; requested "
                         f"{target_chain}/{binder_chain}")

    modeller = Modeller(fixer.topology, fixer.positions)
    system = FF.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                             constraints=None)
    integ = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    ctx = Context(system, integ, _PLATFORM["p"]) if _PLATFORM["p"] else \
        Context(system, integ)
    ctx.setPositions(modeller.positions)
    before = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    if not math.isfinite(before):
        raise ValueError('Nonfinite initial OpenMM energy')
    counter = IterationCount()
    if min_iters:
        LocalEnergyMinimizer.minimize(ctx, tolerance=tolerance, maxIterations=min_iters, reporter=counter)
    state = ctx.getState(getPositions=True, getEnergy=True, getForces=True)
    e_ab = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    pos = state.getPositions()
    qc = force_qc(state.getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer),
                  state.getPositions(asNumpy=True).value_in_unit(nanometer),e_ab,tolerance,bool(min_iters))
    qc.update(initial_energy_kcal_mol=float(before), final_energy_kcal_mol=float(e_ab),
              max_iterations=min_iters, iterations=counter.iterations, seed=seed, platform=ctx.getPlatform().getName(), constraints='none')
    top = modeller.topology
    del ctx, integ, system
    if min_iters and not qc['converged']:
        raise ConvergenceError(qc)

    e_a = _single_point(top, pos, keep_chain=target_chain)
    e_b = _single_point(top, pos, keep_chain=binder_chain)
    if not all(math.isfinite(e) for e in (e_ab, e_a, e_b)):
        raise ValueError('Nonfinite OpenMM energy')
    qc.update(target_energy_kcal_mol=float(e_a), binder_energy_kcal_mol=float(e_b))
    scores = (round(e_ab - e_a - e_b, 2), round(e_ab, 1))
    return (*scores, qc) if return_qc else scores


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv + designs/<name>.pdb")
    ap.add_argument("--sample", type=int, default=0, help="score only N random rows")
    ap.add_argument("--af2-pass-only", action="store_true",
                    help="score only designs that clear the AF2 gate")
    ap.add_argument("--min-iters", type=int, default=500,
                    help="complex minimization iterations (0 = score raw structure)")
    ap.add_argument("--platform", default=None, help="CUDA|OpenCL|CPU (default: best)")
    ap.add_argument('--force-tolerance',type=float,default=10.,help='RMS force convergence threshold in kJ/mol/nm')
    ap.add_argument('--seed',type=int,default=0,help='PDB repair random seed')
    args = ap.parse_args()
    if not math.isfinite(args.force_tolerance) or args.force_tolerance <= 0:
        ap.error('--force-tolerance must be finite and positive')
    if not 0 <= args.seed < 2**32:
        ap.error('--seed must be a 32-bit nonnegative integer')
    if args.sample < 0:
        ap.error('--sample must be nonnegative')
    if args.min_iters < 0:
        ap.error('--min-iters must be nonnegative')

    _PLATFORM["p"] = _platform(args.platform)
    if _PLATFORM['p'] and _PLATFORM['p'].getName() == 'CUDA':
        _PLATFORM['p'].setPropertyDefaultValue('Precision', 'mixed')
        _PLATFORM['p'].setPropertyDefaultValue('DeterministicForces', 'true')
    print(f"OpenMM platform: {_PLATFORM['p'].getName() if _PLATFORM['p'] else 'default'}")

    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    ff_root = Path(forcefield.__file__).parent/'data'
    platform_properties = {p:_PLATFORM['p'].getPropertyDefaultValue(p) for p in _PLATFORM['p'].getPropertyNames()} if _PLATFORM['p'] else {}
    with ScoreSession(csvf, __file__, ['openmm_dE', 'openmm_e_complex'],
                      dict(min_iters=args.min_iters, tolerance=args.force_tolerance, seed=args.seed, platform=_PLATFORM['p'].getName() if _PLATFORM['p'] else 'default', platform_properties=platform_properties, constraints='none', forcefield='amber14/protein.ff14SB+implicit/gbn2'), extra_inputs=(), structure=True, model_inputs={'protein_ff':ff_root/'amber14/protein.ff14SB.xml','solvent_ff':ff_root/'implicit/gbn2.xml'}) as session:
        df = pd.read_csv(csvf)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        for col in ("openmm_dE", "openmm_e_complex"):
            if col not in df.columns:
                df[col] = pd.NA

        df = session.prepare(df)
        session.publish(df)  # publish invalidation before expensive inference
        todo = df[df["openmm_dE"].isna()]
        if args.af2_pass_only:
            todo = todo[af2_mask(todo)]
        if args.sample and len(todo) > args.sample:
            todo = todo.sample(args.sample, random_state=0)
        print(f"{args.design_dir}: {len(todo)} to score "
              f"({len(df) - len(todo)} already done / skipped)")

        failures = 0
        for i, (idx, row) in enumerate(todo.iterrows(), 1):
            pdb = os.path.join(args.design_dir, "designs", f"{row['name']}.pdb")
            if not os.path.exists(pdb):
                sys.exit(f"design PDB missing: {pdb}")
            try:
                dE, e_ab, qc = interface_dE(pdb, args.min_iters, tolerance=args.force_tolerance, seed=args.seed, return_qc=True)
                session.annotate(row['name'], **qc)
            except (OpenMMException, ValueError) as exc:
                # fail loud per design but keep the batch going; record nothing so a
                # rerun retries this row rather than silently treating it as scored.
                failures += 1
                session.annotate(row['name'], **getattr(exc,'qc',dict(status='failed',error=str(exc))))
                session.publish(df)
                print(f"  [{i}/{len(todo)}] {row['name']}: FAILED -- {exc}")
                continue
            df.loc[idx, ["openmm_dE", "openmm_e_complex"]] = [dE, e_ab]
            if i % 10 == 0 or i == len(todo):
                session.publish(df)              # checkpoint for resumability
                print(f"  [{i}/{len(todo)}] {row['name']}: dE={dE} kcal/mol")
        session.publish(df)
        print(f"wrote openmm_dE -> {csvf}")
        if failures:
            raise SystemExit(f'{failures} OpenMM candidates failed QC; scores remain unknown, see scorer sidecar')


if __name__ == "__main__":
    main()
