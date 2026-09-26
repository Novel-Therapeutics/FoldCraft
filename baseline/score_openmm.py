"""Family-neutral physics oracle: an open-source (no-license) interface energy to
break the FoldCraft-vs-BoltzProt tie that the ML oracles can't.

The ML judges are circular here -- AF2 is FoldCraft's family (biased toward it) and
open Boltz-2 turned out non-discriminating (it passes ~73-84% of *everything*), so
the AF2 n Boltz-2 consensus collapses to AF2 and can't honestly rank the two
methods. PyRosetta ddG would be the standard tiebreak but needs a *commercial*
license (Novel-Therapeutics is commercial). OpenMM + the Amber ff14SB / GBN2
implicit-solvent force field is fully open (MIT/LGPL) and family-neutral: it judges
each design on its own predicted complex by molecular-mechanics energy, which --
unlike buried-surface-area -- actually penalises clashes and rewards
complementarity, the thing that separates a real interface from a placed one.

Metric (per design): the rigid-body interface interaction energy

    openmm_dE = E(complex) - E(target alone) - E(binder alone)        [kcal/mol]

all three single-point energies taken on the *same* coordinates after a short
minimization of the complex (which removes the minor clashes AF2/Boltz structures
carry, identically for both methods). More negative = more favourable interface.
This is an interaction energy, not a full binding free energy (no unbound-state
relaxation, no entropy) -- but it is an unbiased, discriminating, open metric,
which is exactly what the benchmark is missing.

Layout/idempotency/usage mirror the other baseline scorers: a design dir has
results.csv (with a 'name' column) + designs/<name>.pdb, complex = chain A
(target) + chain B (binder). Rows already carrying openmm_dE are skipped.
Runs on CPU or GPU; the CUDA platform is ~30x faster -- run on reg-box-1.
Requires the `mm` conda env (openmm + pdbfixer, conda-forge).

Usage:  python baseline/score_openmm.py <design_dir> [--sample N] [--af2-pass-only]
        [--min-iters K] [--platform CUDA|CPU]
"""
import argparse
import os
import sys

import pandas as pd
from openmm import (LangevinIntegrator, Context, Platform, OpenMMException,
                    LocalEnergyMinimizer)
from openmm.app import ForceField, Modeller, NoCutoff, HBonds
from openmm.unit import kilocalorie_per_mole, kelvin, picosecond, picoseconds
from pdbfixer import PDBFixer

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
                             constraints=HBonds)
    integ = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    ctx = Context(system, integ, _PLATFORM["p"]) if _PLATFORM["p"] else \
        Context(system, integ)
    ctx.setPositions(modeller.positions)
    e = ctx.getState(getEnergy=True).getPotentialEnergy()
    del ctx, integ, system
    return e.value_in_unit(kilocalorie_per_mole)


def interface_dE(pdb, min_iters, target_chain="A", binder_chain="B"):
    """Rigid-body interface interaction energy E(AB) - E(A) - E(B) in kcal/mol.

    PDBFixer adds missing heavy atoms + hydrogens (AF2/Boltz outputs are
    heavy-atom only), then the *complex* is energy-minimized so the per-method
    clash level is normalised identically before the three single-point energies
    are read off the one minimized geometry.
    """
    fixer = PDBFixer(filename=pdb)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    chains = {c.id for c in fixer.topology.chains()}
    if target_chain not in chains or binder_chain not in chains:
        raise ValueError(f"{pdb}: chains {sorted(chains)} lack "
                         f"{target_chain}/{binder_chain}")

    modeller = Modeller(fixer.topology, fixer.positions)
    system = FF.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds)
    integ = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    ctx = Context(system, integ, _PLATFORM["p"]) if _PLATFORM["p"] else \
        Context(system, integ)
    ctx.setPositions(modeller.positions)
    LocalEnergyMinimizer.minimize(ctx, maxIterations=min_iters)
    state = ctx.getState(getPositions=True, getEnergy=True)
    e_ab = state.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
    pos = state.getPositions()
    top = modeller.topology
    del ctx, integ, system

    e_a = _single_point(top, pos, keep_chain=target_chain)
    e_b = _single_point(top, pos, keep_chain=binder_chain)
    return round(e_ab - e_a - e_b, 2), round(e_ab, 1)


def _af2_mask(df):
    p, i, e = (("plddt", "iptm", "ipae") if "iptm" in df.columns
               else ("af2_plddt", "af2_iptm", "af2_ipae"))
    return (df[p] > 0.8) & (df[i] > 0.5) & (df[e] < 0.35)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv + designs/<name>.pdb")
    ap.add_argument("--sample", type=int, default=0, help="score only N random rows")
    ap.add_argument("--af2-pass-only", action="store_true",
                    help="score only designs that clear the AF2 gate")
    ap.add_argument("--min-iters", type=int, default=500,
                    help="complex minimization iterations (0 = score raw structure)")
    ap.add_argument("--platform", default=None, help="CUDA|OpenCL|CPU (default: best)")
    args = ap.parse_args()

    _PLATFORM["p"] = _platform(args.platform)
    print(f"OpenMM platform: {_PLATFORM['p'].getName() if _PLATFORM['p'] else 'default'}")

    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    df = pd.read_csv(csvf)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in ("openmm_dE", "openmm_e_complex"):
        if col not in df.columns:
            df[col] = pd.NA

    todo = df[df["openmm_dE"].isna()]
    if args.af2_pass_only:
        todo = todo[_af2_mask(todo)]
    if args.sample and len(todo) > args.sample:
        todo = todo.sample(args.sample, random_state=0)
    print(f"{args.design_dir}: {len(todo)} to score "
          f"({len(df) - len(todo)} already done / skipped)")

    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        pdb = os.path.join(args.design_dir, "designs", f"{row['name']}.pdb")
        if not os.path.exists(pdb):
            sys.exit(f"design PDB missing: {pdb}")
        try:
            dE, e_ab = interface_dE(pdb, args.min_iters)
        except (OpenMMException, ValueError) as exc:
            # fail loud per design but keep the batch going; record nothing so a
            # rerun retries this row rather than silently treating it as scored.
            print(f"  [{i}/{len(todo)}] {row['name']}: FAILED -- {exc}")
            continue
        df.loc[idx, ["openmm_dE", "openmm_e_complex"]] = [dE, e_ab]
        if i % 10 == 0 or i == len(todo):
            df.to_csv(csvf, index=False)              # checkpoint for resumability
            print(f"  [{i}/{len(todo)}] {row['name']}: dE={dE} kcal/mol")
    df.to_csv(csvf, index=False)
    print(f"wrote openmm_dE -> {csvf}")


if __name__ == "__main__":
    main()
