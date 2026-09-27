# =============================================================================
# EXPERIMENTAL — not part of the validated FoldCraft pipeline.
#
# FoldCraft_binder.py implements length-variable de novo binder design
# (exploratory linear / miniprotein binder runs, e.g. Nipah, KEAP1). It is
# provided as-is, is NOT covered by the paper's benchmarks, and its CLI /
# behaviour may change or break. For the published results use FoldCraft.py
# (fold-conditioned and VHH design).
#
# Paths to local modules are anchored to this file; input paths are caller-relative.
# =============================================================================

import argparse
import re
import math
import os
import sys
import json
import secrets
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from input_validation import residue_range, read_chain
from sequence_design import redesign

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run fold-conditioned binder design")

    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the results')
    parser.add_argument('--sample', action='store_true', help='Whether to generate designs until the target number of successful designs is reached')
    parser.add_argument('--target_success', type=int, default=100, help='Target number of successful designs to generate (used only if --sample is enabled)')
    parser.add_argument('--num_designs', type=int, default=1, help='Number of design trajectories to generate (ignored if --sample is enabled)')
    parser.add_argument('--vhh', action='store_true', help='Whether to use VHH framework to construct target cmap (all binder information would be ignored in that case)')

    parser.add_argument('--binder_template', type=str, help='Path to the binder template PDB file (required)')
    parser.add_argument('--target_template', type=str, required=True, help='Path to the target template PDB file (required)')
    parser.add_argument('--target_hotspots', type=str, required=True, help='''Residue ranges for target hotspots, e.g., "14-30,80-81,90-102" (required)''')
    parser.add_argument('--binder_hotspots', type=str, default='', help='Residue ranges for binder, e.g. "14-30,80-81,90-102"')
    parser.add_argument('--binder_mask', type=str, default='', help='Residue ranges in the binder to mask (ignored during loss computation), e.g. "14-30"')
    parser.add_argument('--binder_chain', type=str, default='A', help='Binder template chain (default = A)')
    parser.add_argument('--target_chain', type=str, default='A', help='Target template chain (default = A)')

    parser.add_argument('--design_stages', type=str, default='100,100,20', help="Number of each design stages in 3stage_design (default: 100,100,20)")

    parser.add_argument('--mpnn_weight', type=str, choices=['soluble', 'original'], default='soluble', help="PoteinMPNN weights to use ('soluble', 'original')")
    parser.add_argument('--redesign_method', type=str, choices=['full', 'non-interface'], default='non-interface', help="ProteinMPNN redesign strategy: 'full' or 'non-interface' (default: 'non-interface')")
    parser.add_argument('--mpnn_samples', type=int, default=5, help="Number of sequences to sample with ProteinMPNN (default: 5)")
    parser.add_argument('--mpnn_backbone_noise', type=float, default=0.0, help="Backbone noise during sampling (default: 0.0)")
    parser.add_argument('--mpnn_sampling_temp', type=float, default=0.1, help="Sampling temperature for amino acids 0.0-1.0 (default: 0.1)")
    parser.add_argument('--mpnn_save', action='store_true', help='Whether to save MPNN sampled sequences')
    parser.add_argument('--start_with', type=int, default=0, help='Start with')
    parser.add_argument('--binder_len', type=str, default='100', help='Binder len in str')

    parser.add_argument('--max_trajectories', type=int, default=1000)
    parser.add_argument('--data_dir', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--timeout_minutes', type=float, default=360.)
    parser.add_argument('--preflight_only', action='store_true')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.sample:
        parser.error('This experimental entry point requires --sample; fixed-count mode is not implemented')
    if args.vhh or args.binder_template or args.binder_hotspots or args.binder_mask:
        parser.error('This de novo entry point does not implement VHH/template/binder conditioning; use FoldCraft.py')
    if not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0:
        parser.error('--timeout_minutes must be finite and positive')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error('--seed must be a 32-bit nonnegative integer')
    if args.start_with != 0:
        parser.error('--start_with resume is unsupported; use a fresh output directory')
    if any(n < 1 for n in (args.target_success, args.mpnn_samples, args.max_trajectories)):
        parser.error('Success, sample and trajectory budgets must be positive')
    try:
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', args.binder_len):
            raise ValueError('--binder_len must be a length or inclusive start-end range')
        residue_range(args.binder_len)
        if not math.isfinite(args.mpnn_sampling_temp) or args.mpnn_sampling_temp <= 0 or not math.isfinite(args.mpnn_backbone_noise) or args.mpnn_backbone_noise < 0:
            raise ValueError('MPNN temperature must be positive; noise must be nonnegative')
        stages = [int(n) for n in args.design_stages.split(',')]
        if len(stages) != 3 or min(stages) < 1:
            raise ValueError('Expected three positive design stages')
    except ValueError as exc:
        parser.error(str(exc))
    return args

def main(*, supervised=True):
    args = parse_args()
    if os.path.exists(args.output_folder):
        raise SystemExit('Output exists; choose a fresh directory')
    target = read_chain(args.target_template, args.target_chain)
    mapped = target.selection(args.target_hotspots)
    if args.preflight_only:
        print('Experimental input preflight passed; GPU/PyRosetta dependencies not checked.')
        return
    if supervised and not args._worker:
        from run_watchdog import run_owned
        outcome = run_owned([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--_worker'],
                            args.timeout_minutes * 60)
        path = Path(args.output_folder)/'experimental_run.json'
        if path.exists():
            from baseline.result_io import write_json
            state = json.loads(path.read_text())
            state['supervision'] = vars(outcome)
            if outcome.reason != 'exited':
                state['status'] = outcome.reason
            elif outcome.returncode and state['status'] == 'running':
                state['status'] = 'failed'
            elif outcome.returncode == 0:
                try:
                    if state['status'] != 'complete' or verify_outputs(args.output_folder, args.target_success) != state.get('artifacts'):
                        raise ValueError('Worker did not produce a verified completed run')
                except (OSError, ValueError, KeyError) as exc:
                    state.update(status='failed', error=str(exc))
                    outcome.returncode = 1
            write_json(path, state)
        elif outcome.returncode == 0:
            raise RuntimeError('Worker exited without an experimental run manifest')
        if outcome.returncode:
            raise SystemExit(outcome.returncode)
        return
    args.target_hotspots = mapped
    if args.seed is None:
        args.seed = secrets.randbits(32)
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())
    from baseline.result_io import write_json
    from run_state import sha256
    # Record a run before dependency loading so startup failures are inspectable.
    Path(args.output_folder).mkdir(parents=True, exist_ok=False)
    manifest = Path(args.output_folder)/'experimental_run.json'
    state = dict(schema=1, status='running', config=vars(args).copy(), target=target.manifest(),
                 code_sha256=sha256(__file__), seed_protocol='sha256-stage-v1')
    write_json(manifest, state)
    try:
        execute(args, target)
        state['artifacts'] = verify_outputs(args.output_folder, args.target_success)
        state['status'] = 'complete'
    except BaseException as exc:
        state['status'] = 'exhausted' if isinstance(exc, BudgetExhausted) else 'failed'
        state['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(manifest, state)


class BudgetExhausted(RuntimeError):
    pass


def physical_passes(clashes, scores):
    keys = ('interface_sc', 'interface_delta_unsat_hbonds', 'surface_hydrophobicity', 'interface_dG')
    if not math.isfinite(clashes) or any(not math.isfinite(float(scores[k])) for k in keys):
        raise ValueError('Nonfinite physical evaluation; cannot accept a candidate')
    return (clashes == 0 and scores['interface_sc'] > .55 and
            scores['interface_delta_unsat_hbonds'] < 3 and scores['surface_hydrophobicity'] <= .35)


def verify_outputs(folder, expected):
    import csv
    from run_state import sha256
    root = Path(folder)
    with (root/'results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    names = [r['name'] for r in rows]
    if len(names) != expected or len(set(names)) != len(names):
        raise ValueError('Experimental completion requires the exact accepted-candidate count')
    artifacts = {}
    for row in rows:
        name = row['name']
        if not name or Path(name).name != name:
            raise ValueError('Invalid candidate identity')
        for key, value in row.items():
            if key not in ('name', 'sequence') and not math.isfinite(float(value)):
                raise ValueError('Nonfinite result cannot certify completion')
        for relative in (f'designs/{name}.pdb', f'designs/{name}.pickle', f'relaxed/{name}.pdb'):
            path = root/relative
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError('Missing experimental result artifact: '+relative)
            artifacts[relative] = sha256(path)
    for name in ('results.csv', 'results_pyrosetta.csv', 'target.pdb'):
        artifacts[name] = sha256(root/name)
    return artifacts


def execute(args, target):
    import jax
    import jax.numpy as jnp


    import pandas as pd
    import pickle

    from colabdesign.af.alphafold.common import residue_constants

    from colabdesign import mk_afdesign_model, clear_mem
    from run_state import stage_seed
    from baseline.result_io import atomic_write
    import random
    random.seed(args.seed)

    from colabdesign.mpnn import mk_mpnn_model
    from biopython_utils import hotspot_residues, iter_until_target, calculate_clash_score

    # BindCraft provides the PyRosetta relax + interface-scoring helpers reused
    # below. It is not pip-installable, so locate the checkout (created by
    # test/install_foldcraft_binder.sh or pointed to via $BINDCRAFT_PATH), put it on
    # sys.path, and import only what we use -- avoiding the previous `import *`,
    # which both failed when BindCraft was absent and shadowed FoldCraft's own
    # biopython_utils helpers. This script and bindcraft_deps.py both live in test/;
    # the repo root is added so `biopython_utils` (above) also resolves.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # test/ -> bindcraft_deps
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> biopython_utils
    from bindcraft_deps import ensure_bindcraft_importable, dalphaball_path
    ensure_bindcraft_importable()
    import pyrosetta as pr
    from BindCraft.functions.pyrosetta_utils import pr_relax, score_interface

    rosetta_seed = stage_seed(args.seed, 'rosetta') % (2**31 - 1)
    pr.init(f'-constant_seed -jran {rosetta_seed} -ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball "{dalphaball_path()}" -corrections::beta_nov16 true -relax:default_repeats 1')


    args.target_template = str(Path(args.output_folder) / 'target.pdb')
    Path(args.target_template).write_text(target.pdb)

    #Prepare fold conditioned binder

    pdb_target_path = args.target_template #template for target
    chain_id = args.target_chain #Select chain for target protein.
    lengths = residue_range(args.binder_len)
    binder_len = lengths[0]
    binder_lengths = [lengths[0], lengths[-1]] if len(lengths) > 1 else False
    target_hotspots = args.target_hotspots #Choose hotspots on target protein


    folder_name = args.output_folder #name for output folder

    redesign_method = args.redesign_method
    mpnn_version = args.mpnn_weight
    sample = args.sample
    success_target = args.target_success

    mpnn_samples = args.mpnn_samples

    design_stages = [int(x) for x in args.design_stages.split(',')]
    mpnn_backbone_noise = args.mpnn_backbone_noise
    mpnn_sampling_temp = args.mpnn_sampling_temp

    mpnn_save = args.mpnn_save
    start_with = args.start_with

    model_name = 'v_48_010'

    # Start to design

    # Define the fold-conditioned loss

    names = []
    sequences = []
    plddts = []
    ipaes = []
    iptms = []
    rg = []
    shape_c = []
    uns_hb = []
    hydrophob_per = []
    int_dG = []
    clashes = []
    rmsds = []

    for subdirectory in ('traj','mpnn','designs','relaxed'):
        (Path(folder_name) / subdirectory).mkdir()

    # Generate N number of trajectories

    if sample:
        passed = 0
        #success_target = success_target
        i=start_with
        # `< target` (not `<=`) and the iter_until_target cap on the inner batch
        # loop below make the accepted count exactly --target_success (was `<=`
        # plus an uncapped batch loop -> overshoot by up to mpnn_samples).
        while passed < success_target:
            if i >= args.max_trajectories:
                raise BudgetExhausted('Trajectory budget exhausted; partial results preserved')
            clear_mem()
            if binder_lengths != False:
                binder_len = random.randint(binder_lengths[0], binder_lengths[1])
            i+=1
            name = f'traj_{i}' #@param {type:"string"}
            rm_aa = 'C' #@param {type:"string"}
            def rg_loss(inputs, outputs):
                positions = outputs["structure_module"]["final_atom_positions"]
                ca = positions[:,residue_constants.atom_order["CA"]]
                center = ca.mean(0)
                rg = jnp.sqrt(jnp.square(ca - center).sum(-1).mean() + 1e-8)
                rg_th = 2.38 * ca.shape[0] ** 0.365
                rg = jax.nn.elu(rg - rg_th)
                return {"rg":rg}

            af_model = mk_afdesign_model(data_dir=args.data_dir, model_names=["model_1_ptm", "model_2_ptm"], protocol="binder", loss_callback=rg_loss,
                                                 use_templates=True,)

            af_model.prep_inputs(pdb_filename=pdb_target_path,
                                         chain=chain_id, binder_len = binder_len,
                                         hotspot=target_hotspots,
                                         rm_aa=rm_aa, #fix_pos=fixed_positions,
                                         )

            af_model.opt["weights"]["rg"] = .4
            af_model.opt["weights"].update({"rg":.0, "rmsd":0.0, "fape":0.0, "plddt":0.0, 'pae':0.2,
                                                    "con":1.0, "i_con":1.0, "i_pae":0.2})

            af_model.restart(seed=stage_seed(args.seed, 'design', i), reset_opt=False)
            af_model.design_3stage(design_stages[0],design_stages[1],design_stages[2])
            #af_model.design_semigreedy(20, tries=20, models=["model_1_ptm"], num_models=1, num_models=2)
            af_model.save_pdb(f"{folder_name}/traj/{name}.pdb", get_best=False)
            with open(f'{folder_name}/traj/{name}.pickle', 'wb') as handle:
                pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            if af_model.aux['log']['i_ptm']>0. and af_model.aux['log']['plddt']>.65:
                clear_mem()
                #Running ProteinMPNN on designed trajectory
                interface = list(hotspot_residues(f"{folder_name}/traj/{name}.pdb", 'B').keys())
                mpnn_model = mk_mpnn_model(model_name, backbone_noise=mpnn_backbone_noise, weights=mpnn_version)
                mpnn_model.set_seed(stage_seed(args.seed, 'mpnn', i))
                samples = redesign(mpnn_model, f"{folder_name}/traj/{name}.pdb", binder_len,
                                   interface, redesign_method, mpnn_sampling_temp, mpnn_samples)
                if mpnn_save:
                    with open(f'{folder_name}/mpnn/mpnn_{name}.pickle', 'wb') as handle:
                        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

                #Predict Samples with AF2_ptm
                print('Predicting sequences with AF2_ptm...')
                # cap the batch at the remaining global budget (see iter_until_target)
                for num, seq in iter_until_target(samples['seq'], lambda: passed, success_target):
                    af_model = mk_afdesign_model(data_dir=args.data_dir, model_names=["model_1_ptm", "model_2_ptm"], protocol="binder", loss_callback=rg_loss,
                                                           use_templates=True,)

                    af_model.prep_inputs(pdb_filename=pdb_target_path,
                                                   chain=chain_id, binder_len = binder_len,
                                                   #hotspot='',
                                                   rm_aa=rm_aa, #fix_pos=fixed_positions,
                                                   )
                    af_model.set_seq(seq[-binder_len:])
                    af_model.predict(num_recycles=3, verbose=False, models=["model_1_ptm"], num_models=1, seed=stage_seed(args.seed, "validation", i, num))

                    print(f"predict: {name}_{num} plddt: {af_model.aux['log']['plddt']:.3f}, i_pae: {(af_model.aux['log']['i_pae']):.3f}, i_ptm: {af_model.aux['log']['i_ptm']:.3f}, rg: {af_model.aux['log']['rg']:.3f}")

                    if af_model.aux['log']['i_pae']<0.35 and af_model.aux['log']['plddt']>.8 and af_model.aux['log']['i_ptm']>0.5:
                        af_model.save_pdb(f"{folder_name}/designs/{name}_{num}.pdb", get_best=False)
                        with open(f'{folder_name}/designs/{name}_{num}.pickle', 'wb') as handle:
                            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

                        mpnn_design_relaxed = f'{folder_name}/relaxed/{name}_{num}.pdb'
                        mpnn_design_pdb = f'{folder_name}/designs/{name}_{num}.pdb'
                        pr_relax(mpnn_design_pdb, mpnn_design_relaxed)
                        binder_chain = 'B'
                        num_clashes_mpnn_relaxed = calculate_clash_score(mpnn_design_relaxed)
                        mpnn_interface_scores, mpnn_interface_AA, mpnn_interface_residues = score_interface(mpnn_design_relaxed, binder_chain)

                        if physical_passes(num_clashes_mpnn_relaxed, mpnn_interface_scores):
                            clashes.append(num_clashes_mpnn_relaxed)
                            shape_c.append(mpnn_interface_scores['interface_sc'])
                            uns_hb.append(mpnn_interface_scores['interface_delta_unsat_hbonds'])
                            hydrophob_per.append(mpnn_interface_scores['surface_hydrophobicity'])
                            int_dG.append(mpnn_interface_scores['interface_dG'])

                            names.append(f'{name}_{num}')
                            sequences.append(seq[-binder_len:])
                            plddts.append(af_model.aux['log']['plddt'])
                            ipaes.append(af_model.aux['log']['i_pae'])
                            iptms.append(af_model.aux['log']['i_ptm'])
                            rg.append(af_model.aux['log']['rg'])

                            af_model = mk_afdesign_model(data_dir=args.data_dir, model_names=["model_1_ptm", "model_2_ptm"], protocol="fixbb", use_templates=False)
                            af_model.prep_inputs(pdb_filename=f'{folder_name}/designs/{name}_{num}.pdb', chain='B',
                                                 length=binder_len, ignore_missing=False)
                            af_model.set_seq(seq[-binder_len:])
                            af_model.predict(num_recycles=3, verbose=False, models=["model_1_ptm"], num_models=1, seed=stage_seed(args.seed, "monomer", i, num))

                            rmsds.append(af_model.aux['log']['rmsd'])
                            df = pd.DataFrame({'name':names,
                               'sequence':sequences,
                               'plddt':plddts,
                               'ipae':ipaes,
                               'iptm':iptms,
                               'rg_loss':rg,
                               'shape_c': shape_c,
                                'uns_hb' : uns_hb,
                                'hydrophob_per' : hydrophob_per,
                                'int_dG' : int_dG,
                                'clashes' : clashes,
                                'rmsds': rmsds,})

                            atomic_write(f"{folder_name}/results_pyrosetta.csv", lambda p: df.to_csv(p, index=False))
                            passed+=1
                        else:

                            Path(mpnn_design_relaxed).unlink()



    df = pd.DataFrame({'name':names,
                               'sequence':sequences,
                               'plddt':plddts,
                               'ipae':ipaes,
                               'iptm':iptms,
                               'rg_loss':rg,
                               'shape_c': shape_c,
                                'uns_hb' : uns_hb,
                                'hydrophob_per' : hydrophob_per,
                                'int_dG' : int_dG,
                                'clashes' : clashes,
                                'rmsds': rmsds,})

    atomic_write(f"{folder_name}/results.csv", lambda p: df.to_csv(p, index=False))

if __name__ == '__main__':
    main()
