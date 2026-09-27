"""GPU-free adversarial reproductions for FoldCraft e485f324.

Run with Python + numpy + scipy + biopython. The upstream checkout defaults to
/private/tmp/foldcraft-review-colabdesign; no weights or model inference are used.
For target parsing only, execute upstream pure NumPy functions unchanged, replacing
JAX's tree traversal with equivalent traversal of the dict/list structures present.
"""
from __future__ import annotations

import ast
import os
import io
import runpy
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

REPO = Path(__file__).resolve().parents[3]
UPSTREAM = Path(os.environ.get('COLABDESIGN_REVIEW_SOURCE', '/private/tmp/foldcraft-review-colabdesign'))
sys.path.insert(0, str(REPO))
from biopython_utils import write_atomic
from cmap_utils import assemble_fold_conditioned_cmap


def tree_map(fn, *nodes):
    node = nodes[0]
    if isinstance(node, dict):
        return {k: tree_map(fn, *(n[k] for n in nodes)) for k in node}
    if isinstance(node, (tuple, list)):
        return type(node)(tree_map(fn, *(n[i] for n in nodes)) for i in range(len(node)))
    return fn(*nodes)


def load_without_imports(path, name, supplied, excluded):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        (isinstance(n, ast.Import) and any(x.name in excluded for x in n.names))
        or (isinstance(n, ast.ImportFrom) and n.module in excluded)
    )]
    module = types.ModuleType(name)
    module.__dict__.update(supplied)
    sys.modules[name] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    return module


def load_upstream_prep():
    rc = load_without_imports(
        UPSTREAM / 'colabdesign/af/alphafold/common/residue_constants.py',
        'review_rc', {'tree': types.SimpleNamespace(map_structure=tree_map)}, {'tree'})
    protein = load_without_imports(
        UPSTREAM / 'colabdesign/af/alphafold/common/protein.py',
        'review_protein', {'residue_constants': rc},
        {'colabdesign.af.alphafold.common'})
    shared = load_without_imports(
        UPSTREAM / 'colabdesign/shared/protein.py', 'review_shared',
        {'residue_constants': rc}, {'jax', 'jax.numpy', 'colabdesign.af.alphafold.common'})
    env = {'np': np, 'protein': protein, 'residue_constants': rc,
           'pdb_to_string': shared.pdb_to_string, '_np_get_cb': shared._np_get_cb,
           'jax': types.SimpleNamespace(tree_util=types.SimpleNamespace(tree_map=tree_map))}
    tree = ast.parse((UPSTREAM / 'colabdesign/af/prep.py').read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prep_pdb')
    exec(compile(ast.Module(body=[func], type_ignores=[]), '<upstream prep_pdb>', 'exec'), env)
    return env['prep_pdb']


def fixture(path, positions):
    s = Structure('fixture')
    m = Model(0)
    c = Chain('A')
    s.add(m)
    m.add(c)
    for i, pos in enumerate(positions):
        r = Residue((' ', pos, ' '), 'ALA', '')
        for j, (name, delta) in enumerate([('N', (0., 0., 0.)), ('CA', (1., 0., 0.)),
                                           ('C', (1., 1., 0.)), ('O', (1., 2., 0.))]):
            coord = np.asarray(delta) + (i * 4., 0., 0.)
            r.add(Atom(name, coord, 0., 1., ' ', f'{name:>4}', i*4+j+1, element=name[0]))
        c.add(r)
    writer = PDBIO()
    writer.set_structure(s)
    writer.save(str(path))


def main():
    framework = np.load(REPO / 'framework/vhh.npy')
    print('Current VHH versus bundled reference maps:')
    for name, target, hotspots, count in [
        ('vhh_pd_l1', 'pd-l1-1.pdb', '30-34,50-54,69-76', 180),
        ('vhh_pd_1', 'pd-1.pdb', '44-48,81-88,103-106', 170),
        ('vhh_ifnar', 'ifnar.pdb', '44-46,73-76,87-91', 120),
        ('vhh_egfr', 'egfr.pdb', '106-107,128-131,138-153', 0),
    ]:
        structure = PDBParser(QUIET=True).get_structure('', str(REPO / 'examples/targets' / target))
        target_len = len(list(structure[0]['A'].get_residues()))
        actual = assemble_fold_conditioned_cmap(
            framework, target_len, 127, hotspots, '26-35,55-59,102-116')
        reference = np.load(REPO / 'examples/cmaps' / f'{name}.npy')
        differences = np.count_nonzero(reference != actual)
        print(f'  {name}: {differences} different entries')
        assert differences == count

    print('Hotspot validation / range handling:')
    b = np.zeros((3, 3))
    for th, bh in [('6', '1'), ('0', '1'), ('2', '0'), ('4-2', '1'), ('2', '3-1')]:
        out = assemble_fold_conditioned_cmap(b, 5, 3, th, bh)
        print(f'  target={th!r} binder={bh!r}: nonzero entries {np.argwhere(out).tolist()}')
    assert assemble_fold_conditioned_cmap(b, 5, 3, '6', '1')[5, 5] == 1
    assert not assemble_fold_conditioned_cmap(b, 5, 3, '4-2', '1').any()

    prep_pos = runpy.run_path(str(UPSTREAM / 'colabdesign/shared/prep.py'))['prep_pos']
    residue = np.array([1, 2, 1, 2, 3])
    chain = np.array(['A', 'A', 'B', 'B', 'B'])
    full_mask = ','.join(f'B{x}' for x in range(1, 3))
    mutable = prep_pos(full_mask, residue=residue, chain=chain)['pos']
    fixed = np.delete(np.arange(len(residue)), mutable)
    print('Full MPNN redesign fixed residues:', list(zip(chain[fixed], residue[fixed])))
    assert 4 in fixed  # terminal B3 remains fixed
    try:
        prep_pos('', residue=residue, chain=chain)
    except IndexError as exc:
        print('Empty non-interface mask:', type(exc).__name__, str(exc))
    else:
        raise AssertionError('Expected empty-mask parser crash')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        final = str(tmp / 'results.csv')
        write_atomic(final, lambda p: Path(p).write_text('OLD COMPLETE'), finalize=True)
        write_atomic(final, lambda p: Path(p).write_text('NEW PARTIAL'))
        assert Path(final).read_text() == 'OLD COMPLETE'
        print('Rerun still exposes stale final:', Path(final).read_text())
        def crash(path):
            with open(path, 'w') as out:
                out.write('TRUNCATED')
                raise RuntimeError('simulated failed checkpoint')
        try:
            write_atomic(final, crash)
        except RuntimeError:
            pass
        assert Path(final + '.partial').read_text() == 'TRUNCATED'
        print('Last complete checkpoint replaced by:', Path(final + '.partial').read_text())

        prep_pdb = load_upstream_prep()
        gap = tmp / 'gap.pdb'
        fixture(gap, [1, 2, 4, 5])
        fixed = prep_pdb(str(gap), chain='A', ignore_missing=False)
        binder = prep_pdb(str(gap), chain='A', ignore_missing=True)
        a = len(fixed['residue_index'])
        b = len(binder['residue_index'])
        print(f'Gapped target: cmap target_len={a}, binder model target_len={b}')
        assert (a, b) == (5, 4)
        try:
            np.zeros((b+3, b+3)) * np.zeros((a+3, a+3))
        except ValueError as exc:
            print('Resulting conditioned-mask multiply:', type(exc).__name__, str(exc))

        offset = tmp / 'offset.pdb'
        fixture(offset, list(range(10, 30)))
        parsed = prep_pdb(str(offset), chain='A', ignore_missing=True)
        expected = prep_pos('15', **parsed['idx'])['pos'].item()
        actual = 15 - 1
        print('PDB hotspot A15: upstream array index', expected,
              '; FoldCraft cmap index', actual,
              '; cmap selects original PDB residue', parsed['idx']['residue'][actual])
        assert expected == 5 and parsed['idx']['residue'][actual] == 24


if __name__ == '__main__':
    main()

    # Additional current-branch cases. These execute pure functions, argument
    # parsing and no-inference driver dispatch, not neural model predictions.
    import argparse
    import json
    import pandas as pd
    import os

    def extract_fn(path, name, env):
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        return env[name]

    dispatch = extract_fn(UPSTREAM / 'colabdesign/af/design.py', '_get_model_nums', {'np': np})
    stub = types.SimpleNamespace(opt={'num_models': 1, 'sample_models': True},
                                 _model_names=['model_1_ptm', 'model_2_ptm'])
    actual = dispatch(stub, sample_models=False, models=stub._model_names)
    explicit = dispatch(stub, num_models=2, sample_models=False, models=stub._model_names)
    assert actual == [0] and explicit == [0, 1]
    print('Validation model dispatch:', actual, '; with explicit num_models=2:', explicit)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'missing_n.pdb'
        fixture(path, [1, 2, 3])
        structure = PDBParser(QUIET=True).get_structure('', path)
        structure[0]['A'][2].detach_child('N')
        assert 'CA' in structure[0]['A'][2]
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(str(path))
        prep_pdb = load_upstream_prep()
        padded = prep_pdb(str(path), chain='A', ignore_missing=False)
        compressed = prep_pdb(str(path), chain='A', ignore_missing=True)
        assert len(padded['residue_index']) == 3 and len(compressed['residue_index']) == 2
        print('Target with contiguous numbering but one missing N (CA present):',
              'cmap length 3; binder-model target length 2')

    # Input validation: a binder mask shape can accidentally broadcast.
    bad_shape = assemble_fold_conditioned_cmap(np.array([[0.7]]), 5, 3, '1', '1')
    assert np.all(bad_shape[-3:, -3:] == 0.7)
    bad_finite = assemble_fold_conditioned_cmap(np.full((3, 3), np.nan), 5, 3, '1', '1')
    assert np.isnan(bad_finite).any()
    print('Malformed cmap acceptance: (1,1) silently broadcasts to (3,3); NaNs pass through')

    cli_parse = extract_fn(REPO / 'FoldCraft.py', 'parse_args', {'argparse': argparse})
    old_argv = sys.argv
    sys.argv = ['FoldCraft.py', '--output_folder', 'unused', '--target_template', 'unused',
                '--target_hotspots', '1', '--vhh', '--sample', '--mpnn_samples', '0',
                '--num_designs', '-1', '--target_success', '-1', '--design_stages', '1,2',
                '--mpnn_sampling_temp', '-0.1', '--mpnn_backbone_noise', '-1']
    try:
        args = cli_parse()
    finally:
        sys.argv = old_argv
    assert (args.mpnn_samples, args.num_designs, args.target_success) == (0, -1, -1)
    assert args.design_stages == '1,2' and args.mpnn_sampling_temp < 0
    print('CLI accepts zero MPNN batch, negative quotas/noise/temperature, and only two design stages')

    notebooks = {name: json.loads((REPO / name).read_text())
                 for name in ['FoldCraft.ipynb', 'FoldCraft_VHH.ipynb']}
    for name, notebook in notebooks.items():
        source = ''.join(notebook['cells'][3]['source'])
        tree = ast.parse(source)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'set_range')
        env = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), f'{name} cell 3', 'exec'), env)
        assert env['set_range']('2-3') == [2]
        print(f'{name}: range 2-3 ->', env['set_range']('2-3'))

    standard = notebooks['FoldCraft.ipynb']
    source = ''.join(standard['cells'][5]['source'])
    tree = ast.parse(source)
    select = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'design_pos' for t in n.targets)][0]
    noninterface = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'new_design_pos' for t in n.targets)][0]
    env = {'binder_len': 5, 'interface_residues_list': [2, 3]}
    exec(compile(ast.Module(body=[select, noninterface], type_ignores=[]),
                 'FoldCraft.ipynb cell 5', 'exec'), env)
    assert env['new_design_pos'] == 'B1,B2,B3,B4'
    print('Notebook non-interface mutable set with protected [2,3]:', env['new_design_pos'])

    # Default experimental invocation reaches no design model at all.
    experimental = REPO / 'test/FoldCraft_binder.py'
    exp_env = {'argparse': argparse}
    exp_parse = extract_fn(experimental, 'parse_args', exp_env)
    old_argv = sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        sys.argv = [str(experimental), '--output_folder', tmp, '--target_template', 'unused',
                    '--target_hotspots', '1']
        try:
            exp_args = exp_parse()
        finally:
            sys.argv = old_argv
        calls = []
        exp_main = extract_fn(experimental, 'main', {
            'parse_args': lambda: exp_args, 'pd': pd,
            'os': types.SimpleNamespace(system=lambda cmd: calls.append(cmd)),
            'mk_afdesign_model': lambda **kw: (_ for _ in ()).throw(AssertionError('unexpected inference')),
        })
        exp_main()
        result = pd.read_csv(Path(tmp) / 'results.csv')
        assert result.empty
    print('Experimental default dispatch: 0 model calls, empty results.csv')

    from biopython_utils import calculate_clash_score, target_pdb_rmsd
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        structure = Structure('multimodel')
        for model_id in [0, 1]:
            model = Model(model_id)
            structure.add(model)
            for chain_id, coordinate in [('A', model_id * 100), ('B', (1 - model_id) * 100)]:
                chain = Chain(chain_id)
                model.add(chain)
                residue = Residue((' ', 1, ' '), 'ALA', '')
                chain.add(residue)
                residue.add(Atom('CA', np.array([float(coordinate), 0., 0.]), 0., 1.,
                                 ' ', ' CA ', 1, element='C'))
        writer = PDBIO()
        writer.set_structure(structure)
        path = tmp / 'two_models.pdb'
        writer.save(str(path))
        assert calculate_clash_score(str(path), only_ca=True) == 2
        print('Clash helper counts 2 clashes between different models; each model has none')
        longer, shorter = tmp / 'longer.pdb', tmp / 'shorter.pdb'
        fixture(longer, [1, 2, 3, 4])
        fixture(shorter, [1, 2, 3])
        assert target_pdb_rmsd(str(shorter), str(longer), 'A') == 0.
        print('RMSD helper reports 0.0 for unequal-length chains without coverage warning')

    print('All current-branch CPU correctness probes completed successfully.')
