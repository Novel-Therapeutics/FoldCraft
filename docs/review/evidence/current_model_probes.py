"""Read-only source/CPU probes for the September integrated FoldCraft review."""
from pathlib import Path
import ast
import importlib.util
import json
import tempfile
import types
import os
import glob

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

ROOT = Path(__file__).resolve().parents[3]


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    result = {}
    tree = ast.parse((ROOT / 'baseline/ab_get_best.py').read_text())
    folds = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                 and any(isinstance(t, ast.Name) and t.id == 'FOLDS' for t in n.targets)))
    parser = PDBParser(QUIET=True)
    result['ab_fold_hotspot_ranges'] = {}
    for fold, (file, ranges) in folds.items():
        residues = [r for r in parser.get_structure('t', ROOT/file)[0]['A'] if is_aa(r, standard=False)]
        positions = [v for part in ranges.split(',')
                     for v in (range(int(part.split('-')[0]), int(part.split('-')[1])+1) if '-' in part else [int(part)])]
        result['ab_fold_hotspot_ranges'][fold] = {
            'polymer_residues':len(residues), 'first_last_resseq':[residues[0].id[1],residues[-1].id[1]],
            'hotspots':ranges, 'out_of_bounds':[v for v in positions if not 1 <= v <= len(residues)],
        }

    # The repository loss uses sqrt(sum_{i,j} error^2 / N), not per-active-pair RMS.
    def loss(error):
        return np.sqrt(np.square(error).sum(-1).mean())
    error = np.zeros((4,4)); error[3,2] = error[2,3] = 1
    result['loss_unconstrained_padding'] = {'N4':float(loss(error)), 'N8_same_two_errors':float(loss(np.pad(error,((0,4),(0,4)))))}

    fn = function(ROOT/'baseline/score_esmfold.py', 'ca_rmsd', {'np':np})
    coords = np.array([[0.,0,0], [1,0,0], [0,1,0], [90,90,90], [-90,-90,-90]])
    result['esm_rmsd_truncated_match'] = {'reference_residues':5, 'scored_residues':3, 'reported_rmsd':fn(coords,coords[:3])}

    with tempfile.TemporaryDirectory(prefix='current_model_boltz_') as tmp:
        old = Path(tmp)/'previous_run'; old.mkdir()
        (old/'confidence_old.json').write_text(json.dumps({'confidence_score':.99,'iptm':.99,'complex_plddt':.99}))
        def fake_run(cmd, **kwargs):
            fresh = Path(tmp)/'fresh_run'; fresh.mkdir()
            (fresh/'confidence_new.json').write_text(json.dumps({'confidence_score':.2,'iptm':.2,'complex_plddt':.4}))
            return types.SimpleNamespace(returncode=0)
        fn = function(ROOT/'baseline/score_boltz2.py','run_boltz2',
                      {'os':os,'json':json,'glob':glob,'subprocess':types.SimpleNamespace(run=fake_run)})
        result['boltz_stale_confidence_selection'] = {
            'fresh_iptm':.2, 'stale_iptm':.99, 'returned_iptm':fn('AAA','AAA',tmp,use_msa=False,flash_attn=False)[0]}

    spec = importlib.util.spec_from_file_location('ipsae',ROOT/'baseline/ipsae.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result['ipsae_rectangular_input'] = {'shape':[4,1], 'returned':module.ipsae(np.ones((4,1)),2,2)}

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
