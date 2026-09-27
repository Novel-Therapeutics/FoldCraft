"""Verify frozen-pool scorer identities, paired streams and raw Boltz selections."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from baseline.result_io import write_json
from run_state import sha256
from scripts.score_boltz_paired import paired_seed
from scripts.analyze_pilot import geometry
from baseline.add_rmsd import template_ca_atoms, rmsd_to_template


def audit(root):
    root = Path(root)
    pool = root / 'pool'
    frame = pd.read_csv(pool / 'results.csv')
    manifest = json.loads((pool / 'pool_manifest.json').read_text())
    assert len(frame) == 96 and set(frame.name) == set(manifest['sources'])
    protocol = json.loads((root / 'protocol.json').read_text())
    assert manifest['protocol_sha256'] == sha256(root / 'protocol.json')
    jobs = {job['name']: job for job in protocol['jobs']}
    for row in frame.to_dict('records'):
        source = manifest['sources'][row['name']]
        folder = root / source['source_run']
        job = jobs[source['source_run']]
        assert sha256(folder / 'run.json') == source['run_sha256']
        for key in ('case', 'seed', 'arm', 'split', 'target', 'scaffold', 'family'):
            assert row[key] == job[key]
        assert int(row['draw']) == int(source['candidate'].rsplit('_', 1)[1])
        assert row['trajectory'] == source['candidate'].rsplit('_', 1)[0]
        validation = json.loads((folder / 'designs' / (source['candidate'] + '.validation.json')).read_text())
        primary = validation['models'][validation['primary_model']]
        assert bool(row['single_model_pass']) == primary['passed']
        assert bool(row['two_model_pass']) == validation['all_models_pass']
        assert np.isclose(row['iptm'], primary['metrics']['i_ptm'], rtol=0, atol=1e-12)
        pdb = pool / 'designs' / (row['name'] + '.pdb')
        state = json.loads((folder / 'run.json').read_text())
        metrics = geometry(pdb, state['mapped_hotspots']['target'])
        metrics['rmsd'] = rmsd_to_template(str(pdb), template_ca_atoms(str(folder / 'template.pdb')))
        for key in ('rmsd', 'epitope_coverage', 'interchain_clashes'):
            assert np.isclose(row[key], metrics[key], rtol=0, atol=1e-9)
    receipts = {name: json.loads((pool / (name + '.scores.json')).read_text())
                for name in ('score_esmfold', 'score_boltz_paired')}
    for scorer, receipt in receipts.items():
        assert set(receipt['records']) == set(frame.name)
        for row in frame.to_dict('records'):
            source = manifest['sources'][row['name']]
            structure_hash = sha256(pool / 'designs' / (row['name'] + '.pdb'))
            assert source['sequence'] == row['sequence']
            assert source['pdb_sha256'] == structure_hash
            payload = dict(protocol=receipt['protocol'], name=row['name'],
                           sequence=row['sequence'], structure=structure_hash)
            signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            record = receipt['records'][row['name']]
            assert record['signature'] == signature and record['status'] == 'complete'
            for column, value in record['values'].items():
                assert value is not None and np.isfinite(value)
                assert np.isclose(value, row[column], rtol=0, atol=1e-12)
    selections = 0
    for row in frame.to_dict('records'):
        record = receipts['score_boltz_paired']['records'][row['name']]
        assert record['diagnostics']['seeds'] == [paired_seed(row, r) for r in range(2)]
        values = []
        for repeat in range(2):
            paths = list((root / 'boltz-work' / row['name'] / str(repeat)).glob('invocation-*/selection.json'))
            assert len(paths) == 1
            path = paths[0]
            selected = json.loads(path.read_text())
            assert selected['seed'] == paired_seed(row, repeat)
            assert selected['target_sequence'] + '/' + selected['binder_sequence'] == row['sequence']
            assert selected['diffusion_samples'] == 1 and selected['use_msa'] is False
            for kind in ('confidence', 'structure'):
                assert sha256(path.parent / selected[kind + '_file']) == selected[kind + '_sha256']
            confidence = json.loads((path.parent / selected['confidence_file']).read_text())
            values.append(confidence['iptm'])
            assert np.isclose(values[-1], row[f'boltz2_iptm_r{repeat}'], rtol=0, atol=1e-12)
            selections += 1
        assert np.isclose(np.mean(values), row['boltz2_iptm'], rtol=0, atol=1e-12)
    result = dict(status='passed', candidates=len(frame), esmfold_records=96,
                  boltz_repeats=selections, paired_seed_records=96,
                  notes='Pairing metadata, ranking inputs, CSV scores, scorer signatures, frozen sequences/structures and raw Boltz selections agree.')
    write_json(root / 'evaluator_audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
