"""Read-only repository review probes; all synthetic artifacts live in tmpdirs.

Run with /private/tmp/foldcraft-integration-venv/bin/python. No GPU, network,
model install, or repository writes. These probes assert current bugs exist.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]


def module(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'baseline' / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sch = module('scheduler')
af2 = module('score_af2')
esm = module('score_esmfold')
boltz = module('score_boltz2')


def fold(name, root, mem=8, total=1):
    return dict(fold=name, template=str(root / 'template.pdb'),
                target=str(root / 'target.pdb'), target_hotspots='1',
                binder_hotspots='1', total_traj=total, mem_gb=mem)


def chunk_files(root, f, n=1):
    c = sch.plan_chunks([f], n)[0]
    d = Path(c.out_dir(str(root)))
    (d / 'designs').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'name': ['traj_1_0'], 'plddt': [.9]}).to_csv(d / 'results.csv', index=False)
    (d / 'designs' / 'traj_1_0.pdb').write_text('PDB')
    return c, d


def dryrun_archived_tables():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sch.main(['baseline/repro_config.tsv', '--repo', str(REPO), '--dry-run'])
    first = out.getvalue().splitlines()[0]
    assert 'todo=4 (done=20)' in first
    configs = sch.load_config(str(REPO / 'baseline/repro_config.tsv'))
    missing = sum(not (REPO / 'baseline/repro' / f['fold'] / 'designs').exists()
                  for f in configs if f['fold'] != 'tim')
    assert missing == 5
    return {'dry_run': first, 'historical_folds_without_designs': missing}


def config_ignored_on_resume(root):
    root.mkdir()
    old = fold('x', root)
    c, _ = chunk_files(root, old)
    new = dict(old, target_hotspots='55-60', total_traj=10)
    todo = sch.filter_todo(sch.plan_chunks([new], 10), str(root))
    assert not todo
    return {'changed_hotspots': new['target_hotspots'], 'changed_trajectory_count': 10,
            'scheduled_chunks': len(todo), 'old_chunk_trajectories': c.chunk_traj}


def interrupted_merge_is_done(root):
    root.mkdir()
    f = fold('x', root)
    (root / 'template.pdb').write_text('TEMPLATE')
    c, d = chunk_files(root, f)
    def truncated_csv(self, path, **kwargs):
        Path(path).write_text('name,plddt\n')
        raise OSError('simulated interruption / full disk')
    try:
        with patch.object(pd.DataFrame, 'to_csv', truncated_csv):
            sch.merge_chunks('x', [str(d)], str(root / 'x'), f['template'])
    except OSError:
        pass
    done = sch.fold_done('x', str(root))
    todo = sch.filter_todo([c], str(root))
    assert done and not todo
    return {'interrupted_merge_considered_done': done,
            'rows_in_partial_final_csv': len(pd.read_csv(root / 'x/results.csv'))}


def solo_not_reserved(root):
    root.mkdir()
    (root / 'template.pdb').write_text('TEMPLATE')
    fs = [fold('normal', root, mem=14), fold('small_retry', root, mem=6)]
    live = []
    events = []
    attempts = {}
    # An external GPU tenant initially leaves 10 GB free, then releases memory
    # after the retry starts. This is precisely the live-free-memory case the
    # scheduler documents supporting. No injected/reordered queue is needed.
    readings = iter([(24, 24), (10, 24), (10, 24), (10, 24)])
    def mem(*args):
        return next(readings, (24, 24))
    class Proc:
        def __init__(self, c, rc):
            self.pid = len(events)
            self.c = c
            self.rc = rc
        def poll(self):
            live.remove(self.c.tag)
            return self.rc
    def launch(c, repro, gpu, repo, python):
        events.append({'tag': c.tag, 'solo': c.solo, 'concurrent': list(live)})
        live.append(c.tag)
        attempts[c.tag] = attempts.get(c.tag, 0) + 1
        rc = 1 if c.fold == 'small_retry' and attempts[c.tag] == 1 else 0
        d = Path(c.out_dir(repro))
        d.mkdir(exist_ok=True)
        if rc == 0:
            (d / 'results.csv').write_text('name\n')
        return Proc(c, rc), io.StringIO()
    with patch.object(sch, '_nvidia_smi', return_value='fake'), \
         patch.object(sch, 'gpu_mem_gb', mem), \
         patch.object(sch, 'launch', launch), \
         patch.object(sch, 'merge_chunks', return_value=0):
        sch.run(fs, str(root), str(REPO), ['0'], 1, 2, sys.executable,
                poll_s=0, log=lambda _: None)
    assert events[1]['solo'] is True
    assert events[2]['concurrent'] == ['small_retry__c0']
    return {'launches': events}


def score_dir(root):
    (root / 'designs').mkdir(parents=True)
    (root / 'designs' / 'candidate.pdb').write_text('PDB')
    pd.DataFrame({'name': ['candidate'], 'sequence': ['AAA']}).to_csv(root / 'results.csv', index=False)
    target = root / 'target.pdb'
    target.write_text('PDB')
    return target


def invoke(m, args):
    with patch.object(sys, 'argv', [m.__name__] + args), contextlib.redirect_stdout(io.StringIO()):
        m.main()


def scorer_lost_update(root):
    target = score_dir(root)
    def interleave(target, seq, **kwargs):
        # AF2 loaded its snapshot; ESMFold now completes and writes its columns.
        with patch.object(esm, 'binder_seq_and_ca', return_value=('AAA', np.zeros((3, 3)))), \
             patch.object(esm, 'esmfold_predict', return_value=(90, np.zeros((3, 3)))), \
             patch.object(esm, 'ca_rmsd', return_value=0.0):
            invoke(esm, [str(root)])
        assert 'esmfold_plddt' in pd.read_csv(root / 'results.csv').columns
        return .9, .7, .2
    with patch.object(af2, 'af2_score', interleave):
        invoke(af2, [str(root), '--target', str(target)])
    columns = list(pd.read_csv(root / 'results.csv').columns)
    assert 'af2_plddt' in columns and 'esmfold_plddt' not in columns
    return {'columns_after_both_successful_scorers': columns}


def scorer_config_ignored(root):
    target = score_dir(root)
    calls = []
    def predict(t, seq, **kwargs):
        calls.append((Path(t).name, kwargs['hotspot']))
        return .9, .7, .2
    target2 = root / 'other_target.pdb'
    target2.write_text('OTHER PDB')
    with patch.object(af2, 'af2_score', predict):
        invoke(af2, [str(root), '--target', str(target), '--hotspot', '1'])
        invoke(af2, [str(root), '--target', str(target2), '--hotspot', '50'])
    assert len(calls) == 1
    return {'requested_scoring_configs': 2, 'actual_prediction_calls': calls}


def boltz_stale_result(root):
    root.mkdir()
    (root / 'confidence_previous.json').write_text(json.dumps({'confidence_score': .99, 'iptm': .99}))
    def fake_predict(*args, **kwargs):
        (root / 'confidence_current.json').write_text(json.dumps({'confidence_score': .1, 'iptm': .1}))
        return SimpleNamespace(returncode=0, stderr='')
    with patch.object(boltz.subprocess, 'run', fake_predict):
        score = boltz.run_boltz2('AAA', 'CCC', str(root), use_msa=False)
    assert score[0] == .99
    return {'current_result_iptm': .1, 'reported_iptm': score[0]}


def campaign_false_success(root):
    root.mkdir()
    fakepy = root / 'python'
    fakepy.write_text('#!/bin/sh\nexit 23\n')
    fakepy.chmod(0o755)
    result = subprocess.run(['bash', str(REPO / 'baseline/run_campaign.sh')],
                            env=dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH']),
                            text=True, capture_output=True)
    assert result.returncode == 0
    assert result.stdout.count('FOLD_FAIL') == 6
    return {'exit_code': result.returncode, 'failed_folds': result.stdout.count('FOLD_FAIL'),
            'last_line': result.stdout.splitlines()[-1]}


if __name__ == '__main__':
    results = {'archive_resume': dryrun_archived_tables()}
    with tempfile.TemporaryDirectory(prefix='foldcraft-current-ops-') as tmp:
        root = Path(tmp)
        for probe in (config_ignored_on_resume, interrupted_merge_is_done,
                      solo_not_reserved, scorer_lost_update, scorer_config_ignored,
                      boltz_stale_result, campaign_false_success):
            results[probe.__name__] = probe(root / probe.__name__)
    print(json.dumps(results, indent=2))
