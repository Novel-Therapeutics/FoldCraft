"""Adversarial regression tests; no weights, CUDA, or remote services."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from input_validation import read_chain, residue_range, validate_cmap
from sequence_design import mutable_positions, redesign
from baseline.result_io import atomic_write, publish_columns
from baseline.structure_checks import minimize_if_requested
from baseline.score_esmfold import binder_seq_and_ca, ca_rmsd
from baseline.add_rmsd import template_ca_atoms, rmsd_to_template
from baseline.add_ipsae import single_model_pae
from baseline.ipsae import ipsae
from run_state import completed_run, finish_run

ROOT = Path(__file__).resolve().parents[1]


def pdb_text(ids=(1,2,3), chain='A', missing=None):
    lines = []
    for i, resid in enumerate(ids):
        for atom, delta in [('N',0), ('CA',1), ('C',2), ('O',3)]:
            if missing == (resid, atom):
                continue
            lines.append(f'ATOM  {len(lines)+1:5d} {atom:^4s} ALA {chain}{resid:4d}    {i*4+delta:8.3f}{i%2:8.3f}{i%3:8.3f}  1.00 20.00           {atom[0]:>2s}')
    return '\n'.join(lines) + '\n'


@pytest.mark.parametrize('bad', ['', '0', '3-1', '1,1', '1-3,3', '-1', '1,,2', 'a', '1-10000'])
def test_bad_ranges(bad):
    with pytest.raises(ValueError):
        residue_range(bad)


def test_offset_mapping_selects_same_residue(tmp_path):
    p = tmp_path / 'x.pdb'; p.write_text(pdb_text((41,42,43)))
    prepared = read_chain(p, 'A')
    assert prepared.selection('42-43') == '2,3'
    assert prepared.manifest()['residues'][1] == {'original':42,'model_position':2}
    normalized = tmp_path / 'normalized.pdb'; normalized.write_text(prepared.pdb)
    assert read_chain(normalized, 'A').residue_ids == (1,2,3)
    with pytest.raises(ValueError, match='absent'):
        prepared.selection('1')


@pytest.mark.parametrize('text,reason', [
    (pdb_text((1,3,4)), 'gaps'),
    (pdb_text(missing=(2,'N')), 'backbone'),
    (pdb_text().replace('A   2 ', 'A   2A'), 'insertion'),
    ('MODEL        1\n'+pdb_text()+'ENDMDL\nMODEL        2\n'+pdb_text()+'ENDMDL\n', 'MODEL'),
])
def test_unsupported_pdb_fails_early(tmp_path, text, reason):
    p = tmp_path / 'x.pdb'; p.write_text(text)
    with pytest.raises(ValueError, match=reason):
        read_chain(p, 'A')


@pytest.mark.parametrize('value', [np.zeros((2,3)), np.zeros((0,0)), [[np.nan]], [[-1]], [[2]], [['x']], [[1j]]])
def test_invalid_contact_map(value):
    with pytest.raises(ValueError):
        validate_cmap(value)


def test_cli_preflight_needs_no_gpu_or_output(tmp_path):
    out = tmp_path / 'never-created'
    p = subprocess.run([sys.executable, str(ROOT/'FoldCraft.py'), '--output_folder', str(out),
        '--binder_template', str(ROOT/'examples/templates/1qys1.pdb'),
        '--target_template', str(ROOT/'examples/targets/pd-l1-1.pdb'),
        '--target_hotspots', '30-34,50-54,69-76', '--binder_hotspots', '26-40,58-71', '--preflight_only'],
        capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert not out.exists()


def test_mpnn_includes_terminal_and_never_unfreezes_target(tmp_path):
    assert mutable_positions(3, [1], 'non-interface') == [2,3]
    assert mutable_positions(3, [], 'full') == [1,2,3]
    model = Mock()
    redesign(model, 'x', 3, [1], 'non-interface', .1, 5)
    assert model.prep_inputs.call_args.kwargs['fix_pos'] == 'B2,B3'
    assert model.prep_inputs.call_args.kwargs['inverse'] is True
    p = tmp_path/'x.pdb'; p.write_text(pdb_text()+pdb_text(chain='B'))
    model.reset_mock()
    result = redesign(model, str(p), 3, [1,2,3], 'non-interface', .1, 5)
    assert result['seq'] == ['AAA/AAA']
    model.prep_inputs.assert_not_called()
    model.sample_parallel.assert_not_called()


def test_atomic_write_failure_preserves_previous_checkpoint(tmp_path):
    p = tmp_path/'checkpoint'; p.write_text('old')
    def crash(temp):
        Path(temp).write_text('half')
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        atomic_write(p, crash)
    assert p.read_text() == 'old'
    assert list(tmp_path.iterdir()) == [p]


def test_stale_scorer_snapshots_preserve_each_others_columns(tmp_path):
    p = tmp_path/'results.csv'
    base = pd.DataFrame({'name':['a','b'], 'sequence':['AAA','BBB']}); base.to_csv(p,index=False)
    first = base.assign(score_one=[1.,2.]); second = base.iloc[::-1].assign(score_two=[3.,4.])
    publish_columns(p, first, ['score_one']); publish_columns(p, second, ['score_two'])
    merged = pd.read_csv(p).set_index('name')
    assert merged.loc['a','score_one'] == 1
    assert merged.loc['a','score_two'] == 4
    second.loc[second['name']=='a','sequence'] = 'CCC'
    with pytest.raises(ValueError, match='sequences'):
        publish_columns(p, second, ['score_two'])


def test_completion_requires_artifacts_and_detects_modification(tmp_path):
    (tmp_path/'results.csv').write_text('name\na\n')
    assert not completed_run(tmp_path)
    with pytest.raises(ValueError):
        finish_run(tmp_path, {'schema':1})
    (tmp_path/'designs').mkdir()
    for ext in ('.pdb','.pickle'):
        (tmp_path/'designs'/('a'+ext)).write_text('original')
    finish_run(tmp_path, {'schema':1})
    assert completed_run(tmp_path)
    (tmp_path/'designs'/'a.pdb').write_text('changed')
    assert not completed_run(tmp_path)


def test_rmsd_rejects_truncation_and_missing_internal_ca(tmp_path):
    a = np.array([[0,0,0],[1,1,0],[2,0,1]], dtype=float)
    assert ca_rmsd(a,a+3) == 0
    with pytest.raises(ValueError):
        ca_rmsd(a, np.vstack([a,a[-1]]))
    target = tmp_path/'template.pdb'; target.write_text(pdb_text())
    design = tmp_path/'design.pdb'; design.write_text(pdb_text((1,2,3,4),chain='B'))
    with pytest.raises(ValueError):
        rmsd_to_template(str(design), template_ca_atoms(str(target)))
    design.write_text(pdb_text(chain='B', missing=(2,'CA')))
    with pytest.raises(ValueError, match='Every polymer'):
        binder_seq_and_ca(str(design))


@pytest.mark.parametrize('pae,a,b,cutoff', [(np.zeros((4,3)),2,2,10), (np.full((4,4),np.nan),2,2,10), (np.full((4,4),-1),2,2,10), (np.zeros((4,4)),0,4,10), (np.zeros((4,4)),2,2,0)])
def test_ipsae_rejects_invalid_data(pae,a,b,cutoff):
    with pytest.raises(ValueError):
        ipsae(pae,a,b,cutoff)


def test_multiple_pae_models_not_silently_dropped():
    with pytest.raises(ValueError, match='multiple models'):
        single_model_pae(np.zeros((2,4,4)))
    assert single_model_pae(np.zeros((1,4,4))).shape == (4,4)


def test_zero_minimization_means_no_minimization():
    minimizer = Mock()
    minimize_if_requested(minimizer, 'context', 0)
    minimizer.minimize.assert_not_called()
    minimize_if_requested(minimizer, 'context', 5)
    minimizer.minimize.assert_called_once_with('context', maxIterations=5)
    with pytest.raises(ValueError):
        minimize_if_requested(minimizer, 'context', -1)


def test_boltz_cannot_select_stale_confidence(tmp_path, monkeypatch):
    from baseline import score_boltz2 as boltz
    stale = tmp_path/'confidence_old.json'; stale.write_text(json.dumps({'confidence_score':1.,'iptm':.99}))
    def predict(cmd, **kwargs):
        output = Path(cmd[cmd.index('--out_dir')+1])
        (output/'new.cif').write_text('new structure')
        (output/'confidence_new.json').write_text(json.dumps({'confidence_score':.2,'iptm':.2}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(boltz.subprocess, 'run', predict)
    assert boltz.run_boltz2('AAA','AAA',str(tmp_path),use_msa=False)[0] == .2


def test_score_cache_invalidates_changed_settings_and_structures(tmp_path):
    from baseline.score_cache import ScoreSession
    p = tmp_path/'results.csv'; (tmp_path/'designs').mkdir()
    structure = tmp_path/'designs'/'a.pdb'; structure.write_text('first')
    pd.DataFrame({'name':['a'], 'sequence':['AAA'], 'metric':[.9]}).to_csv(p,index=False)
    def session(setting):
        return ScoreSession(p, __file__, ['metric'], {'setting':setting})
    with session(1) as cache:
        df = cache.prepare(pd.read_csv(p))
        assert pd.isna(df.loc[0,'metric'])  # legacy CSV alone is not evidence
        df.loc[0,'metric'] = .7; cache.publish(df)
    with session(1) as cache:
        assert cache.prepare(pd.read_csv(p)).loc[0,'metric'] == .7
    with session(2) as cache:
        df = cache.prepare(pd.read_csv(p)); assert pd.isna(df.loc[0,'metric'])
        cache.publish(df)
    assert pd.isna(pd.read_csv(p).loc[0,'metric'])
    with session(2) as cache:
        df = cache.prepare(pd.read_csv(p)); df.loc[0,'metric'] = .4
        structure.write_text('modified during scoring')
        with pytest.raises(ValueError,match='inputs changed'):
            cache.publish(df)


def test_consensus_reports_missing_as_unknown():
    from baseline.consensus import consensus_counts
    frames = [pd.read_csv(p) for p in (ROOT/'baseline/repro').glob('*/results.csv')]
    result = consensus_counts(pd.concat(frames,ignore_index=True))
    assert result == dict(n=1000,known_pass=155,unknown=27,lower=155,upper=182)
    d = pd.DataFrame({'plddt':[.9,.2,np.nan],'iptm':[.9,.9,.9], 'ipae':[.2,.2,.2]})
    assert consensus_counts(d)['unknown'] == 2  # known AF2 failure needs no Boltz


def test_clashes_do_not_combine_alternative_models(tmp_path):
    from biopython_utils import calculate_clash_score
    p = tmp_path/'x.pdb'
    single = pdb_text((1,2,3))+pdb_text((1,2,3),chain='B')
    p.write_text(single)
    expected = calculate_clash_score(str(p))
    assert expected > 0
    p.write_text('MODEL        1\n'+single+'ENDMDL\nMODEL        2\n'+single+'ENDMDL\n')
    assert calculate_clash_score(str(p)) == expected


def test_main_failure_publishes_failed_state(tmp_path,monkeypatch):
    import FoldCraft
    p = tmp_path/'input.pdb'; p.write_text(pdb_text())
    out = tmp_path/'out'
    args = ['FoldCraft.py','--output_folder',str(out),'--target_template',str(p),
            '--binder_template',str(p),'--target_hotspots','1']
    monkeypatch.setattr(sys,'argv',args)
    def fail(*args):
        raise RuntimeError('weights unavailable')
    monkeypatch.setattr(FoldCraft,'execute',fail)
    with pytest.raises(RuntimeError):
        FoldCraft.main(supervised=False)
    state = json.loads((out/'run.json').read_text())
    assert state['status'] == 'failed'
    assert state['validation_models'] == ['model_1_ptm']
    with pytest.raises(SystemExit,match='already exists'):
        FoldCraft.main(supervised=False)


def test_scheduler_cleanup_on_launch_failure(tmp_path,monkeypatch):
    from baseline import scheduler as sch
    folds = [dict(fold='a',template='a.pdb',target='t.pdb',target_hotspots='1',binder_hotspots='1',total_traj=2,mem_gb=4)]
    monkeypatch.setattr(sch,'_nvidia_smi',lambda:'fake')
    monkeypatch.setattr(sch,'gpu_mem_gb',lambda *a:(24,24))
    proc = Mock(pid=12345); proc.poll.return_value = None
    logfile = Mock()
    calls = []
    def launch(*args):
        calls.append(args[0].tag)
        if len(calls)==2:
            raise RuntimeError('launch failed')
        return proc,logfile
    monkeypatch.setattr(sch,'launch',launch)
    kill = Mock(); monkeypatch.setattr(sch.os,'killpg',kill)
    with pytest.raises(RuntimeError,match='launch failed'):
        sch._run(folds,str(tmp_path/'runs'),str(tmp_path),['0'],1,2,sys.executable,poll_s=0,log=lambda *a:None)
    kill.assert_called_once()
    assert kill.call_args.args[0] == 12345
    proc.wait.assert_called_once_with(timeout=10)
    logfile.close.assert_called_once()


def test_scheduler_solo_job_blocks_new_neighbors(tmp_path,monkeypatch):
    from baseline import scheduler as sch
    folds = [dict(fold='a',template='a.pdb',target='t.pdb',target_hotspots='1',binder_hotspots='1',total_traj=2,mem_gb=4)]
    chunks = sch.plan_chunks(folds,1); chunks[0].solo = True
    completed = set(); polls = [0]
    monkeypatch.setattr(sch,'plan_chunks',lambda *a:chunks)
    monkeypatch.setattr(sch,'_nvidia_smi',lambda:'fake')
    monkeypatch.setattr(sch,'gpu_mem_gb',lambda *a:(24,24))
    monkeypatch.setattr(sch,'chunk_done',lambda c,*a:c.tag in completed)
    monkeypatch.setattr(sch,'merge_chunks',lambda *a:2)
    def launch(c,*a):
        if c.idx == 1:
            assert chunks[0].tag in completed, 'solo job acquired a neighbor'
        def poll():
            if c.idx == 0:
                polls[0] += 1
                if polls[0] == 1:
                    return None
            completed.add(c.tag)
            return 0
        proc = Mock(pid=c.idx+1); proc.poll.side_effect = poll
        return proc,Mock()
    monkeypatch.setattr(sch,'launch',launch)
    sch._run(folds,str(tmp_path/'runs'),str(tmp_path),['0'],1,2,sys.executable,poll_s=0,log=lambda *a:None)
    assert len(completed)==2


def test_failed_merge_does_not_publish_directory(tmp_path):
    from baseline.scheduler import merge_chunks
    chunk = tmp_path/'chunk'; (chunk/'designs').mkdir(parents=True)
    (chunk/'results.csv').write_text('name\na\n')
    (chunk/'designs'/'a.pdb').write_text('structure') # missing pickle
    template = tmp_path/'template.pdb'; template.write_text('template')
    with pytest.raises(FileNotFoundError):
        merge_chunks('a',[str(chunk)],str(tmp_path/'merged'),str(template))
    assert not (tmp_path/'merged').exists()
    assert not list(tmp_path.glob('*.merging-*'))


def test_campaign_failure_is_nonzero_without_running_gpu(tmp_path):
    import os
    fake = tmp_path/'python'; fake.write_text('#!/bin/sh\nexit 23\n'); fake.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'])
    result = subprocess.run(['bash',str(ROOT/'baseline/run_campaign.sh'),'1,1,1','1','1',str(tmp_path/'out with spaces')],env=env,capture_output=True,text=True)
    assert result.returncode != 0
    assert 'CAMPAIGN_FAILED folds=6' in result.stderr


def test_experimental_help_and_unsupported_default_are_cpu_only(tmp_path):
    script = str(ROOT/'test/FoldCraft_binder.py')
    assert subprocess.run([sys.executable,script,'--help'],capture_output=True).returncode == 0
    result = subprocess.run([sys.executable,script,'--output_folder',str(tmp_path/'out'),'--target_template','absent','--target_hotspots','1'],capture_output=True,text=True)
    assert result.returncode == 2 and 'requires --sample' in result.stderr
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('notebook', ['FoldCraft.ipynb','FoldCraft_VHH.ipynb'])
def test_notebook_order_reaches_shared_inference_adapter(notebook,monkeypatch):
    book = json.loads((ROOT/notebook).read_text())
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        # Test the actual CPU preflight in this interpreter, without inference.
        if '--preflight_only' in cmd:
            import FoldCraft
            old = sys.argv
            try:
                sys.argv = cmd[1:]
                FoldCraft.main(supervised=False)
            finally:
                sys.argv = old
        return SimpleNamespace(returncode=0)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(subprocess,'run',run)
    namespace = {'display':lambda x:None}
    for c in book['cells']:
        if c['cell_type']=='code':
            exec(compile(''.join(c['source']), notebook,'exec'), namespace)
    assert len(calls) == 2 and '--preflight_only' in calls[0]
    assert calls[1] == calls[0][:-1]
    assert ('--vhh' in calls[1]) == ('VHH' in notebook)


def test_vhh_probability_roundoff_preserves_historical_values():
    values = np.load(ROOT/'framework/vhh.npy')
    assert np.array_equal(validate_cmap(values,size=127),values)
    with pytest.raises(ValueError):
        validate_cmap(np.array([[1.0001]],dtype=np.float32))


@pytest.mark.parametrize('sample,successful', [(False,True),(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('validation_models', ['model_1_ptm','model_1_ptm,model_2_ptm'])
def test_driver_with_cpu_inference_doubles(tmp_path,monkeypatch,sample,successful,validation_models):
    """Exercise real loop control, artifact publication and quota termination."""
    import types
    import FoldCraft
    target = tmp_path/'input.pdb'; target.write_text(pdb_text())
    out = tmp_path/'out'
    args = ['FoldCraft.py','--output_folder',str(out),'--target_template',str(target),
            '--binder_template',str(target),'--target_hotspots','1','--num_designs','2',
            '--mpnn_samples','2','--max_trajectories','2','--target_success','1',
            '--validation_models',validation_models]
    if sample: args += ['--sample']
    monkeypatch.setattr(sys,'argv',args)
    predict_calls=[]; design_calls=[]
    class AF:
        def __init__(self, **kwargs):
            self._model_names=['model_1_ptm','model_2_ptm']
            self.opt={'weights':{}}; self._len=3; self._wt_aatype=np.zeros(3,dtype=int)
            self.aux={'cmap':np.eye(3), 'log':{'plddt':.9 if successful else .1,'i_pae':.1,'i_ptm':.9,'cmap_loss_binder':.2},'all':{'pae':np.zeros((1,6,6)), 'atom_positions':np.zeros((1,6,37,3))}}
        def prep_inputs(self, **kwargs): pass
        def set_seq(self, seq): pass
        def predict(self, **kwargs):
            predict_calls.append(kwargs)
            self.aux['log'].update(models=[self._model_names.index(kwargs.get('models',['model_1_ptm'])[0])],recycles=kwargs['num_recycles'])
        def restart(self,**kwargs): pass
        def design_3stage(self,*args):
            design_calls.append(args)
            self._tmp={'log':[dict(self.aux['log'], loss=.2, models=[0], recycles=0) for _ in range(sum(args))]}
        def save_pdb(self,path,**kwargs): Path(path).write_text(pdb_text()+pdb_text(chain='B'))
    class MPNN:
        def set_seed(self,*args):pass
        def prep_inputs(self,**kwargs):pass
        def sample_parallel(self,**kwargs):return {'seq':['AAA/AAA','AAA/GGG']}
    def mod(name, **attrs):
        value=types.ModuleType(name)
        for k,v in attrs.items():setattr(value,k,v)
        monkeypatch.setitem(sys.modules,name,value)
        return value
    mod('jax',numpy=np);monkeypatch.setitem(sys.modules,'jax.numpy',np)
    pyplot=mod('matplotlib.pyplot',subplots=lambda:(Mock(),Mock()),imshow=Mock(),savefig=Mock(),close=Mock())
    patches=mod('matplotlib.patches',Rectangle=Mock())
    mod('matplotlib',pyplot=pyplot,patches=patches)
    mod('colabdesign',mk_afdesign_model=AF,clear_mem=lambda:None)
    mod('colabdesign.af');mod('colabdesign.af.alphafold')
    mod('colabdesign.af.alphafold.common',residue_constants=SimpleNamespace(restypes=['A'],restype_num=1))
    mod('colabdesign.af.loss',get_contact_map=Mock())
    mod('colabdesign.mpnn',mk_mpnn_model=lambda *a,**k:MPNN())
    import biopython_utils
    monkeypatch.setattr(biopython_utils,'hotspot_residues',lambda *a:{1:'A'})
    if sample and not successful:
        with pytest.raises(SystemExit,match='budget exhausted'):
            FoldCraft.main(supervised=False)
        state=json.loads((out/'run.json').read_text())
        assert state['status']=='exhausted' and state['attempted_trajectories']==2
        assert not completed_run(out)
        assert (out/'results.csv.partial').is_file()
    else:
        FoldCraft.main(supervised=False)
        assert completed_run(out)
        results=pd.read_csv(out/'results.csv')
        assert len(results)==(1 if sample else 4)
        validations=[c for c in predict_calls if 'models' in c]
        expected_models=validation_models.split(',')
        assert len(validations)==len(results)*len(expected_models)
        assert [c['models'][0] for c in validations]==expected_models*len(results)
        assert all(c['num_models']==1 for c in validations)
    for artifact in (out/'designs').glob('*.pickle'):
        import pickle
        with artifact.open('rb') as stream:
            saved=pickle.load(stream)
        assert saved['_artifact_schema']=='foldcraft-prediction-compact-v1'
        np.testing.assert_array_equal(saved['atom_positions'],np.zeros((1,6,37,3)))
    attempts = json.loads((out/"attempts.json").read_text())
    assert all(row["accepted"] is successful for row in attempts if row["candidate"] is not None)
    assert len(design_calls)==(1 if sample and successful else 2)
    histories=list((out/'optimization').glob('*.json'))
    assert len(histories)==len(design_calls)
    assert all(len(json.loads(p.read_text())['records'])==220 for p in histories)
    if not (sample and not successful):
        histories[0].write_text('{}')
        assert not completed_run(out)


def test_stage_streams_and_arm_order_are_replayable():
    from run_state import stage_seed
    from baseline.experiment_protocol import arm_order
    seeds = [stage_seed(42, stage, i) for stage in ('design','mpnn','validation') for i in range(10)]
    assert len(set(seeds)) == 30
    assert all(0 <= s < 2**32 for s in seeds)
    assert stage_seed(42,'design',3) == stage_seed(42,'design',3)
    orders = [arm_order(['a','b'],42,i) for i in range(10)]
    assert orders == [arm_order(['a','b'],42,i) for i in range(10)]
    assert len({tuple(order) for order in orders}) == 2


@pytest.mark.parametrize('script',['ab_get_best','ab_loss','ab_loss_fp'])
def test_ab_driver_seeds_and_artifacts_with_cpu_doubles(tmp_path,monkeypatch,script):
    import importlib.util
    import types
    class AF:
        def __init__(self, **kw):
            self.opt={'weights':{}}; self._tmp={}
            self.aux={'log':dict(plddt=.9,i_ptm=.9,i_pae=.1,cmap_loss_binder=.2),
                      'all':{'pae':np.zeros((1,6,6))}}
        def prep_inputs(self,**kw): pass
        def set_seq(self,seq):pass
        def restart(self,**kw): design_seeds.append(kw['seed'])
        def design_3stage(self,*a):pass
        def predict(self,**kw): predict_seeds.append(kw['seed'])
        def save_pdb(self,path,**kw): Path(path).write_text(pdb_text()+pdb_text(chain='B'))
    class MPNN:
        def set_seed(self,seed):mpnn_seeds.append(seed)
        def prep_inputs(self,**kw):pass
        def sample_parallel(self,**kw):return {'seq':['AAA/AAA','AAA/GGG']}
    design_seeds=[];mpnn_seeds=[];predict_seeds=[]
    def mod(name,**attrs):
        module=types.ModuleType(name)
        for k,v in attrs.items():setattr(module,k,v)
        monkeypatch.setitem(sys.modules,name,module)
    mod('jax',numpy=np);monkeypatch.setitem(sys.modules,'jax.numpy',np)
    mod('colabdesign',mk_afdesign_model=AF,clear_mem=lambda:None)
    mod('colabdesign.af');mod('colabdesign.af.alphafold')
    mod('colabdesign.af.alphafold.common',residue_constants=SimpleNamespace())
    mod('colabdesign.af.loss',get_contact_map=Mock())
    mod('colabdesign.mpnn',mk_mpnn_model=lambda *a,**kw:MPNN())
    monkeypatch.syspath_prepend(str(ROOT/'baseline'))
    def load(name):
        spec=importlib.util.spec_from_file_location(name,ROOT/'baseline'/(name+'.py'))
        module=importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules,name,module);spec.loader.exec_module(module)
        return module
    shared=load('ab_get_best')
    from baseline.scheduler import load_config
    protocol={f['fold']:f for f in load_config(ROOT/'baseline/repro_config.tsv')}
    for fold,(template,hotspots) in shared.FOLDS.items():
        assert hotspots == protocol[fold]['binder_hotspots']
        assert Path(template) == ROOT/protocol[fold]['template']
    monkeypatch.setattr(shared,'build_cond_cmap',lambda *a,**kw:(np.eye(6),np.eye(6),3))
    monkeypatch.setattr(shared,'hotspot_residues',lambda *a:{1:'A'})
    module=shared if script=='ab_get_best' else load(script)
    out=tmp_path/script
    monkeypatch.setattr(sys,'argv',[script,'--n','1','--mpnn-samples','2','--seed','42','--out',str(out)])
    module.main()
    receipt=json.loads((out/'experiment.json').read_text())
    arms=len(receipt['arms'])
    assert len(set(design_seeds)) == 1
    assert len(mpnn_seeds) == arms and len(set(mpnn_seeds)) == 1
    assert predict_seeds == predict_seeds[:2]*arms
    for arm in receipt['arms']:
        df=pd.read_csv(out/arm/'results.csv')
        assert len(df)==2
        for name in df['name']:
            assert (out/arm/'designs'/(name+'.pickle')).is_file()
