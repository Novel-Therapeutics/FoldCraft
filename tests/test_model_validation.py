from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import pytest

from model_validation import validate_candidate, candidate_files, model_names
from run_state import finish_run, completed_run
from baseline.scheduler import merge_chunks
from baseline.gates import af2_mask
from baseline.consensus import consensus_counts


class Predictor:
    _model_names = ['model_1_ptm','model_2_ptm']

    def __init__(self, fail_second=False, wrong_model=False):
        self.calls=[]; self.fail_second=fail_second; self.wrong_model=wrong_model

    def set_seq(self, sequence):
        self.sequence=sequence

    def predict(self, **kw):
        self.calls.append(kw)
        n=self._model_names.index(kw['models'][0])
        if n == 1 and self.fail_second:
            raise RuntimeError('GPU failed during second model')
        self.aux={'log':dict(models=[0 if self.wrong_model else n],recycles=kw['num_recycles'],
                            plddt=.9,i_ptm=.8 if n==0 else .2,i_pae=.1,cmap_loss_binder=float(n)),
                  'all':{'pae':np.full((1,6,6),n+1.),
                         'atom_positions':np.full((1,6,37,3),float(n))}}
        self.current=n

    def save_pdb(self, path, **kw):
        Path(path).write_text(f'model={self.current}; sequence={self.sequence}\n')


def validate(tmp_path, predictor=None):
    p=predictor or Predictor()
    report=validate_candidate(p,'AAA/GGG',3,tmp_path,'traj_1_0',
                              ['model_1_ptm','model_2_ptm'],3,42)
    return p,report


def test_models_keep_individual_structures_scores_and_pae(tmp_path):
    predictor,report=validate(tmp_path)
    assert [c['models'] for c in predictor.calls] == [['model_1_ptm'],['model_2_ptm']]
    assert all(c['num_models']==1 and not c['sample_models'] for c in predictor.calls)
    assert report['model_disagreement'] and not report['all_models_pass']
    assert report['models']['model_1_ptm']['metrics']['i_ptm'] == .8
    assert report['models']['model_2_ptm']['metrics']['i_ptm'] == .2
    for n,suffix in [(0,''),(1,'.model_2_ptm')]:
        assert f'model={n}' in (tmp_path/'designs'/f'traj_1_0{suffix}.pdb').read_text()
        with open(tmp_path/'designs'/f'traj_1_0{suffix}.pickle','rb') as stream:
            assert np.all(pickle.load(stream)['pae']==n+1)
    assert len(candidate_files(tmp_path,'traj_1_0',require_receipt=True)) == 5


def test_mid_ensemble_failure_cannot_publish_receipt(tmp_path):
    with pytest.raises(RuntimeError):validate(tmp_path,Predictor(fail_second=True))
    assert not (tmp_path/'designs/traj_1_0.validation.json').exists()
    with pytest.raises(ValueError,match='Missing validation receipt'):
        candidate_files(tmp_path,'traj_1_0',require_receipt=True)
    with pytest.raises(FileExistsError):validate(tmp_path)


def test_dispatch_and_missing_weights_fail_loudly(tmp_path):
    with pytest.raises(RuntimeError,match='dispatch mismatch'):
        validate(tmp_path,Predictor(wrong_model=True))
    p=Predictor();p._model_names=['model_1_ptm']
    with pytest.raises(ValueError,match='weights unavailable'):
        validate(tmp_path/'missing',p)
    assert not p.calls


@pytest.mark.parametrize('names',['','model_3_ptm','model_1_ptm,model_1_ptm'])
def test_invalid_models(names):
    with pytest.raises(ValueError):model_names(names)


def completed_fixture(folder):
    _,report=validate(folder)
    pd.DataFrame([dict(name='traj_1_0',sequence='AAA/GGG',plddt=.9,iptm=.8,ipae=.1,
                      primary_model=report['primary_model'],validation_pass=False)]).to_csv(folder/'results.csv',index=False)
    finish_run(folder,dict(schema=1,validation_artifacts='per-model-v1'))


def test_completion_and_merge_require_secondary_artifacts(tmp_path):
    source=tmp_path/'chunk';completed_fixture(source)
    assert completed_run(source)
    template=tmp_path/'template.pdb';template.write_text('template')
    out=tmp_path/'merged'
    assert merge_chunks('fold',[str(source)],str(out),str(template))==1
    assert completed_run(out)
    assert (out/'designs/c0_traj_1_0.model_2_ptm.pdb').exists()
    (out/'designs/c0_traj_1_0.model_2_ptm.pickle').unlink()
    assert not completed_run(out)


def test_downstream_gates_respect_disagreement_and_missing_decisions():
    d=pd.DataFrame(dict(plddt=[.9]*3,iptm=[.8]*3,ipae=[.1]*3,boltz2_iptm=[.8]*3,
                        validation_pass=[True,False,None]))
    assert af2_mask(d).tolist()==[True,False,False]
    assert consensus_counts(d)==dict(n=3,known_pass=1,unknown=1,lower=1,upper=2)


def test_receipt_hash_mismatch_cannot_be_finalized(tmp_path):
    completed_fixture(tmp_path)
    (tmp_path/'designs/traj_1_0.model_2_ptm.pdb').write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):
        finish_run(tmp_path,dict(schema=1,validation_artifacts='per-model-v1'))
    assert not completed_run(tmp_path)


def test_gpu_smoke_plan_never_touches_gpu_or_starts_processes(tmp_path,monkeypatch,capsys):
    from scripts import gpu_smoke
    monkeypatch.setattr(gpu_smoke.subprocess,'run',lambda *a,**kw:pytest.fail('Unexpected process execution'))
    out=tmp_path/'future-run'
    gpu_smoke.main(['--output-root',str(out),'--data-dir',str(tmp_path/'weights')])
    data=json.loads(capsys.readouterr().out)
    assert data['status']=='plan_only' and len(data['cases'])==5
    assert not out.exists()
    for case in data['cases']:
        command=case['command']
        assert command[command.index('--validation_models')+1]=='model_1_ptm,model_2_ptm'


def test_checkpoint_lookup_matches_pinned_dependency(tmp_path):
    from scripts.gpu_probe import checkpoint_paths
    (tmp_path/'params').mkdir()
    (tmp_path/'params/params_model_1_ptm.npz').write_bytes(b'first')
    (tmp_path/'model_1_ptm.npz').write_bytes(b'ignored fallback')
    (tmp_path/'model_2_ptm.npz').write_bytes(b'second')
    files=checkpoint_paths(tmp_path)
    assert files['model_1_ptm']==tmp_path/'params/params_model_1_ptm.npz'
    assert files['model_2_ptm']==tmp_path/'model_2_ptm.npz'


def test_smoke_timeout_terminates_only_owned_process_group(monkeypatch):
    import subprocess
    from unittest.mock import Mock
    from scripts import gpu_smoke
    process=Mock(pid=54321)
    process.wait.side_effect=[subprocess.TimeoutExpired('command',1),0]
    process.poll.return_value=None
    monkeypatch.setattr(gpu_smoke.subprocess,'Popen',Mock(return_value=process))
    kill=Mock();monkeypatch.setattr(gpu_smoke.os,'killpg',kill)
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_smoke.run_owned(['fake'],env={},timeout=1)
    kill.assert_called_once_with(54321,gpu_smoke.signal.SIGTERM)


def test_multimodel_ipSAE_export_keeps_each_pae_separate(tmp_path,monkeypatch):
    import sys
    from baseline import add_ipsae
    folder=tmp_path/'fold'
    validate(folder)
    pd.DataFrame([dict(name='traj_1_0',sequence='AAA/GGG')]).to_csv(folder/'results.csv',index=False)
    monkeypatch.setattr(add_ipsae,'complex_chain_lens',lambda path:(3,3))
    monkeypatch.setattr(sys,'argv',['add_ipsae.py',str(tmp_path),'10'])
    add_ipsae.main()
    scores=json.loads((folder/'ipsae.models.json').read_text())['candidates']['traj_1_0']
    assert scores['model_1_ptm']['ipsae'] > scores['model_2_ptm']['ipsae']
    assert pd.read_csv(folder/'results.csv').loc[0,'ipsae']==scores['model_1_ptm']['ipsae']


def test_csv_cannot_claim_secondary_scores_for_primary_structure(tmp_path):
    completed_fixture(tmp_path)
    frame=pd.read_csv(tmp_path/'results.csv')
    frame.loc[0,'iptm']=.2
    frame.to_csv(tmp_path/'results.csv',index=False)
    assert not completed_run(tmp_path)


def test_gpu_replay_checks_coordinates_even_when_scores_match(tmp_path):
    from scripts.gpu_smoke import replay_check
    validate(tmp_path/'fixed')
    validate(tmp_path/'replay')
    assert replay_check(tmp_path)['max_coordinate_delta']==0
    path=tmp_path/'replay/designs/traj_1_0.model_2_ptm.pickle'
    with path.open('rb') as stream: data=pickle.load(stream)
    data['atom_positions'][0,0,1,0]+=.01
    with path.open('wb') as stream: pickle.dump(data,stream)
    with pytest.raises(ValueError,match='coordinates'):
        replay_check(tmp_path)
