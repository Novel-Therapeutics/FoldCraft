from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from baseline.score_cache import ScoreSession
from baseline.structure_checks import force_qc
from baseline.checkpoint_files import af2_checkpoint, esm_snapshot


def test_checkpoint_contents_invalidate_scores_at_the_same_path(tmp_path):
    csv=tmp_path/'results.csv';weights=tmp_path/'model.ckpt'
    pd.DataFrame({'name':['a'],'sequence':['AAA']}).to_csv(csv,index=False)
    weights.write_bytes(b'original')
    def session():
        return ScoreSession(csv,__file__,['metric'],{},structure=False,model_inputs={'model':weights})
    with session() as cache:
        frame=cache.prepare(pd.read_csv(csv));frame.loc[0,'metric']=.9
        cache.annotate('a',status='converged',rms_force=2.);cache.publish(frame)
    with session() as cache:
        assert cache.prepare(pd.read_csv(csv)).loc[0,'metric']==.9
        assert cache.diagnostics['a']['rms_force']==2.
    weights.write_bytes(b'modified')
    with session() as cache:
        frame=cache.prepare(pd.read_csv(csv));assert pd.isna(frame.loc[0,'metric'])
        frame.loc[0,'metric']=.1
        weights.write_bytes(b'replaced')
        with pytest.raises(ValueError,match='Model inputs changed'):
            cache.publish(frame)


def test_snapshot_addition_or_missing_weights_cannot_reuse_scores(tmp_path):
    csv=tmp_path/'results.csv';pd.DataFrame({'name':['a'],'sequence':['AAA']}).to_csv(csv,index=False)
    model=tmp_path/'snapshot';model.mkdir();(model/'weights.bin').write_bytes(b'weights')
    with ScoreSession(csv,__file__,['metric'],{},structure=False,model_inputs={'snapshot':model}) as cache:
        frame=cache.prepare(pd.read_csv(csv));frame.loc[0,'metric']=.9
        (model/'config.json').write_text('{}')
        with pytest.raises(ValueError,match='files changed'):cache.publish(frame)
    with pytest.raises(ValueError,match='Missing model input'):
        ScoreSession(csv,__file__,['metric'],{},model_inputs={'missing':tmp_path/'missing'})
    with pytest.raises(FileNotFoundError):esm_snapshot(model)


@pytest.mark.parametrize('forces', [[[float('nan'),0,0]],[[float('inf'),0,0]],[]])
def test_nonfinite_or_empty_forces_are_not_valid_scores(forces):
    with pytest.raises(ValueError):force_qc(forces,[[0.,0.,0.]],-10.,10.,True)


def test_force_qc_distinguishes_converged_capped_and_raw_geometry():
    xyz=np.zeros((2,3));forces=np.full((2,3),11.)
    assert force_qc(forces,xyz,-20.,10.,True)['status']=='unconverged'
    assert force_qc(forces,xyz,-20.,10.,False)['status']=='raw'
    assert force_qc(forces/2,xyz,-20.,10.,True)['converged'] is True
    with pytest.raises(ValueError):force_qc(forces,xyz,float('nan'),10.,True)


def test_af2_resolves_the_file_the_loader_will_use(tmp_path):
    (tmp_path/'params').mkdir()
    primary=tmp_path/'params/params_model_1_ptm.npz';primary.write_bytes(b'1')
    (tmp_path/'model_1_ptm.npz').write_bytes(b'2')
    assert af2_checkpoint(tmp_path)==primary
    with pytest.raises(FileNotFoundError):af2_checkpoint(tmp_path,'model_2_ptm')


def test_boltz_receives_explicit_weights_and_seed(tmp_path,monkeypatch):
    from baseline import score_boltz2 as boltz
    def run(command,**kwargs):
        assert command[command.index('--checkpoint')+1]=='chosen.ckpt'
        assert command[command.index('--cache')+1]=='chosen-cache'
        assert command[command.index('--seed')+1]=='42'
        assert command[command.index('--model')+1]=='boltz2'
        assert '--flash_attn' not in command
        out=Path(command[command.index('--out_dir')+1])
        (out/'0.cif').write_text('selected structure')
        (out/'confidence_0.json').write_text(json.dumps({'iptm':.5}))
        return type('Result',(),{'returncode':0, 'stdout':'', 'stderr':''})()
    monkeypatch.setattr(boltz.subprocess,'run',run)
    assert boltz.run_boltz2('AAA','AAA',str(tmp_path),use_msa=False,flash_attn=False,
                           checkpoint='chosen.ckpt',cache='chosen-cache',seed=42)[0]==.5


def test_target_changed_after_session_construction_is_rejected(tmp_path):
    csv=tmp_path/'results.csv';target=tmp_path/'target.pdb';target.write_text('original')
    frame=pd.DataFrame({'name':['a'],'sequence':['AAA']});frame.to_csv(csv,index=False)
    with ScoreSession(csv,__file__,['metric'],{},extra_inputs=[target],structure=False) as cache:
        target.write_text('changed')
        with pytest.raises(ValueError,match='before scoring'):cache.prepare(frame)


def test_openmm_comparison_cannot_ignore_failed_second_model():
    from baseline.openmm_compare import af2_mask
    frame=pd.DataFrame(dict(plddt=[.9,.9],iptm=[.8,.8],ipae=[.1,.1],validation_pass=[True,False]))
    assert af2_mask(frame).tolist()==[True,False]


def test_esmfold_fractional_confidence_is_converted_before_rounding():
    from baseline.structure_checks import fractional_plddt_percent
    assert fractional_plddt_percent([.7,.9])==80.
    for bad in ([float('nan')],[],[-.1],[80.]):
        with pytest.raises(ValueError):fractional_plddt_percent(bad)
