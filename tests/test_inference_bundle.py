import json
import os
from pathlib import Path
import shlex
import sys
from types import SimpleNamespace

import pytest
import inference_bundle as b
from baseline import scheduler as s
from baseline.result_io import write_json
from run_state import completed_run, finish_run


def capture(files, **kwargs):
    return b.capture_bundle(files['data'],repo_root=files['repo'],**kwargs)


def completed_chunk(tmp_path, bundle):
    target=tmp_path/'input.pdb';target.write_text('test input')
    spec=dict(target=str(target),template=str(target),target_hotspots='1',binder_hotspots='1')
    chunk=s.Chunk('fold',0,1,1.,spec);chunk.runtime_bundle=bundle
    out=Path(chunk.out_dir(tmp_path));(out/'designs').mkdir(parents=True)
    (out/'results.csv').write_text('name,sequence\nx,AAA\n')
    for ext in ('.pdb','.pickle'):(out/'designs'/('x'+ext)).write_text('artifact')
    finish_run(out,dict(schema=1,inference_bundle=bundle))
    write_json(str(out)+'.job.json',s.chunk_signature(chunk))
    return chunk,out


@pytest.mark.parametrize('label', ['af2:model_1_ptm','af2:model_2_ptm','mpnn:v_48_010','conditioning:vhh'])
def test_inplace_same_size_same_mtime_replacement_invalidates_resume(tmp_path,inference_files,label):
    old=capture(inference_files,vhh=True);chunk,out=completed_chunk(tmp_path,old)
    assert s.chunk_done(chunk,tmp_path)
    path=Path(old['identity']['files'][label]['path']);st=path.stat();path.write_bytes(b'X'*st.st_size)
    os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    with pytest.raises(ValueError):b.validate_snapshot(old)
    fresh=capture(inference_files,vhh=True)
    assert fresh['identity']!=old['identity']
    chunk.runtime_bundle=fresh
    assert not s.chunk_done(chunk,tmp_path)
    # Historic artifacts remain independently readable; never rewrite them.
    assert completed_run(out)


def test_unchanged_content_touch_between_runs_still_resumes(tmp_path,inference_files):
    old=capture(inference_files);chunk,_=completed_chunk(tmp_path,old)
    Path(old['identity']['files']['af2:model_1_ptm']['path']).touch()
    fresh=capture(inference_files)
    assert fresh['identity']==old['identity']
    chunk.runtime_bundle=fresh
    assert s.chunk_done(chunk,tmp_path)


def test_changed_lookup_precedence_is_detected(inference_files):
    old=capture(inference_files)
    top=inference_files['data']/'params';top.mkdir()
    (top/'params_model_1_ptm.npz').write_bytes(b'original-weight')
    with pytest.raises(ValueError):b.validate_snapshot(old)
    assert capture(inference_files)['identity']!=old['identity']


def test_new_optional_model_changes_effective_bundle(inference_files):
    path=inference_files['data']/'params_model_2_ptm.npz';path.unlink()
    old=capture(inference_files)
    path.write_bytes(b'new model')
    with pytest.raises(ValueError,match='file set'):b.validate_snapshot(old)
    assert capture(inference_files)['identity']!=old['identity']


def test_missing_requested_model_or_mpnn_fails_before_inference(inference_files):
    (inference_files['data']/'params_model_2_ptm.npz').unlink()
    with pytest.raises(FileNotFoundError):capture(inference_files,validation_models='model_1_ptm,model_2_ptm')
    (inference_files['package']/'mpnn/weights_soluble/v_48_010.pkl').unlink()
    with pytest.raises(FileNotFoundError):capture(inference_files)


def test_variants_and_vhh_only_include_effective_inputs(inference_files):
    soluble=capture(inference_files)
    original=capture(inference_files,mpnn_weight='original')
    assert 'weights_soluble' in soluble['identity']['files']['mpnn:v_48_010']['path']
    assert '/weights/' in original['identity']['files']['mpnn:v_48_010']['path']
    assert 'conditioning:vhh' not in soluble['identity']['files']
    (inference_files['repo']/'framework/vhh.npy').write_bytes(b'changed')
    b.validate_snapshot(soluble)


def test_midrun_change_cannot_publish_completion(tmp_path,inference_files):
    old=capture(inference_files)
    folder=tmp_path/'run';folder.mkdir()
    state=dict(schema=1,status='running',inference_bundle=old)
    write_json(folder/'run.json',state)
    (folder/'results.csv.partial').write_text('retained checkpoint')
    path=Path(old['identity']['files']['mpnn:v_48_010']['path'])
    original=path.read_bytes();path.write_bytes(b'changed');path.write_bytes(original)
    # Even changing and restoring bytes in one run must not certify mixed loads.
    with pytest.raises(ValueError):finish_run(folder,state)
    assert json.loads((folder/'run.json').read_text())['status']=='failed'
    assert (folder/'results.csv.partial').read_text()=='retained checkpoint'
    assert not completed_run(folder)


def test_change_while_hashing_is_rejected(inference_files,monkeypatch):
    original=b.sha256
    def mutate(path):
        result=original(path)
        if path.suffix=='.npz':path.write_bytes(b'changed')
        return result
    monkeypatch.setattr(b,'sha256',mutate)
    with pytest.raises(ValueError,match='while hashing'):capture(inference_files)


def test_selected_python_discovers_weights_without_importing_gpu_package(tmp_path,inference_files):
    # The selected interpreter uses a package search path different from the
    # scheduler's. __init__ raises if discovery accidentally imports ColabDesign.
    wrapper=tmp_path/'worker python'
    wrapper.write_text('#!/bin/sh\nPYTHONPATH='+shlex.quote(str(inference_files['package'].parent))+
                       ' exec '+shlex.quote(sys.executable)+' "$@"\n')
    wrapper.chmod(0o755)
    root=Path(__file__).resolve().parents[1]
    bundle=s.capture_runtime(str(wrapper),root,inference_files['data'])
    assert bundle['identity']['package_root']==str(inference_files['package'].resolve())
    assert 'mpnn:v_48_010' in bundle['identity']['files']


def test_old_or_mismatched_worker_bundle_never_resumes(tmp_path,inference_files):
    bundle=capture(inference_files);chunk,out=completed_chunk(tmp_path,bundle)
    state=json.loads((out/'run.json').read_text());state.pop('inference_bundle')
    write_json(out/'run.json',state)
    assert not s.chunk_done(chunk,tmp_path)
    assert completed_run(out)


def test_worker_mismatch_fails_before_creating_output(tmp_path,inference_files,monkeypatch):
    import FoldCraft
    args=SimpleNamespace(output_folder=str(tmp_path/'out'),preflight_only=False,
                         data_dir=str(inference_files['data']),mpnn_weight='soluble',vhh=False,
                         validation_models='model_1_ptm',expected_bundle=str(tmp_path/'job.json'))
    write_json(args.expected_bundle,dict(inference_bundle={'different':'identity'}))
    monkeypatch.setattr(FoldCraft,'parse_args',lambda:args)
    monkeypatch.setattr(FoldCraft,'preflight',lambda a:None)
    monkeypatch.setattr(FoldCraft,'execute',lambda *a:pytest.fail('Inference started'))
    with pytest.raises(ValueError,match='scheduler plan'):FoldCraft.main()
    assert not Path(args.output_folder).exists()


def test_guarded_model_loading_rejects_mutation_and_wrong_dispatch(inference_files):
    bundle=capture(inference_files)
    with pytest.raises(ValueError,match='model set'):
        b.guarded_load(bundle,lambda **kw:SimpleNamespace(_model_names=['model_2_ptm']),model_names=['model_1_ptm'])
    def mutate():
        Path(bundle['identity']['files']['mpnn:v_48_010']['path']).write_bytes(b'mutated during load')
        return object()
    with pytest.raises(ValueError,match='changed'):b.guarded_load(bundle,mutate)
