"""Exercise the experimental loop; mocks cannot certify PyRosetta numerics."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
from unittest.mock import Mock
import pandas as pd
import pytest
from test_cpu_contracts import pdb_text

ROOT=Path(__file__).resolve().parents[1]


def load():
    spec=importlib.util.spec_from_file_location('experimental_driver',ROOT/'test/FoldCraft_binder.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def argv(tmp_path):
    target=tmp_path/'input.pdb';target.write_text(pdb_text())
    return ['binder','--sample','--target_template',str(target),'--target_hotspots','1',
            '--output_folder',str(tmp_path/'out'),'--binder_len','3-4','--seed','42',
            '--max_trajectories','2','--target_success','1','--mpnn_samples','2',
            '--design_stages','1,1,1']


@pytest.mark.parametrize('successful',[True,False])
def test_actual_experimental_loop_and_failure_state(tmp_path,monkeypatch,successful):
    driver=load();monkeypatch.setattr(sys,'argv',argv(tmp_path))
    lengths=[];predicts=[];seeds=[];saved=[]
    def mod(name,**values):
        m=types.ModuleType(name)
        for k,v in values.items():setattr(m,k,v)
        monkeypatch.setitem(sys.modules,name,m)
    class AF:
        def __init__(self,**kw):
            assert kw['data_dir']==str(ROOT)
            assert kw['model_names']==['model_1_ptm','model_2_ptm']
            self.opt={'weights':{}}
            self.aux={'log':dict(plddt=.9,i_ptm=.8,i_pae=.1,rg=.2,rmsd=.3),'all':{}}
        def prep_inputs(self,**kw):
            if 'binder_len' in kw:lengths.append(kw['binder_len'])
        def restart(self,**kw):seeds.append(kw['seed'])
        def design_3stage(self,*a):pass
        def save_pdb(self,path,**kw):
            saved.append(kw);Path(path).write_text(pdb_text()+pdb_text(chain='B'))
        def set_seq(self,*a):pass
        def predict(self,**kw):predicts.append(kw)
    class MPNN:
        def set_seed(self,seed):seeds.append(seed)
        def prep_inputs(self,**kw):pass
        def sample_parallel(self,**kw):return {'seq':['AAA/AAAA','AAA/GGGG']}
    mod('jax');mod('jax.numpy')
    mod('colabdesign',mk_afdesign_model=AF,clear_mem=lambda:None)
    mod('colabdesign.af');mod('colabdesign.af.alphafold')
    mod('colabdesign.af.alphafold.common',residue_constants=Mock())
    mod('colabdesign.mpnn',mk_mpnn_model=lambda *a,**k:MPNN())
    mod('bindcraft_deps',ensure_bindcraft_importable=lambda:None,dalphaball_path=lambda:'unused')
    mod('pyrosetta',init=lambda *a:None)
    mod('BindCraft');mod('BindCraft.functions')
    scores=dict(interface_sc=.7 if successful else .1,interface_delta_unsat_hbonds=0,
                surface_hydrophobicity=.1,interface_dG=-3.)
    mod('BindCraft.functions.pyrosetta_utils',pr_relax=shutil.copyfile,
        score_interface=lambda *a:(scores,{},{}))
    import biopython_utils
    monkeypatch.setattr(biopython_utils,'hotspot_residues',lambda *a:{1:'A'})
    monkeypatch.setattr(biopython_utils,'calculate_clash_score',lambda *a:0)
    if successful:driver.main(supervised=False)
    else:
        with pytest.raises(driver.BudgetExhausted):driver.main(supervised=False)
    state=json.loads((tmp_path/'out/experimental_run.json').read_text())
    assert state['status']==('complete' if successful else 'exhausted')
    assert state['config']['seed']==42
    assert all(3<=n<=4 for n in lengths)
    assert all(p['models']==['model_1_ptm'] and p['num_models']==1 and 'seed' in p for p in predicts)
    assert all(p['get_best'] is False for p in saved)
    assert seeds
    if successful:
        frame=pd.read_csv(tmp_path/'out/results.csv')
        assert len(frame)==1 and frame.iloc[0]['rmsds']==.3
        assert 'results.csv' in state['artifacts']
        (tmp_path/'out/relaxed/traj_1_0.pdb').unlink()
        with pytest.raises(ValueError,match='Missing'):driver.verify_outputs(tmp_path/'out',1)
    else:assert not (tmp_path/'out/results.csv').exists()


def test_preflight_and_dependency_failure_are_reached_without_local_os_bug(tmp_path,monkeypatch):
    driver=load();args=argv(tmp_path)
    monkeypatch.setattr(sys,'argv',args+['--preflight_only'])
    driver.main();assert not (tmp_path/'out').exists()
    monkeypatch.setattr(sys,'argv',args)
    def fail(*a):raise ImportError('missing test dependency')
    monkeypatch.setattr(driver,'execute',fail)
    with pytest.raises(ImportError):driver.main(supervised=False)
    assert json.loads((tmp_path/'out/experimental_run.json').read_text())['status']=='failed'


def test_experimental_supervisor_propagates_timeout(tmp_path,monkeypatch):
    from run_watchdog import Outcome
    import run_watchdog
    driver=load();monkeypatch.setattr(sys,'argv',argv(tmp_path))
    calls=[]
    def timeout(command,seconds):
        calls.append((command,seconds))
        out=tmp_path/'out';out.mkdir();(out/'experimental_run.json').write_text('{"status":"running"}')
        return Outcome(124,'timed_out',.1)
    monkeypatch.setattr(run_watchdog,'run_owned',timeout)
    with pytest.raises(SystemExit) as exc:driver.main()
    assert exc.value.code==124
    assert calls[0][0][-1]=='--_worker'
    assert json.loads((tmp_path/'out/experimental_run.json').read_text())['status']=='timed_out'


@pytest.mark.parametrize('extra',[['--vhh'],['--binder_template','ignored.pdb'],['--timeout_minutes','nan'],['--seed','-1']])
def test_unsupported_or_invalid_arguments_fail_early(tmp_path,monkeypatch,extra):
    driver=load()
    with pytest.raises(SystemExit):driver.parse_args(argv(tmp_path)[1:]+extra)


def test_nonfinite_physical_energy_cannot_be_accepted():
    driver=load()
    scores=dict(interface_sc=.8,interface_delta_unsat_hbonds=0,surface_hydrophobicity=.1,interface_dG=float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):driver.physical_passes(0,scores)


def test_successful_exit_without_results_is_not_completion(tmp_path,monkeypatch):
    from run_watchdog import Outcome
    import run_watchdog
    driver=load();monkeypatch.setattr(sys,'argv',argv(tmp_path))
    monkeypatch.setattr(run_watchdog,'run_owned',lambda *a:Outcome(0,'exited',.1))
    with pytest.raises(RuntimeError,match='manifest'):driver.main()
