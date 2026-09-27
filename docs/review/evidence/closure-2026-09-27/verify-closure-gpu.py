"""VHH archive reproduction, unchanged-current replay, and full scheduler history."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
from scripts.gpu_smoke import plan,run_owned,replay_check
from baseline.result_io import write_json
from run_state import completed_run
base=Path('/home/bizon/projects/foldcraft-gpu-20260927');repo=base/'closure-repo'
root=base/'closure-extra-01';root.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',
         JAX_COMPILATION_CACHE_DIR=str(base/'jax-cache'))
report={'status':'running'}
try:
    smoke=json.loads((base/'closure-smoke-01/smoke.json').read_text());assert smoke['status']=='passed'
    report['smoke']=smoke
    for name in ('fixed','vhh','long_target'):
        pair=root/f'parity-{name}';pair.mkdir()
        (pair/'fixed').symlink_to(base/'closure-smoke-01'/name,target_is_directory=True)
        (pair/'replay').symlink_to(base/'bundle-smoke-01'/name,target_is_directory=True)
        report[f'unchanged_{name}']=replay_check(pair)
    command=next(c['command'] for c in plan(root,base/'weights',sys.executable,3) if c['name']=='vhh')
    command += ['--vhh_convention','foldcraft-127-historical-v0']
    with (root/'historical-vhh.log').open('w') as log:
        code=run_owned(command,env=env,timeout=300,stdout=log,stderr=-2)
    assert code==0 and completed_run(root/'vhh')
    np.testing.assert_array_equal(np.load(root/'vhh/fold_cond_cmap.npy'),np.load(repo/'examples/cmaps/vhh_pd_l1.npy'))
    np.testing.assert_array_equal(np.load(root/'vhh/fold_cond_cmap_mask.npy'),np.load(repo/'examples/cmaps/vhh_pd_l1_mask.npy'))
    state=json.loads((root/'vhh/run.json').read_text())
    assert state['vhh_convention']=='foldcraft-127-historical-v0'
    report['historical_vhh_exact_maps']=True
    config=root/'config.tsv'
    config.write_text((base/'bundle-campaign-01/config.tsv').read_text().replace('bundle-repo','closure-repo'))
    command=[sys.executable,str(repo/'baseline/scheduler.py'),str(config),'--repo',str(repo),
             '--repro',str(root/'runs'),'--data-dir',str(base/'weights'),'--gpus','0','--chunk-traj','1','--timeout-minutes','10']
    with (root/'campaign.log').open('w') as log:
        code=run_owned(command,env=env,timeout=600,stdout=log,stderr=-2)
    assert code==0
    for name in ('audit__c0','audit'):
        out=root/'runs'/name;assert completed_run(out)
        files=list((out/'optimization').glob('*.json'));assert len(files)==1
        history=json.loads(files[0].read_text());assert len(history['records'])==220
        counts={stage:sum(r['stage']==stage for r in history['records']) for stage in ('logits','temperature','hard')}
        assert list(counts.values())==[100,100,20]
        state=json.loads((out/'run.json').read_text());assert str(files[0].relative_to(out)) in state['artifacts']
        report[name+'_history']=dict(iterations=220,stage_counts=counts,hashed=True)
    checked=subprocess.run(command+['--dry-run','--verify-runtime'],env=env,text=True,capture_output=True,timeout=120)
    assert checked.returncode==0 and 'todo=0 (done=1)' in checked.stdout
    report['scheduler_resume']=checked.stdout
    report['gpu_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    report['status']='passed'
finally:
    write_json(root/'verification.json',report)
print(json.dumps(report,indent=2))
