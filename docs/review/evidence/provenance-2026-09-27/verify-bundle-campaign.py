"""Real scheduler resume test; mutate only an isolated copy of MPNN weights."""
import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from baseline.result_io import write_json
from inference_bundle import colabdesign_root
from scripts.gpu_smoke import run_owned,replay_check
from run_state import completed_run,sha256

base=Path('/home/bizon/projects/foldcraft-gpu-20260927')
repo=base/'bundle-repo';root=base/'bundle-campaign-01';root.mkdir(exist_ok=False)
overlay=root/'worker-package'
shutil.copytree(colabdesign_root(),overlay/'colabdesign',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
env=dict(os.environ,PYTHONPATH=str(overlay)+os.pathsep+str(repo),CUDA_VISIBLE_DEVICES='0',
         XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(base/'jax-cache'))
config=root/'config.tsv'
config.write_text('fold\ttemplate\ttarget\ttarget_hotspots\tbinder_hotspots\ttotal_traj\tmem_gb\n'+
    f'audit\t{repo}/examples/templates/1qys1.pdb\t{repo}/examples/targets/pd-l1-1.pdb\t30-34,50-54,69-76\t26-40,58-71\t1\t12\n')
command=[sys.executable,str(repo/'baseline/scheduler.py'),str(config),'--repo',str(repo),
         '--repro',str(root/'runs'),'--python',sys.executable,'--data-dir',str(base/'weights'),
         '--gpus','0','--chunk-traj','1','--timeout-minutes','10']
report=dict(status='running',command=command)
def plan(label):
    result=subprocess.run(command+['--dry-run','--verify-runtime'],env=env,capture_output=True,text=True,timeout=120)
    (root/(label+'.log')).write_text(result.stdout+result.stderr)
    assert result.returncode==0
    return result.stdout
try:
    pair=root/'parity';pair.mkdir()
    (pair/'fixed').symlink_to(base/'bundle-smoke-01/fixed',target_is_directory=True)
    (pair/'replay').symlink_to(base/'smoke-r3-01/fixed',target_is_directory=True)
    report['prior_runtime_parity']=replay_check(pair)
    began=time.monotonic()
    with (root/'campaign.log').open('w') as log:
        code=run_owned(command,env=env,timeout=600,stdout=log,stderr=-2)
    report['generation_seconds']=time.monotonic()-began
    assert code==0
    chunk=root/'runs/audit__c0';merged=root/'runs/audit'
    assert completed_run(chunk) and completed_run(merged)
    state=json.loads((chunk/'run.json').read_text())
    job=json.loads(Path(str(chunk)+'.job.json').read_text())
    assert job['inference_bundle']==state['inference_bundle']['identity']
    assert json.loads((merged/'run.json').read_text())['chunk_inference_bundles'][str(chunk)]==job['inference_bundle']
    original_artifacts={str(p):sha256(p) for p in (chunk/'run.json',chunk/'results.csv',merged/'run.json',merged/'results.csv')}
    report['unchanged_resume']=plan('unchanged-resume');assert 'todo=0 (done=1)' in report['unchanged_resume']
    weight=Path(job['inference_bundle']['files']['mpnn:v_48_010']['path'])
    assert weight.is_relative_to(overlay)
    st=weight.stat()
    with weight.open('rb') as stream:original=stream.read(1)
    try:
        with weight.open('r+b') as stream:stream.write(bytes([original[0]^1]))
        os.utime(weight,ns=(st.st_atime_ns,st.st_mtime_ns))
        assert weight.stat().st_size==st.st_size and weight.stat().st_mtime_ns==st.st_mtime_ns
        report['changed_resume']=plan('changed-resume');assert 'todo=1 (done=0)' in report['changed_resume']
        rejected=subprocess.run(command,env=env,capture_output=True,text=True,timeout=120)
        (root/'changed-execution.log').write_text(rejected.stdout+rejected.stderr)
        assert rejected.returncode!=0 and 'does not match' in rejected.stderr
        assert all(sha256(path)==digest for path,digest in original_artifacts.items())
        report['changed_execution_rejected']=True
        report['existing_artifacts_unchanged']=True
    finally:
        with weight.open('r+b') as stream:stream.write(original)
        os.utime(weight,ns=(st.st_atime_ns,st.st_mtime_ns))
    report['restored_resume']=plan('restored-resume');assert 'todo=0 (done=1)' in report['restored_resume']
    report['gpu_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    report['status']='passed'
finally:
    write_json(root/'verification.json',report)
print(json.dumps(report,indent=2))
