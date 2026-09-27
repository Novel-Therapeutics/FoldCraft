import json,os,subprocess,sys,time
from pathlib import Path
from scripts.gpu_smoke import plan,run_owned,replay_check
from run_state import completed_run
from baseline.result_io import write_json
base=Path('/home/bizon/projects/foldcraft-gpu-20260927')
root=base/'reliability-final-01';root.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',
         JAX_COMPILATION_CACHE_DIR=str(base/'jax-cache'))
report=dict(status='running',cases=[])
try:
    command=plan(root,base/'weights',recycles=3)[0]['command']+['--timeout_minutes','5']
    with (root/'fixed.log').open('w') as log:
        code=run_owned(command,env=env,timeout=180,stdout=log,stderr=-2)
    assert code==0 and completed_run(root/'fixed')
    attempts=json.loads((root/'fixed/attempts.json').read_text())
    for row in attempts:
        if row['candidate']:
            receipt=json.loads((root/'fixed/designs'/(row['candidate']+'.validation.json')).read_text())
            assert row['accepted']==receipt['all_models_pass']
    report['cases'].append(dict(name='fixed',command=command,returncode=code,attempt_decisions_verified=True))
    (root/'replay').symlink_to(base/'smoke-r3-01/fixed',target_is_directory=True)
    report['replay_against_previous_runtime']=replay_check(root)
    command=list(command)
    command[command.index('--output_folder')+1]=str(root/'timeout')
    command[command.index('--timeout_minutes')+1]='0.2'
    command[command.index('--design_stages')+1]='100,100,20'
    with (root/'timeout.gpu.csv').open('w') as telemetry:
        monitor=subprocess.Popen(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv','-l','1'],stdout=telemetry)
        try:
            started=time.monotonic()
            with (root/'timeout.log').open('w') as log:
                code=run_owned(command,env=env,timeout=60,stdout=log,stderr=-2)
            elapsed=time.monotonic()-started
        finally:
            monitor.terminate();monitor.wait(timeout=5)
    state=json.loads((root/'timeout/run.json').read_text())
    assert code==124 and state['status']=='timed_out' and not completed_run(root/'timeout')
    assert elapsed<30
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    assert 'runtime-clean-01' not in gpu
    report['cases'].append(dict(name='timeout',command=command,returncode=code,seconds=elapsed,run_status=state['status'],remaining_gpu_processes=gpu))
    report['status']='passed'
finally:
    write_json(root/'verification.json',report)
print(json.dumps(report,indent=2))
