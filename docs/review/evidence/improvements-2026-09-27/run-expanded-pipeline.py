"""Run the frozen authorized plan serially after the performance gate."""
import json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'improvements-repo'))
from baseline.result_io import write_json
from scripts.gpu_smoke import run_owned
base=Path('/home/bizon/projects/foldcraft-gpu-20260927');repo=base/'improvements-repo';root=base/'expanded-01'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',MPLBACKEND='Agg',
         JAX_COMPILATION_CACHE_DIR=str(base/'jax-cache'))
report=dict(status='waiting_for_performance',stages=[])
status=base/'improvements-pipeline.json';write_json(status,report)
try:
    deadline=time.monotonic()+3600
    while True:
        profile=json.loads((base/'performance-02/report.json').read_text())
        if profile['status']=='passed':break
        if profile['status']=='failed':raise RuntimeError('Performance parity failed; generation not started')
        if time.monotonic()>deadline:raise TimeoutError('Performance gate wait exceeded one hour')
        time.sleep(5)
    steps=[
        ('generation',[sys.executable,str(repo/'scripts/expanded_benchmark.py'),'--output-root',str(root),'--data-dir',str(base/'weights'),'--execute'],5*3600+60),
        ('pool',[sys.executable,str(repo/'scripts/analyze_expanded.py'),str(root),'--prepare'],300),
        ('esmfold',[str(base/'oracle-venv/bin/python'),str(repo/'baseline/score_esmfold.py'),str(root/'pool'),'--model-dir','/home/bizon/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a'],1800),
        ('boltz',[str(base/'oracle-venv/bin/python'),str(repo/'scripts/score_boltz_paired.py'),str(root/'pool'),'--workdir',str(root/'boltz-work')],4*3600),
        ('analysis',[sys.executable,str(repo/'scripts/analyze_expanded.py'),str(root)],300)]
    for name,command,timeout in steps:
        report['status']=name;write_json(status,report);start=time.monotonic()
        with (base/f'expanded-01-{name}.log').open('w') as log:
            code=run_owned(command,env=env,timeout=timeout,stdout=log,stderr=-2)
        record=dict(stage=name,returncode=code,seconds=time.monotonic()-start,command=command)
        report['stages'].append(record);write_json(status,report);print(json.dumps(record),flush=True)
        if code:raise RuntimeError(f'{name} failed')
    report['status']='passed'
except BaseException as exc:
    report.update(status='failed',error=str(exc));raise
finally:write_json(status,report)
