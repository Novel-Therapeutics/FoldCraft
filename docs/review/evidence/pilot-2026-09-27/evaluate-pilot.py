"""Owned, serial evaluation of the prespecified frozen pilot pool."""
import json,os,sys,time,subprocess
from pathlib import Path
from baseline.result_io import write_json
from scripts.gpu_smoke import run_owned
base=Path('/home/bizon/projects/foldcraft-gpu-20260927')
root=base/'pilot-01';pool=root/'pool'
assert json.loads((root/'execution.json').read_text())['status']=='passed'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
report=dict(status='running',stages=[]);write_json(root/'evaluation.json',report)
steps=[
 ('pool',[sys.executable,'scripts/analyze_pilot.py',str(root),'--build-pool'],120),
 ('esmfold',[str(base/'oracle-venv/bin/python'),'baseline/score_esmfold.py',str(pool),'--model-dir','/home/bizon/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots/75a3841ee059df2bf4d56688166c8fb459ddd97a'],1800),
 ('boltz',[str(base/'oracle-venv/bin/python'),'baseline/score_boltz2.py',str(pool),'--no-msa','--no-kernels','--seed','20260927','--workdir',str(root/'boltz-work')],2700),
 ('openmm',[str(base/'openmm-venv/bin/python'),'baseline/score_openmm.py',str(pool),'--platform','CUDA','--min-iters','500','--seed','20260927'],1800),
 ('analysis',[sys.executable,'scripts/analyze_pilot.py',str(root)],120)]
try:
 for name,command,timeout in steps:
  start=time.monotonic()
  with (root/(name+'.evaluation.log')).open('w') as log:
   code=run_owned(command,env=env,timeout=timeout,stdout=log,stderr=-2)
  stage=dict(name=name,command=command,returncode=code,seconds=time.monotonic()-start)
  report['stages'].append(stage);write_json(root/'evaluation.json',report);print(json.dumps(stage),flush=True)
  if code and name!='openmm':raise RuntimeError(f'{name} evaluation failed')
 report['status']='complete_with_unknowns' if any(s['returncode'] for s in report['stages']) else 'complete'
except BaseException as exc:
 report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
finally:write_json(root/'evaluation.json',report)
