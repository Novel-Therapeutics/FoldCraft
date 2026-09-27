"""Validate and collect the authorized campaign after its owned GPU stages finish."""
import json,os,shutil,sys,time
from pathlib import Path
base=Path('/home/bizon/projects/foldcraft-gpu-20260927')
repo=base/'final-check-repo';root=base/'expanded-01'
sys.path.insert(0,str(repo))
from baseline.result_io import write_json
from scripts.gpu_smoke import run_owned,replay_check
from scripts.audit_expanded import audit as audit_predictions
from scripts.audit_evaluators import audit as audit_evaluators
from scripts.export_benchmark_evidence import export
from run_state import completed_run,sha256
state=dict(status='waiting_for_generation',stages=[])
status=base/'improvements-finalization.json'
def publish():write_json(status,state)
def run(name,command,timeout):
    state['status']=name;publish();began=time.monotonic()
    with (base/f'improvements-{name}.log').open('w') as log:
        code=run_owned(command,env=env,timeout=timeout,stdout=log,stderr=-2)
    record=dict(stage=name,command=command,returncode=code,seconds=time.monotonic()-began)
    state['stages'].append(record);publish();print(json.dumps(record),flush=True)
    if code:raise RuntimeError(name+' failed')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',MPLBACKEND='Agg',JAX_COMPILATION_CACHE_DIR=str(base/'performance-02/cache-short-full'))
publish()
try:
    deadline=time.monotonic()+8*3600
    audited=False
    while True:
        pipeline=json.loads((base/'improvements-pipeline.json').read_text())
        if pipeline['status']=='failed':raise RuntimeError('Main campaign failed: '+str(pipeline.get('error')))
        execution=json.loads((root/'execution.json').read_text())
        if execution['status']=='passed' and not audited:
            state['prediction_audit']=audit_predictions(root);audited=True
            state['status']='waiting_for_evaluation';publish();print(json.dumps(state['prediction_audit']),flush=True)
        if pipeline['status']=='passed':break
        if time.monotonic()>deadline:raise TimeoutError('Campaign exceeded finalization wait budget')
        time.sleep(15)
    state['evaluator_audit']=audit_evaluators(root);publish()
    run('runtime-check',[str(base/'oracle-venv/bin/python'),str(repo/'scripts/evaluator_runtime.py'),str(base/'evaluator-runtime-after.json'),'--compare',str(base/'evaluator-runtime-before.json')],300)
    perf=json.loads((base/'performance-02/report.json').read_text())
    command=list(next(r['command'] for r in perf['runs'] if r['name']=='short_warm_full'))
    index=command.index('--artifact_mode');del command[index:index+2]
    output=base/'default-storage-01'
    command[command.index('--output_folder')+1]=str(output)
    command=[arg.replace(str(base/'improvements-repo'),str(repo)) for arg in command]
    run('default-storage',command,1200)
    assert completed_run(output)
    for receipt in (output/'designs').glob('*.validation.json'):
        assert json.loads(receipt.read_text())['artifact_mode']=='compact'
    comparison=base/'default-storage-parity-01';comparison.mkdir()
    (comparison/'fixed').symlink_to(base/'performance-02/short_cold_full',target_is_directory=True)
    (comparison/'replay').symlink_to(output,target_is_directory=True)
    state['default_storage_replay']=replay_check(comparison,tolerance=0);publish()
    run('reports',[sys.executable,str(repo/'scripts/write_benchmark_reports.py'),'--performance',str(base/'performance-02'),'--expanded',str(root),'--pipeline',str(base/'improvements-pipeline.json'),'--destination',str(repo)],300)
    evidence=base/'improvements-evidence-01';evidence.mkdir()
    for name,source in [('performance',base/'performance-02'),('expanded',root),('default-storage',output)]:export(source,evidence/name)
    for name in ('improvements-pipeline.json','evaluator-runtime-before.json','evaluator-runtime-after.json','improvements-final-cpu.log','run-expanded-pipeline.py','finalize-improvements.py'):
        shutil.copy2(base/name,evidence/name)
    shutil.copy2(root/'analyze_expanded.original.py',evidence/'expanded/analyze_expanded.original.py')
    for pattern in ('expanded-01-*.log','improvements-*.log'):
        for path in base.glob(pattern):
            if path.name=='improvements-finalization.log':continue
            target=evidence/path.name
            target.write_text('\n'.join(line.rstrip() for line in path.read_text(errors='replace').splitlines())+'\n')
    state['evidence']=str(evidence);state['status']='passed';publish()
    shutil.copy2(status,evidence/status.name)
    write_json(evidence/'evidence_manifest.json',{str(p.relative_to(evidence)):sha256(p) for p in evidence.rglob('*') if p.is_file()})
    print(json.dumps(state),flush=True)
except BaseException as exc:
    state.update(status='failed',error=str(exc));publish();raise
