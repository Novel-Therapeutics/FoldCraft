"""Real CLI startup failure and process supervision; no simulated science results."""
import json,os,subprocess,sys,time
from pathlib import Path
from baseline.result_io import write_json
base=Path('/home/bizon/projects/foldcraft-gpu-20260927');repo=base/'closure-repo'
root=base/'experimental-ops-01';root.mkdir(exist_ok=False)
report={'status':'running'}
command=[sys.executable,str(repo/'test/FoldCraft_binder.py'),'--sample','--target_template',
         str(repo/'examples/targets/pd-l1-1.pdb'),'--target_hotspots','30-34',
         '--binder_len','30-35','--max_trajectories','1','--target_success','1',
         '--data_dir',str(base/'weights'),'--seed','42']
try:
    output=root/'missing-dependencies'
    result=subprocess.run(command+['--output_folder',str(output)],text=True,capture_output=True,timeout=60)
    (root/'dependency-failure.log').write_text(result.stdout+result.stderr)
    assert result.returncode!=0
    state=json.loads((output/'experimental_run.json').read_text())
    assert state['status']=='failed' and 'requires BindCraft' in state['error']
    report['dependency_failure_recorded']=True
    shadow=root/'shadow';(shadow/'jax').mkdir(parents=True)
    marker=root/'worker-entered-import'
    (shadow/'jax/__init__.py').write_text(f'from pathlib import Path\nimport time\nPath({str(marker)!r}).write_text("entered")\ntime.sleep(300)\n')
    output=root/'timed-out'
    began=time.monotonic()
    result=subprocess.run(command+['--output_folder',str(output),'--timeout_minutes','0.05'],
        env=dict(os.environ,PYTHONPATH=str(shadow)),capture_output=True,text=True,timeout=25)
    elapsed=time.monotonic()-began
    (root/'timeout.log').write_text(result.stdout+result.stderr)
    state=json.loads((output/'experimental_run.json').read_text())
    assert marker.is_file() and result.returncode==124 and state['status']=='timed_out'
    assert not (output/'results.csv').exists()
    report['timeout']=dict(seconds=elapsed,returncode=result.returncode,status=state['status'])
    report['gpu_processes_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    report['status']='passed'
finally:
    write_json(root/'verification.json',report)
print(json.dumps(report,indent=2))
