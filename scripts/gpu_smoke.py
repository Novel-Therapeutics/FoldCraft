"""Print a GPU smoke-test plan; use --execute only once GPU access is authorized.

No SSH, package installation, weight downloads or background polling. Execution
runs serially on the explicitly selected GPU and records every command/outcome.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.result_io import write_json
from run_state import completed_run


def plan(output_root, data_dir, python=sys.executable, recycles=0):
    output_root=Path(output_root).resolve()
    base=[python,str(ROOT/'FoldCraft.py'),'--data_dir',str(Path(data_dir).resolve()),
          '--validation_models','model_1_ptm,model_2_ptm','--validation_recycles',str(recycles),
          '--seed','20260927','--design_stages','1,1,1','--mpnn_samples','2','--num_designs','1']
    cases=[]
    for name in ('fixed','replay','bounded_sample','vhh','long_target'):
        target='egfr.pdb' if name=='long_target' else 'pd-l1-1.pdb'
        hotspots='106-107,128-131,138-153' if name=='long_target' else '30-34,50-54,69-76'
        command=base+['--output_folder',str(output_root/name),
                      '--target_template',str(ROOT/'examples/targets'/target),'--target_hotspots',hotspots]
        if name in ('vhh','long_target'):command+=['--vhh']
        else:command+=['--binder_template',str(ROOT/'examples/templates/1qys1.pdb'),
                      '--binder_hotspots','26-40,58-71']
        if name=='bounded_sample':command+=['--sample','--target_success','1','--max_trajectories','1']
        cases.append(dict(name=name,command=command))
    return cases


def replay_check(output_root, tolerance=1e-5):
    import numpy as np
    import pickle
    root=Path(output_root)
    first=sorted((root/'fixed/designs').glob('*.validation.json'))
    second=sorted((root/'replay/designs').glob('*.validation.json'))
    if not first or [p.name for p in first] != [p.name for p in second]:
        raise ValueError('Replay candidate identities differ')
    max_pae_delta=0.
    max_coordinate_delta=0.
    for a,b in zip(first,second):
        left,right=json.loads(a.read_text()),json.loads(b.read_text())
        if left['sequence']!=right['sequence'] or left['model_order']!=right['model_order']:
            raise ValueError('Replay sequences or models differ')
        for model in left['model_order']:
            x,y=left['models'][model],right['models'][model]
            if x['seed']!=y['seed'] or x['passed']!=y['passed']:
                raise ValueError('Replay seeds or decisions differ')
            if not all(abs(x['metrics'][k]-y['metrics'][k])<=tolerance for k in x['metrics']):
                raise ValueError('Replay metrics differ beyond tolerance')
        for pk in (root/'fixed/designs').glob(a.name.replace('.validation.json','')+'*.pickle'):
            with open(pk,'rb') as f: x=np.asarray(pickle.load(f)['pae'])
            with open(root/'replay/designs'/pk.name,'rb') as f: y=np.asarray(pickle.load(f)['pae'])
            if x.shape!=y.shape or not np.allclose(x,y,rtol=0,atol=tolerance):
                raise ValueError('Replay PAE differs beyond tolerance')
            max_pae_delta=max(max_pae_delta,float(np.max(abs(x-y))))
            # Check unrounded coordinates as well as confidence predictions.
            with open(pk,'rb') as f: x=np.asarray(pickle.load(f)['atom_positions'])
            with open(root/'replay/designs'/pk.name,'rb') as f: y=np.asarray(pickle.load(f)['atom_positions'])
            if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.allclose(x,y,rtol=0,atol=tolerance):
                raise ValueError('Replay coordinates differ beyond tolerance')
            max_coordinate_delta=max(max_coordinate_delta,float(np.max(abs(x-y))))
    return dict(status='passed',absolute_tolerance=tolerance,max_pae_delta=max_pae_delta,
                max_coordinate_delta=max_coordinate_delta)


def run_owned(command, *, env, timeout, stdout=None, stderr=None):
    """A smoke-test timeout/interrupt terminates only this invocation's children."""
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            try:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            except ProcessLookupError:
                pass
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--gpu',default='0')
    parser.add_argument('--recycles',type=int,default=0)
    parser.add_argument('--timeout-minutes',type=int,default=30)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.recycles<0 or args.timeout_minutes<1:
        parser.error('Recycles must be nonnegative and timeout must be positive')
    args.data_dir=str(Path(args.data_dir).resolve())
    cases=plan(args.output_root,args.data_dir,recycles=args.recycles)
    if not args.execute:
        print(json.dumps(dict(status='plan_only',gpu=args.gpu,cases=cases),indent=2))
        return
    output=Path(args.output_root).resolve()
    output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,XLA_PYTHON_CLIENT_PREALLOCATE='false')
    report=dict(status='running',cases=[])
    write_json(output/'smoke.json',report)
    try:
        rc=run_owned([sys.executable,str(ROOT/'scripts/gpu_probe.py'),'--data-dir',args.data_dir,
                      '--report',str(output/'runtime.json')],env=env,timeout=args.timeout_minutes*60)
        if rc!=0:raise RuntimeError('GPU runtime probe failed; inspect runtime.json')
        for case in cases:
            start=time.monotonic()
            with open(output/(case['name']+'.log'),'w') as log:
                rc=run_owned(case['command'],env=env,timeout=args.timeout_minutes*60,stdout=log,stderr=subprocess.STDOUT)
            folder=output/case['name']
            state_path=folder/'run.json'
            state=json.loads(state_path.read_text()) if state_path.exists() else {'status':'not_started'}
            valid=(rc==0 and completed_run(folder))
            if case['name']=='bounded_sample' and rc!=0 and state['status']=='exhausted':
                valid=(state['attempted_trajectories']==1 and (folder/'results.csv.partial').exists())
            record=dict(case,seconds=time.monotonic()-start,returncode=rc,
                        run_status=state['status'],valid=valid)
            report['cases'].append(record);write_json(output/'smoke.json',report)
            if not valid:raise RuntimeError(f"Smoke case failed: {case['name']}")
        report['replay']=replay_check(output)
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'smoke.json',report)


if __name__=='__main__':main()
