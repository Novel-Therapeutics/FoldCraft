"""Prespecified paired structural pilot; prints the plan unless --execute is used."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.result_io import write_json
from baseline.experiment_protocol import arm_order
from run_state import completed_run,sha256
from scripts.gpu_smoke import run_owned
from FoldCraft import parse_args
from input_validation import preflight

CASES={
    'pdl1_top7':dict(target='pd-l1-1.pdb',hotspots='30-34,50-54,69-76',family='PD-L1'),
    'egfr_top7':dict(target='egfr.pdb',hotspots='106-107,128-131,138-153',family='EGFR'),
}
ARMS={
    'baseline':dict(loss_mode='legacy',mpnn_sampling_temp=.1),
    'normalized_pairs':dict(loss_mode='normalized_pairs',mpnn_sampling_temp=.1),
    'mpnn_temp_02':dict(loss_mode='legacy',mpnn_sampling_temp=.2),
}
SEEDS=[20260927,20260928]


def plan(output, weights, python=sys.executable):
    output=Path(output).resolve();weights=Path(weights).resolve();jobs=[]
    for case,config in CASES.items():
        for seed in SEEDS:
            for arm in arm_order(ARMS,seed,case):
                name=f'{case}_{seed}_{arm}'
                command=[python,str(ROOT/'FoldCraft.py'),'--output_folder',str(output/name),
                    '--data_dir',str(weights),'--seed',str(seed),'--num_designs','1',
                    '--design_stages','100,100,20','--mpnn_samples','2',
                    '--validation_models','model_1_ptm,model_2_ptm','--validation_recycles','3',
                    '--binder_template',str(ROOT/'examples/templates/1qys1.pdb'),'--binder_hotspots','26-40,58-71',
                    '--target_template',str(ROOT/'examples/targets'/config['target']),'--target_hotspots',config['hotspots']]
                for key,value in ARMS[arm].items():command += ['--'+key,str(value)]
                jobs.append(dict(name=name,case=case,family=config['family'],seed=seed,arm=arm,command=command))
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args();jobs=plan(args.output_root,args.data_dir)
    for job in jobs:preflight(parse_args(job['command'][2:]))
    protocol=dict(kind='exploratory_structural_pilot',baseline_ref='corrected-baseline-2026-09-27',baseline_commit='d29ba3b60f41a4c6e192c6716fbb9a7f788b6a07',cases=CASES,arms=ARMS,seeds=SEEDS,
                  jobs=jobs,design_stages=[100,100,20],mpnn_samples=2,
                  analysis_unit='case/seed trajectory; MPNN siblings are not independent replicates',
                  primary_endpoints=['ESMFold fold agreement','Boltz ipTM','intended epitope coverage','template CA RMSD'],
                  selection_comparison='model_1_pass versus all_models_pass on the identical frozen candidate pool',
                  promotion='none: two target families and four paired blocks are an underpowered pilot')
    protocol['inputs']={str(path):sha256(path) for path in [ROOT/'examples/templates/1qys1.pdb', *[ROOT/'examples/targets'/c['target'] for c in CASES.values()]]}
    protocol['code']={name:sha256(ROOT/name) for name in ('FoldCraft.py','design_objective.py','model_validation.py','sequence_design.py','scripts/paired_pilot.py','scripts/analyze_pilot.py')}
    if not args.execute:
        print(json.dumps(protocol,indent=2));return
    output=Path(args.output_root).resolve();output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',protocol)
    report=dict(status='running',jobs=[]);write_json(output/'execution.json',report)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    start=time.monotonic();deadline=start+90*60
    try:
        rc=run_owned([sys.executable,str(ROOT/'scripts/gpu_probe.py'),'--data-dir',args.data_dir,
                      '--report',str(output/'runtime.json')],env=env,timeout=120)
        if rc:raise RuntimeError('GPU runtime probe failed')
        for job in jobs:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Pilot exceeded 90-minute total budget')
            began=time.monotonic()
            telemetry=open(output/(job['name']+'.gpu.csv'),'w')
            monitor=subprocess.Popen(['nvidia-smi','--id=0','--query-gpu=timestamp,memory.used,utilization.gpu',
                                      '--format=csv','-l','2'],stdout=telemetry,stderr=subprocess.DEVNULL)
            try:
                with open(output/(job['name']+'.log'),'w') as log:
                    rc=run_owned(job['command'],env=env,timeout=min(20*60,remaining),stdout=log,stderr=subprocess.STDOUT)
            finally:
                monitor.terminate();monitor.wait(timeout=10);telemetry.close()
            valid=rc==0 and completed_run(output/job['name'])
            record=dict(name=job['name'],case=job['case'],seed=job['seed'],arm=job['arm'],
                        seconds=time.monotonic()-began,returncode=rc,valid=valid)
            report['jobs'].append(record);write_json(output/'execution.json',report)
            print(json.dumps(record),flush=True)
            if not valid:raise RuntimeError(f'Pilot job failed: {job["name"]}')
        # Temperature is the only changed factor: generation must replay exactly.
        for case in CASES:
            for seed in SEEDS:
                a=output/f'{case}_{seed}_baseline/traj/traj_1.pdb'
                b=output/f'{case}_{seed}_mpnn_temp_02/traj/traj_1.pdb'
                if sha256(a)!=sha256(b):raise ValueError('MPNN temperature arms did not share an identical generated backbone')
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report['total_seconds']=time.monotonic()-start;write_json(output/'execution.json',report)


if __name__=='__main__':main()
