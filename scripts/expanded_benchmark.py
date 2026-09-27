"""Frozen expanded temperature/ranking benchmark; no assay-accuracy claims."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.experiment_protocol import arm_order
from baseline.result_io import write_json
from run_state import completed_run,sha256
from FoldCraft import parse_args
from input_validation import preflight

TARGETS={
 'pdl1':dict(file='pd-l1-1.pdb',hotspots='30-34,50-54,69-76',family='immune_checkpoint',split='development'),
 'pd1':dict(file='pd-1.pdb',hotspots='44-48,81-88,103-106',family='immune_checkpoint',split='development'),
 'egfr':dict(file='egfr.pdb',hotspots='106-107,128-131,138-153',family='EGFR',split='development'),
 'ifnar':dict(file='ifnar.pdb',hotspots='44-46,73-76,87-91',family='IFNAR',split='holdout'),
}
SCAFFOLDS={
 'top7':dict(file='1qys1.pdb',hotspots='26-40,58-71'),
 'barrel':dict(file='6d0t1.pdb',hotspots='12-15,25-31,40-43'),
 'ankyrin':dict(file='5aao1.pdb',hotspots='15-26,48-58,81-91,116-124'),
}
SEEDS=[20261001,20261002]
ARMS={'baseline':.1,'mpnn_temp_02':.2}


def protocol(output,weights,python=sys.executable):
    output=Path(output).resolve();weights=Path(weights).resolve();jobs=[]
    # All development runs precede the held-out target; no adaptive choices.
    for target,t in TARGETS.items():
        for scaffold,s in SCAFFOLDS.items():
            case=f'{target}_{scaffold}'
            for seed in SEEDS:
                for arm in arm_order(ARMS,seed,case):
                    name=f'{case}_{seed}_{arm}'
                    cmd=[python,str(ROOT/'FoldCraft.py'),'--output_folder',str(output/name),
                         '--data_dir',str(weights),'--seed',str(seed),'--num_designs','1',
                         '--design_stages','100,100,20','--mpnn_samples','2','--artifact_mode','compact',
                         '--validation_models','model_1_ptm,model_2_ptm','--validation_recycles','3',
                         '--binder_template',str(ROOT/'examples/templates'/s['file']),'--binder_hotspots',s['hotspots'],
                         '--target_template',str(ROOT/'examples/targets'/t['file']),'--target_hotspots',t['hotspots'],
                         '--mpnn_sampling_temp',str(ARMS[arm])]
                    preflight(parse_args(cmd[2:]))
                    jobs.append(dict(name=name,case=case,target=target,scaffold=scaffold,family=t['family'],split=t['split'],seed=seed,arm=arm,command=cmd))
    inputs=[ROOT/'examples/targets'/x['file'] for x in TARGETS.values()]+[ROOT/'examples/templates'/x['file'] for x in SCAFFOLDS.values()]
    return dict(schema=1,baseline_ref='validated-baseline-2026-09-27',baseline_commit='c5e3f788f483407f0568f66071845a92076008bb',
        targets=TARGETS,scaffolds=SCAFFOLDS,seeds=SEEDS,arms=ARMS,jobs=jobs,
        inputs={str(x):sha256(x) for x in inputs},
        code={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'FoldCraft.py',ROOT/'model_validation.py',ROOT/'prediction_artifacts.py',ROOT/'scripts/expanded_benchmark.py',ROOT/'scripts/analyze_expanded.py',ROOT/'scripts/score_boltz_paired.py',ROOT/'EXPANDED_BENCHMARK_PLAN.md']},
        analysis_unit='paired case/seed trajectory; average MPNN siblings before comparison',
        primary_proxy='ESMFold confidence >=70 and ESMFold/design RMSD <=3.5 A and mean repeated Boltz ipTM >=0.6',
        boltz_repeats=2,boltz_seed_protocol='sha256-stage-v1(case,trajectory seed,draw,repeat), excludes arm',
        ranking=dict(quota=1,baseline='highest AF2 ipTM',candidate='fewest violations of template RMSD<=3.5, clashes<=5, epitope coverage>=0.1, then highest AF2 ipTM'),
        promotion=dict(primary_gain_pp=10,confidence='paired 95% bootstrap lower bound >0',
                       max_template_rmsd_regression_A=.5,max_clash_regression=1.,min_unseen_target_families=2,
                       binding_accuracy_requires_assays=True),
        holdout_limit='IFNAR is the only fully untouched family; PD1/PDL1 grouped conservatively; no broad default promotion from one holdout family',
        max_generation_hours=5,concurrency=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True,type=Path);p.add_argument('--data-dir',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    root=a.output_root.resolve();plan=protocol(root,a.data_dir)
    if not a.execute:print(json.dumps(plan,indent=2));return
    root.mkdir(parents=True,exist_ok=False);write_json(root/'protocol.json',plan)
    report=dict(status='running',jobs=[],concurrency=2);write_json(root/'execution.json',report)
    began=time.monotonic();running=[];queue=list(plan['jobs'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    try:
        while queue or running:
            if time.monotonic()-began>plan['max_generation_hours']*3600:raise TimeoutError('Campaign generation budget exceeded')
            # Never overlap development and holdout phases.
            while queue and len(running)<2 and (not running or queue[0]['split']==running[0][0]['split']):
                job=queue.pop(0);log=(root/(job['name']+'.log')).open('w')
                proc=subprocess.Popen(job['command'],env=env,cwd=ROOT,stdout=log,stderr=-2,start_new_session=True)
                running.append((job,proc,log,time.monotonic()))
            for item in list(running):
                job,proc,log,start=item
                if time.monotonic()-start>1800:raise TimeoutError('Single run exceeded 30 minutes')
                if proc.poll() is None:continue
                log.close();running.remove(item)
                valid=proc.returncode==0 and completed_run(root/job['name'])
                record=dict(name=job['name'],seconds=time.monotonic()-start,returncode=proc.returncode,valid=valid)
                report['jobs'].append(record);write_json(root/'execution.json',report);print(json.dumps(record),flush=True)
                if not valid:raise RuntimeError('Invalid campaign run: '+job['name'])
            time.sleep(1)
        for target in TARGETS:
            for scaffold in SCAFFOLDS:
                for seed in SEEDS:
                    paths=[root/f'{target}_{scaffold}_{seed}_{arm}'/'traj/traj_1.pdb' for arm in ARMS]
                    if sha256(paths[0])!=sha256(paths[1]):raise ValueError('Temperature arms changed the shared backbone')
        report['backbone_pairs_verified']=24;report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        from run_watchdog import terminate_group
        for job,proc,log,start in running:
            terminate_group(proc,grace_seconds=15);log.close()
        report['allocated_gpu_seconds']=time.monotonic()-began;write_json(root/'execution.json',report)

if __name__=='__main__':main()
