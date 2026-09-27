"""Serial cold/warm artifact benchmark with sampled host/GPU memory and exact replay."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.result_io import write_json
from scripts.gpu_smoke import replay_check
from run_state import completed_run

CASES={
 'short':('ifnar.pdb','44-46,73-76,87-91','1qys1.pdb','26-40,58-71'),
 'medium':('pd-l1-1.pdb','30-34,50-54,69-76','5aao1.pdb','15-26,48-58,81-91,116-124'),
 'long':('egfr.pdb','106-107,128-131,138-153','5bvl_af2.pdb','42-65,68-87,89-113'),
}


def measured(command,env,log_path,timeout=1200):
    began=time.monotonic();peak_rss=peak_gpu=0;readings=0
    with Path(log_path).open('w') as log:
        p=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=-2,start_new_session=True)
        try:
            while p.poll() is None:
                if time.monotonic()-began>timeout:raise TimeoutError('Benchmark timeout')
                table=[tuple(map(int,row.split())) for row in subprocess.check_output(
                    ['ps','-e','-o','pid=,ppid=,rss='],text=True).splitlines() if row.strip()]
                owned={p.pid}
                while True:
                    children={pid for pid,ppid,rss in table if ppid in owned}
                    if children <= owned:break
                    owned |= children
                rss=sum(rss*1024 for pid,ppid,rss in table if pid in owned)
                peak_rss=max(peak_rss,rss)
                gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
                peak_gpu=max(peak_gpu,int(gpu.strip())*1024**2);readings+=1
                time.sleep(1)
            code=p.returncode
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    return dict(returncode=code,seconds=time.monotonic()-began,peak_process_tree_rss_bytes=peak_rss,
                peak_device_used_bytes=peak_gpu,samples=readings)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True,type=Path);p.add_argument('--data-dir',required=True);a=p.parse_args()
    root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',runs=[],parity={},scope='3 design iterations, 2 candidates, 2 validation models, 3 recycles; performance screen')
    write_json(root/'report.json',report)
    try:
        for case,(target,hot,template,bhot) in CASES.items():
            for phase in ('cold','warm'):
                for mode in ('full','compact'):
                    name=f'{case}_{phase}_{mode}';out=root/name
                    cmd=[sys.executable,str(ROOT/'FoldCraft.py'),'--output_folder',str(out),'--data_dir',a.data_dir,
                        '--target_template',str(ROOT/'examples/targets'/target),'--target_hotspots',hot,
                        '--binder_template',str(ROOT/'examples/templates'/template),'--binder_hotspots',bhot,
                        '--design_stages','1,1,1','--num_designs','1','--mpnn_samples','2','--seed','20261001',
                        '--validation_models','model_1_ptm,model_2_ptm','--artifact_mode',mode]
                    # Separate fresh mode caches give a fair cold comparison; phase 2 reuses each cache.
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',MPLBACKEND='Agg',
                             JAX_COMPILATION_CACHE_DIR=str(root/f'cache-{case}-{mode}'),PYTHONUNBUFFERED='1')
                    record=dict(name=name,case=case,phase=phase,mode=mode,command=cmd,**measured(cmd,env,root/f'{name}.log'))
                    if record['returncode'] or not completed_run(out):raise RuntimeError(f'Invalid performance run: {name}')
                    record['pickle_bytes']=sum(x.stat().st_size for x in out.rglob('*.pickle'))
                    record['candidate_pickle_bytes']=sum(x.stat().st_size for x in (out/'designs').glob('*.pickle'))
                    record['total_bytes']=sum(x.stat().st_size for x in out.rglob('*') if x.is_file())
                    report['runs'].append(record);write_json(root/'report.json',report);print(json.dumps(record),flush=True)
            pair=root/f'parity-{case}';pair.mkdir()
            (pair/'fixed').symlink_to(root/f'{case}_cold_full',target_is_directory=True)
            (pair/'replay').symlink_to(root/f'{case}_warm_compact',target_is_directory=True)
            report['parity'][case]=replay_check(pair,tolerance=0)
        report['status']='passed'
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:write_json(root/'report.json',report)

if __name__=='__main__':main()
