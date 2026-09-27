import json, shutil, subprocess, sys, time, pstats
from pathlib import Path
BASE=Path('/home/bizon/projects/foldcraft-gpu-20260927')
ROOT=BASE/'throughput-repo';sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts.benchmark_boltz_workers import artifacts
from baseline.result_io import write_json
from baseline.score_cache import digest


def main():
    root=BASE/'throughput-validation-01';root.mkdir(exist_ok=False)
    source=BASE/'expanded-01/pool';frame=pd.read_csv(source/'results.csv')
    frame=frame[(frame.case=='ifnar_top7')&(frame.seed==20261001)&(frame.draw==0)]
    assert len(frame)==2
    report=dict(status='running',runs=[]);write_json(root/'report.json',report)
    try:
        for label,script,extra in [('single','baseline/score_boltz2.py',['--no-msa','--no-kernels']),('paired','scripts/score_boltz_paired.py',[])]:
            pool=root/label;pool.mkdir();(pool/'designs').mkdir()
            frame.to_csv(pool/'results.csv',index=False)
            shutil.copy2(source/'pool_manifest.json',pool/'pool_manifest.json')
            for name in frame.name:shutil.copy2(source/'designs'/f'{name}.pdb',pool/'designs'/f'{name}.pdb')
            out=root/(label+'-outputs')
            command=[sys.executable,str(ROOT/script),str(pool),'--workdir',str(out),*extra]
            if label=='single':command += ['--execution-backend','persistent']
            began=time.monotonic()
            with (root/(label+'.log')).open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=-2,check=True)
            result=pd.read_csv(pool/'results.csv');assert result.boltz2_iptm.notna().all()
            receipts=list(out.rglob('selection.json'));assert len(receipts)==(2 if label=='single' else 4)
            assert all(json.loads(p.read_text())['execution_backend']=='persistent' for p in receipts)
            hashes={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
            began_resume=time.monotonic()
            with (root/(label+'-resume.log')).open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=-2,check=True)
            assert hashes=={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
            pd.testing.assert_frame_equal(result,pd.read_csv(pool/'results.csv'))
            report['runs'].append(dict(cli=label,receipts=len(receipts),seconds=time.monotonic()-began,
                cache_resume_seconds=time.monotonic()-began_resume,cache_hit_exact=True,command=command))
            write_json(root/'report.json',report)
        # Diagnostic profile is deliberately separate from timed throughput batches.
        yml=next((root/'single-outputs').rglob('complex.yaml'))
        command=[sys.executable,'-m','cProfile','-o',str(root/'startup.pstats'),'-m','boltz.main','predict',str(yml),
            '--out_dir',str(root/'profile-output'),'--override','--diffusion_samples','1','--model','boltz2',
            '--seed','20261021','--checkpoint',str(Path.home()/'.boltz/boltz2_conf.ckpt'),
            '--cache',str(Path.home()/'.boltz'),'--no_kernels']
        with (root/'profile.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=-2,check=True)
        stats=pstats.Stats(str(root/'startup.pstats'))
        rows=[dict(file=k[0],line=k[1],function=k[2],calls=v[1],self_seconds=v[2],cumulative_seconds=v[3]) for k,v in stats.stats.items()]
        write_json(root/'profile-summary.json',dict(total_seconds=stats.total_tt,command=command,
            top_cumulative=sorted(rows,key=lambda x:-x['cumulative_seconds'])[:60],
            lifecycle=[r for r in rows if r['function'] in ('load_from_checkpoint','_load_from_checkpoint','_load_state','predict','__init__','_run') and ('boltz/model/models' in r['file'] or 'pytorch_lightning' in r['file'])]))
        report['status']='passed'
    except BaseException as exc:report.update(status='failed',error=str(exc));raise
    finally:write_json(root/'report.json',report)

if __name__=='__main__':main()
