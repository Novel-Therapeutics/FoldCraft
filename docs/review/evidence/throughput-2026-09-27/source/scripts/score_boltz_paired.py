"""Two repeated Boltz scores with arm-independent paired RNG streams."""
import argparse,math,sys
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
from baseline.score_boltz2 import chain_sequences,run_boltz2
from baseline.score_cache import ScoreSession, digest
from baseline.boltz_worker import BoltzWorkerPool
from run_state import stage_seed


def paired_seed(row,repeat):
    return stage_seed(int(row['seed']),'boltz-paired-v1',row['case'],row['trajectory'],int(row['draw']),repeat)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('pool',type=Path);p.add_argument('--cache',type=Path,default=Path.home()/'.boltz');p.add_argument('--workdir',required=True,type=Path);p.add_argument('--execution-backend',choices=('subprocess','persistent'),default='persistent');a=p.parse_args()
    from boltz.data.const import canonical_tokens
    checkpoint=a.cache/'boltz2_conf.ckpt'
    inputs={'checkpoint':checkpoint,**{f'molecule:{n}':a.cache/'mols'/f'{n}.pkl' for n in canonical_tokens}}
    columns=['boltz2_iptm_r0','boltz2_iptm_r1','boltz2_iptm']
    with ScoreSession(a.pool/'results.csv',__file__,columns,
                      dict(repeats=2,seed_protocol='boltz-paired-v1',msa=False,kernels=False,execution_backend=a.execution_backend,
                           worker_source=digest(ROOT/'baseline/boltz_worker.py'),scorer_source=digest(ROOT/'baseline/score_boltz2.py')),
                      extra_inputs=[a.pool/'pool_manifest.json'],model_inputs=inputs) as session, (BoltzWorkerPool(2) if a.execution_backend=='persistent' else nullcontext()) as worker_pool:
        frame=session.prepare(pd.read_csv(a.pool/'results.csv'));session.publish(frame)
        tasks=[]
        # Parse structures serially; Bio.PDB's shared parser is not thread-safe.
        for idx,row in frame.iterrows():
            if pd.notna(row['boltz2_iptm']):continue
            target,binder=chain_sequences(a.pool/'designs'/f'{row["name"]}.pdb')
            tasks.append((idx,row,target,binder))
        def evaluate(task):
            idx,row,target,binder=task;values=[];seeds=[]
            for repeat in range(2):
                seed=paired_seed(row,repeat);seeds.append(seed)
                value,*_=run_boltz2(target,binder,str(a.workdir/row['name']/str(repeat)),
                    use_msa=False,diffusion_samples=1,flash_attn=False,checkpoint=checkpoint,
                    cache=a.cache,seed=seed,no_kernels=True,worker_pool=worker_pool)
                if value is None or not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid Boltz ipTM')
                values.append(value)
            return idx,row['name'],values,seeds
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(evaluate,task) for task in tasks]
            for completed,future in enumerate(as_completed(futures),1):
                idx,name,values,seeds=future.result()
                frame.loc[idx,columns]=[*values,sum(values)/2]
                session.annotate(name,status='complete',seeds=seeds)
                session.publish(frame)
                print(f'{completed}/{len(tasks)} {name}: {sum(values)/2:.4f}',flush=True)

if __name__=='__main__':main()
