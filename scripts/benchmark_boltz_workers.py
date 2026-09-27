"""Paired fresh-process vs persistent-process Boltz throughput benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from baseline.boltz_worker import BoltzWorkerPool
from baseline.score_boltz2 import run_boltz2
from baseline.result_io import write_json
from baseline.score_cache import digest
from scripts.evaluator_runtime import snapshot


def artifacts(root):
    receipts=list(root.rglob('selection.json'))
    if len(receipts)!=1:raise ValueError('Expected exactly one prediction receipt')
    receipt=receipts[0];base=receipt.parent;data=json.loads(receipt.read_text())
    conf=base/data['confidence_file'];structure=base/data['structure_file']
    if digest(conf)!=data['confidence_sha256'] or digest(structure)!=data['structure_sha256']:
        raise ValueError('Artifact hash mismatch')
    arrays={}
    for p in sorted(base.rglob('*.npz')):
        if 'predictions' not in p.parts:continue
        with np.load(p,allow_pickle=False) as archive:
            arrays[p.name]={k:dict(shape=list(archive[k].shape),dtype=str(archive[k].dtype),
                sha256=hashlib.sha256(archive[k].tobytes()).hexdigest()) for k in archive.files}
    return dict(confidence=json.loads(conf.read_text()),structure_sha256=digest(structure),arrays=arrays)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--pilot',action='store_true');p.add_argument('--cache',type=Path,default=Path.home()/'.boltz')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(a.pool/'results.csv')
    cases=['ifnar_top7','pdl1_ankyrin','egfr_barrel']
    tasks=[]
    for case in cases:
        row=frame[(frame.case==case)&(frame.arm=='baseline')&(frame.seed==20261001)&(frame.draw==0)].iloc[0]
        target,binder=row.sequence.split('/')
        for draw in range(2):tasks.append(dict(name=f'{case}-{draw}',target=target,binder=binder,seed=20261021+draw))
    if a.pilot:tasks=tasks[:2]
    protocol=dict(tasks=tasks,workers=1 if a.pilot else 2,rounds=1 if a.pilot else 3,
        orders=[['subprocess','persistent'],['persistent','subprocess'],['subprocess','persistent']],
        model=dict(msa=False,kernels=False,diffusion_samples=1,checkpoint=str(a.cache/'boltz2_conf.ckpt')),
        gates=dict(exact_artifact_parity=True,minimum_median_wall_reduction=.10,max_round_slowdown=.05),
        sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'baseline/score_boltz2.py',ROOT/'baseline/boltz_worker.py']},
        input_csv_sha256=digest(a.pool/'results.csv'),checkpoint_sha256=digest(a.cache/'boltz2_conf.ckpt'),
        scope='Identical seeded requests, full startup/teardown included; OS disk caches warm; no model reuse')
    write_json(a.output/'protocol.json',protocol)
    before=snapshot();write_json(a.output/'runtime-before.json',before)
    report=dict(status='running',batches=[],comparisons=[]);write_json(a.output/'report.json',report)
    references={}
    try:
        for round_id in range(protocol['rounds']):
            for mode in protocol['orders'][round_id]:
                root=a.output/f'round-{round_id}-{mode}';root.mkdir()
                stop=threading.Event();samples=[]
                def monitor():
                    while not stop.is_set():
                        readings=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
                        samples.append([time.time(),*[int(x.strip()) for x in readings.split(',')]])
                        stop.wait(1)
                thread=threading.Thread(target=monitor,daemon=True);thread.start()
                began=time.monotonic();worker=BoltzWorkerPool(protocol['workers']) if mode=='persistent' else None
                def run(task):
                    start=time.monotonic();values=run_boltz2(task['target'],task['binder'],str(root/task['name']),
                        use_msa=False,flash_attn=False,no_kernels=True,checkpoint=a.cache/'boltz2_conf.ckpt',
                        cache=a.cache,seed=task['seed'],worker_pool=worker)
                    if not np.isfinite(values[0]):raise ValueError('Nonfinite ipTM')
                    return dict(name=task['name'],seconds=time.monotonic()-start,values=values,artifacts=artifacts(root/task['name']))
                try:
                    with ThreadPoolExecutor(max_workers=protocol['workers']) as executor:results=list(executor.map(run,tasks))
                finally:
                    if worker:worker.__exit__()
                    seconds=time.monotonic()-began;stop.set();thread.join()
                for result in results:
                    ref=references.setdefault(result['name'],result['artifacts'])
                    if ref!=result['artifacts']:raise ValueError('Prediction parity failed: '+result['name'])
                batch=dict(round=round_id,mode=mode,seconds=seconds,results=results,
                    peak_gpu_mib=max(x[1] for x in samples),samples=samples)
                report['batches'].append(batch);write_json(a.output/'report.json',report)
                print(json.dumps({k:v for k,v in batch.items() if k not in ('results','samples')}),flush=True)
            fresh=next(b for b in report['batches'] if b['round']==round_id and b['mode']=='subprocess')
            warm=next(b for b in report['batches'] if b['round']==round_id and b['mode']=='persistent')
            report['comparisons'].append(dict(round=round_id,ratio=warm['seconds']/fresh['seconds']))
        after=snapshot();write_json(a.output/'runtime-after.json',after)
        if before!=after:raise ValueError('Runtime changed')
        ratios=[x['ratio'] for x in report['comparisons']]
        report.update(status='passed',exact_parity=True,median_wall_ratio=float(np.median(ratios)),
            promote=not a.pilot and np.median(ratios)<=.9 and max(ratios)<=1.05)
    except BaseException as exc:
        report.update(status='failed',error=str(exc));raise
    finally:write_json(a.output/'report.json',report)

if __name__=='__main__':main()
