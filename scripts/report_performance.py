"""Audit every storage-mode output and apply the prespecified promotion gate."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gpu_smoke import replay_check
from baseline.result_io import write_json


def audit(root):
    root=Path(root).resolve();report=json.loads((root/'report.json').read_text())
    if report['status']!='passed':raise ValueError('Performance run incomplete')
    comparisons=[];parity={}
    for case in ('short','medium','long'):
        records={r['name']:r for r in report['runs'] if r['case']==case}
        for phase in ('cold','warm'):
            for mode in ('full','compact'):
                if phase=='cold' and mode=='full':continue
                key=f'{case}-{phase}-{mode}';pair=root/f'audit-{key}';pair.mkdir(exist_ok=True)
                for side,target in [('fixed',root/f'{case}_cold_full'),('replay',root/f'{case}_{phase}_{mode}')]:
                    link=pair/side
                    if link.is_symlink() and link.resolve()!=target:link.unlink()
                    if not link.exists():link.symlink_to(target,target_is_directory=True)
                parity[key]=replay_check(pair,tolerance=0)
        full=records[f'{case}_warm_full'];compact=records[f'{case}_warm_compact']
        comparisons.append(dict(case=case,full_warm_seconds=full['seconds'],compact_warm_seconds=compact['seconds'],
            full_candidate_bytes=full['candidate_pickle_bytes'],compact_candidate_bytes=compact['candidate_pickle_bytes'],
            storage_reduction=1-compact['candidate_pickle_bytes']/full['candidate_pickle_bytes'],
            warm_runtime_ratio=compact['seconds']/full['seconds'],
            full_rss_bytes=full['peak_process_tree_rss_bytes'],compact_rss_bytes=compact['peak_process_tree_rss_bytes'],
            full_gpu_bytes=full['peak_device_used_bytes'],compact_gpu_bytes=compact['peak_device_used_bytes']))
    gates=dict(exact_replay=all(r['max_coordinate_delta']==r['max_pae_delta']==0 for r in parity.values()),
               storage_reduction=all(r['storage_reduction']>=.8 for r in comparisons),
               warm_runtime=all(r['warm_runtime_ratio']<=1.1 for r in comparisons))
    result=dict(status='passed',comparisons=comparisons,parity=parity,gates=gates,promote_compact=all(gates.values()),
                limitations=['One observation per case/mode/phase; no speedup significance claim.',
                             'RSS sums process memory and may double-count shared pages; GPU used includes device baseline; sampled every second.',
                             'Three-iteration trajectories measure setup/storage, not full-search throughput.'])
    write_json(root/'promotion.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
