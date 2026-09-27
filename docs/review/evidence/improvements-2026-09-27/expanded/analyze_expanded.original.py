"""Prespecified paired temperature/ranking analysis with explicit holdout limits."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from baseline.result_io import write_json,atomic_write
from scripts.analyze_pilot import build_pool
from run_state import sha256

METRICS=['proxy_pass','rmsd','esmfold_rmsd','esmfold_plddt','boltz2_iptm','epitope_coverage','interchain_clashes']


def prepare(root):
    build_pool(root)
    protocol=json.loads((root/'protocol.json').read_text())
    jobs={j['name']:j for j in protocol['jobs']}
    manifest=json.loads((root/'pool/pool_manifest.json').read_text())
    frame=pd.read_csv(root/'pool/results.csv')
    for idx,row in frame.iterrows():
        source=manifest['sources'][row['name']];job=jobs[source['source_run']]
        for key in ('split','target','scaffold'):frame.loc[idx,key]=job[key]
        frame.loc[idx,'draw']=int(source['candidate'].rsplit('_',1)[1])
    frame['draw']=frame['draw'].astype(int)
    atomic_write(root/'pool/results.csv',lambda p:frame.to_csv(p,index=False))


def bootstrap_mean(values,seed=20261001,repeats=10000,clusters=None):
    values=np.asarray(values,dtype=float)
    n=len(values)
    if clusters is not None:
        values=pd.Series(values).groupby(list(clusters)).mean().to_numpy()
    rng=np.random.default_rng(seed)
    samples=values[rng.integers(len(values),size=(repeats,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),conditional_95_interval=[float(x) for x in np.quantile(samples,[.025,.975])],n_paired_trajectories=n,n_case_clusters=len(values))


def ranking_choices(frame):
    picks=[]
    for (case,seed),part in frame.groupby(['case','seed']):
        baseline=part.sort_values(['iptm','name'],ascending=[False,True]).iloc[0]
        part=part.assign(violations=(part.rmsd>3.5).astype(int)+(part.interchain_clashes>5).astype(int)+(part.epitope_coverage<.1).astype(int))
        improved=part.sort_values(['violations','iptm','name'],ascending=[True,False,True]).iloc[0]
        for policy,row in [('af2_rank',baseline),('fold_contact_rank',improved)]:
            picks.append(dict(row,policy=policy))
    return pd.DataFrame(picks)


def analyze(frame,protocol):
    needed=['esmfold_rmsd','esmfold_plddt','boltz2_iptm','rmsd','epitope_coverage','interchain_clashes']
    if frame['name'].duplicated().any() or not np.isfinite(frame[needed].to_numpy(dtype=float)).all():
        raise ValueError('Analysis requires complete finite independent evaluations and unique identities')
    expected={(j['case'],j['seed'],j['arm']) for j in protocol['jobs']}
    observed=set(map(tuple,frame[['case','seed','arm']].drop_duplicates().to_numpy()))
    if expected!=observed or not frame.groupby(['case','seed','arm']).size().eq(2).all():raise ValueError('Unbalanced or incomplete paired pool')
    frame=frame.copy()
    frame['proxy_pass']=(frame.esmfold_plddt>=70)&(frame.esmfold_rmsd<=3.5)&(frame.boltz2_iptm>=.6)
    keys=['split','family','target','scaffold','case','seed','arm']
    blocks=frame.groupby(keys)[METRICS].mean().reset_index()
    summaries=[];comparisons={}
    for split in ('development','holdout'):
        part=blocks[blocks.split==split]
        for arm in ('baseline','mpnn_temp_02'):
            rows=part[part.arm==arm]
            summaries.append(dict(split=split,arm=arm,paired_blocks=len(rows),families=int(rows.family.nunique()),
                                  metrics={m:float(rows[m].mean()) for m in METRICS}))
        a=part[part.arm=='baseline'].set_index(['case','seed']);b=part[part.arm=='mpnn_temp_02'].set_index(['case','seed']).reindex(a.index)
        comparisons[split]={m:bootstrap_mean((b[m]-a[m]).to_numpy(),clusters=a.index.get_level_values("case")) for m in METRICS}
    selected=ranking_choices(frame)
    rank=[];rank_delta={}
    for split in ('development','holdout'):
        part=selected[selected.split==split]
        for policy,rows in part.groupby('policy'):
            rank.append(dict(split=split,policy=policy,selected=len(rows),metrics={m:float(rows[m].mean()) for m in METRICS}))
        a=part[part.policy=='af2_rank'].set_index(['case','seed']);b=part[part.policy=='fold_contact_rank'].set_index(['case','seed']).reindex(a.index)
        rank_delta[split]=bootstrap_mean((b.proxy_pass.astype(float)-a.proxy_pass.astype(float)).to_numpy(),clusters=a.index.get_level_values("case"))
    # Confidence policy coverage on the identical frozen pool is descriptive.
    policies=[]
    for split in ('development','holdout'):
        part=frame[frame.split==split]
        for policy in ('single_model_pass','two_model_pass'):
            chosen=part[part[policy].eq(True)]
            policies.append(dict(split=split,policy=policy,selected=len(chosen),total=len(part),
                proxy_pass_fraction=None if chosen.empty else float(chosen.proxy_pass.mean())))
    heldout_families=int(frame.loc[frame.split=='holdout','family'].nunique())
    temp=comparisons['holdout'];threshold=protocol['promotion']
    gates=dict(enough_unseen_families=heldout_families>=threshold['min_unseen_target_families'],
               primary_gain=temp['proxy_pass']['mean']>=threshold['primary_gain_pp']/100,
               interval_positive=temp['proxy_pass']['conditional_95_interval'][0]>0,
               fold_noninferiority=temp['rmsd']['mean']<=threshold['max_template_rmsd_regression_A'],
               clash_noninferiority=temp['interchain_clashes']['mean']<=threshold['max_clash_regression'])
    ranking_gates=dict(enough_unseen_families=gates['enough_unseen_families'],
        primary_gain=rank_delta['holdout']['mean']>=threshold['primary_gain_pp']/100,
        interval_positive=rank_delta['holdout']['conditional_95_interval'][0]>0)
    report=dict(status='complete',candidates=len(frame),paired_blocks=len(blocks)//2,
        summaries=summaries,temperature_deltas=comparisons,ranking=rank,ranking_deltas=rank_delta,confidence_selection=policies,
        temperature_promotion=dict(promote=all(gates.values()),gates=gates),
        ranking_promotion=dict(promote=all(ranking_gates.values()),gates=ranking_gates),
        family_means=blocks.groupby(['split','family','arm'])[METRICS].mean().reset_index().to_dict('records'),
        diversity={arm:int(part.binder_sequence.nunique()) for arm,part in frame.groupby('arm')},
        limitations=['Proxy outcomes are not binding labels.','Only one unseen family; confidence intervals condition on these observed targets/scaffolds.',
                     'Sibling draws are aggregated; closely related case outcomes may remain correlated. No population-level family-generalization claim.',
                     'Fixed draw comparison, not equal-GPU-time search yield.'])
    return report,blocks,selected


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--prepare',action='store_true');a=p.parse_args();root=a.root
    if a.prepare:prepare(root);return
    manifest=json.loads((root/'pool/pool_manifest.json').read_text());frame=pd.read_csv(root/'pool/results.csv')
    if set(frame.name)!=set(manifest['sources']):raise ValueError('Pool membership changed')
    for row in frame.to_dict('records'):
        source=manifest['sources'][row['name']]
        if source['sequence']!=row['sequence'] or sha256(root/'pool/designs'/f'{row["name"]}.pdb')!=source['pdb_sha256']:raise ValueError('Frozen pool input changed')
    report,blocks,selected=analyze(frame,json.loads((root/'protocol.json').read_text()))
    write_json(root/'analysis.json',report)
    atomic_write(root/'paired_blocks.csv',lambda p:blocks.to_csv(p,index=False))
    atomic_write(root/'ranking_selection.csv',lambda p:selected.to_csv(p,index=False))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
