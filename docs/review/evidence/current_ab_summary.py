"""Exploratory trajectory-bootstrap summaries; not confirmatory hypothesis tests."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[3]
rng=np.random.default_rng(20260926)
NBOOT=100000

def load(group,arm):
    d=pd.read_csv(root/'baseline'/group/arm/'results.csv')
    d['trajectory']=d['name'].str.replace(r'_\d+$','',regex=True)
    d['passes']=(d.plddt>.8)&(d.iptm>.5)&(d.ipae<.35)
    return d.groupby('trajectory')['passes'].mean().sort_index()

def ci(x): return [round(float(v),2) for v in np.quantile(x,[.025,.975])*100]

last=load('ab_get_best','last'); best=load('ab_get_best','best')
assert last.index.equals(best.index)
diff=(best-last).to_numpy(); draws=rng.choice(diff,size=(NBOOT,len(diff)),replace=True).mean(1)
results={'best_minus_last': {'trajectories':len(diff),'difference_pp':round(float(diff.mean()*100),2),'paired_trajectory_bootstrap_95_pp':ci(draws)}}
for group,control,arms in [('ab_loss','cmap',['ipae0.05','ipae0.1','ipae0.2']),('ab_loss_fp','fp0',['fp0.1','fp0.3'])]:
    baseline=load(group,control).to_numpy()
    for arm in arms:
        a=load(group,arm).to_numpy()
        draws=rng.choice(a,size=(NBOOT,len(a)),replace=True).mean(1)-rng.choice(baseline,size=(NBOOT,len(baseline)),replace=True).mean(1)
        results[group+'/'+arm]={'trajectories_per_arm':len(a),'difference_pp':round(float((a.mean()-baseline.mean())*100),2),'unpaired_trajectory_bootstrap_95_pp':ci(draws)}
print(json.dumps(results,indent=2))
