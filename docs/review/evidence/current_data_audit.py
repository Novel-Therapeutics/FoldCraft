"""Read-only audit of recorded FoldCraft metrics; no structural rescoring."""
import ast
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[3]
records=[]
for path in sorted(ROOT.glob('baseline/**/results.csv')):
    df=pd.read_csv(path)
    entry={'path':str(path.relative_to(ROOT)), 'rows':len(df),'columns':list(df.columns)}
    if 'name' in df:
        entry['duplicate_names']=int(df.name.duplicated().sum())
        entry['trajectories']=int(df.name.str.replace(r'_\d+$','',regex=True).nunique())
    if 'sequence' in df:
        entry['duplicate_sequences']=int(df.sequence.duplicated().sum())
        entry['sequence_lengths']=[int(df.sequence.str.len().min()),int(df.sequence.str.len().max())]
        entry['sequence_separators']=int(df.sequence.str.contains('/').sum())
    entry['missing_metrics']={c:int(df[c].isna().sum()) for c in ['plddt','iptm','ipae','rmsd','openmm_dE','boltz2_iptm','esmfold_rmsd'] if c in df}
    if 'openmm_dE' in df:
        v=df.openmm_dE.dropna()
        entry['openmm_extreme_abs_gt_1e6']=int((v.abs()>1e6).sum())
        entry['openmm_positive']=int((v>0).sum())
        entry['openmm_max']=float(v.max()) if len(v) else None
    if {'plddt','iptm','ipae'}<=set(df.columns):
        af=(df.plddt>.8)&(df.iptm>.5)&(df.ipae<.35)
        entry['af_pass']=int(af.sum())
        if 'boltz2_iptm' in df:
            entry['afpass_missing_boltz']=int(df.loc[af,'boltz2_iptm'].isna().sum())
        if 'rmsd' in df:
            entry['fold_pass']=int((df.rmsd<3.5).sum())
            entry['combined_pass']=int((af&(df.rmsd<3.5)).sum())
    records.append(entry)
print(json.dumps(records,indent=2))
