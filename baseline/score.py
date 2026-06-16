import os, sys, math, warnings; warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, Superimposer
import pandas as pd
ROOT="/root/FoldCraft"
RUNS=os.path.join(ROOT, sys.argv[1] if len(sys.argv)>1 else "baseline/runs")
TEMPL=os.path.join(ROOT,"baseline/templates")
p=PDBParser(QUIET=True)
def rmsd_to_template(design_pdb, template_pdb):
    d=p.get_structure("d",design_pdb); t=p.get_structure("t",template_pdb)
    db=[r["CA"] for r in d[0]["B"] if "CA" in r]
    ta=[r["CA"] for r in list(t[0])[0] if "CA" in r]
    n=min(len(db),len(ta))
    if n<3: return None
    s=Superimposer(); s.set_atoms(ta[:n], db[:n]); return round(s.rms,2)
def wilson(k,n):
    if n==0: return (0,0,0)
    z=1.96; ph=k/n; d=1+z*z/n
    c=(ph+z*z/(2*n))/d; h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/d
    return (round(100*ph,1), round(100*max(0,c-h),1), round(100*min(1,c+h),1))
CRIT=[("plddt>.8",lambda r:r["plddt"]>0.8),
      ("iptm>.5", lambda r:r["iptm"]>0.5),
      ("ipae<.35",lambda r:r["ipae"]<0.35),
      ("rmsd<3.5",lambda r:r["_rmsd"] is not None and r["_rmsd"]<3.5)]
print(f"{'fold':9} {'n':>3} " + " ".join(f"{k:>8}" for k,_ in CRIT) + f" {'ALL':>4} {'success% [95%CI]':>20}")
for fold in sorted(os.listdir(RUNS)):
    csvf=os.path.join(RUNS,fold,"results.csv")
    if not os.path.exists(csvf): continue
    df=pd.read_csv(csvf); tmpl=os.path.join(TEMPL,f"{fold}.pdb"); recs=[]
    for _,row in df.iterrows():
        dp=os.path.join(RUNS,fold,"designs",f"{row['name']}.pdb")
        if not os.path.exists(dp): continue
        r=dict(row); r["_rmsd"]=rmsd_to_template(dp,tmpl); recs.append(r)
    n=len(recs)
    counts=[sum(1 for r in recs if fn(r)) for _,fn in CRIT]
    npass=sum(1 for r in recs if all(fn(r) for _,fn in CRIT))
    sr,lo,hi=wilson(npass,n)
    print(f"{fold:9} {n:>3} " + " ".join(f"{c:>8}" for c in counts) + f" {npass:>4} {f'{sr}% [{lo}-{hi}]':>20}")
