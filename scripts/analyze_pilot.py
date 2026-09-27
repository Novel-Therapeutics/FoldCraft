"""Build a frozen candidate pool and summarize paired structural pilot outcomes."""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.add_rmsd import template_ca_atoms,rmsd_to_template
from baseline.ipsae import ipsae
from baseline.result_io import write_json,atomic_write
from input_validation import residue_range
from run_state import completed_run,sha256


def geometry(pdb,hotspots):
    model=PDBParser(QUIET=True).get_structure('design',str(pdb))[0]
    a=[x for x in model['A'].get_atoms() if x.element!='H']
    b=[x for x in model['B'].get_atoms() if x.element!='H']
    ta,tb=cKDTree([x.coord for x in a]),cKDTree([x.coord for x in b])
    contacts=ta.query_ball_tree(tb,4.)
    residues={a[i].parent.id[1] for i,v in enumerate(contacts) if v}
    hotspots=set(residue_range(hotspots))
    return dict(epitope_coverage=len(residues&hotspots)/len(hotspots),
                epitope_precision=len(residues&hotspots)/len(residues) if residues else 0.,
                interchain_clashes=sum(len(v) for v in ta.query_ball_tree(tb,2.)),
                target_contact_residues=len(residues))


def build_pool(root):
    protocol=json.loads((root/'protocol.json').read_text())
    execution=json.loads((root/'execution.json').read_text())
    if execution['status']!='passed':raise ValueError('Pilot is incomplete')
    pool=root/'pool';pool.mkdir(exist_ok=False);(pool/'designs').mkdir()
    timings={j['name']:j['seconds'] for j in execution['jobs']}
    rows=[];sources={}
    for job in protocol['jobs']:
        folder=root/job['name']
        if not completed_run(folder):raise ValueError(f'Invalid source run: {folder}')
        state=json.loads((folder/'run.json').read_text())
        for record in pd.read_csv(folder/'results.csv').to_dict('records'):
            candidate=record['name'];name=job['name']+'_'+candidate
            receipt=json.loads((folder/'designs'/f'{candidate}.validation.json').read_text())
            primary=receipt['models'][receipt['primary_model']]
            pdb=folder/'designs'/f'{candidate}.pdb'
            with (folder/'designs'/f'{candidate}.pickle').open('rb') as f:pae=np.asarray(pickle.load(f)['pae'])[0]
            tlen=len(receipt['sequence'].split('/')[0]);blen=len(receipt['sequence'].split('/')[1])
            shutil.copy2(pdb,pool/'designs'/f'{name}.pdb')
            row=dict(record,name=name,case=job['case'],family=job['family'],arm=job['arm'],seed=job['seed'],
                trajectory=candidate.rsplit('_',1)[0],binder_sequence=receipt['sequence'].split('/')[1],
                single_model_pass=primary['passed'],two_model_pass=receipt['all_models_pass'],
                rmsd=rmsd_to_template(str(pdb),template_ca_atoms(str(folder/'template.pdb'))),
                ipsae=ipsae(pae,tlen,blen,10),job_seconds=timings[job['name']],
                second_model_seconds=receipt['models']['model_2_ptm']['prediction_seconds'],
                **geometry(pdb,state['mapped_hotspots']['target']))
            rows.append(row)
            sources[name]=dict(source_run=job['name'],candidate=candidate,run_sha256=sha256(folder/'run.json'),
                               pdb_sha256=sha256(pdb),sequence=row['sequence'],primary_model=receipt['primary_model'])
    atomic_write(pool/'results.csv',lambda path:pd.DataFrame(rows).to_csv(path,index=False))
    write_json(pool/'pool_manifest.json',dict(protocol_sha256=sha256(root/'protocol.json'),sources=sources))


def summarize(root):
    frame=pd.read_csv(root/'pool/results.csv')
    manifest=json.loads((root/'pool/pool_manifest.json').read_text())
    if set(frame['name'])!=set(manifest['sources']):raise ValueError('Candidate pool membership changed')
    for row in frame.to_dict('records'):
        source=manifest['sources'][row['name']]
        if row['sequence']!=source['sequence'] or sha256(root/'pool/designs'/f'{row["name"]}.pdb')!=source['pdb_sha256']:
            raise ValueError('Frozen candidate input changed')
    metrics=['single_model_pass','two_model_pass','rmsd','ipsae','epitope_coverage','interchain_clashes',
             'esmfold_plddt','esmfold_rmsd','boltz2_iptm','openmm_dE']
    for column in metrics:
        if column not in frame:frame[column]=np.nan
    # Aggregate siblings first. Each case/seed is one paired observation.
    paired=frame.groupby(['case','seed','arm'])[metrics].mean().reset_index()
    atomic_write(root/'paired_blocks.csv',lambda path:paired.to_csv(path,index=False))
    summaries=[]
    for arm,part in frame.groupby('arm'):
        block=paired[paired.arm==arm]
        entry=dict(arm=arm,candidates=len(part),paired_blocks=len(block),
                   unique_sequences=int(part.binder_sequence.nunique()),
                   generation_seconds=float(part.groupby(['case','seed']).job_seconds.first().sum()) if 'job_seconds' in part else None,
                   second_model_prediction_seconds=float(part.second_model_seconds.sum()) if 'second_model_seconds' in part else None,
                   metrics={k:dict(mean=None if block[k].isna().all() else float(block[k].mean()),
                                   measured=int(part[k].notna().sum()),missing=int(part[k].isna().sum())) for k in metrics})
        summaries.append(entry)
    deltas=[]
    for (case,seed),group in paired.groupby(['case','seed']):
        by=group.set_index('arm');base=by.loc['baseline']
        for arm in by.index:
            if arm=='baseline':continue
            deltas.append(dict(case=case,seed=int(seed),arm=arm,delta={k:None if pd.isna(by.loc[arm,k]) or pd.isna(base[k]) else float(by.loc[arm,k]-base[k]) for k in metrics}))
    selection=[]
    for policy in ('single_model_pass','two_model_pass'):
        chosen=frame[frame[policy].eq(True)]
        selection.append(dict(policy=policy,selected=len(chosen),total=len(frame),
            boltz_iptm_mean=None if chosen.boltz2_iptm.isna().all() else float(chosen.boltz2_iptm.mean()),
            esmfold_rmsd_mean=None if chosen.esmfold_rmsd.isna().all() else float(chosen.esmfold_rmsd.mean())))
    report=dict(status='exploratory',summaries=summaries,paired_deltas=deltas,selection=selection,
                limitations=['No assay labels; no binding-accuracy conclusion.','Two target families, one scaffold, four paired blocks; no promotion or significance claim.',
                             'Independent metrics are computational proxies. OpenMM is a geometry diagnostic.',
                             'Runtime includes compilation and all inference; persistent cache was shared.'])
    write_json(root/'analysis.json',report)
    lines=['# Paired structural pilot','', 'Exploratory results; no production defaults were changed. No binding-accuracy claim is supported.','',
           '| Arm | Candidates | Single-model pass | Two-model pass | Template RMSD (Å) | Epitope coverage | ESMFold RMSD (Å) | Boltz ipTM |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def fmt(entry,key):
        v=entry['metrics'][key]['mean'];return 'unknown' if v is None else f'{v:.3f}'
    for e in summaries:
        lines.append('| '+ ' | '.join([e['arm'],str(e['candidates']),*[fmt(e,k) for k in ('single_model_pass','two_model_pass','rmsd','epitope_coverage','esmfold_rmsd','boltz2_iptm')]])+' |')
    lines+=['','Means give each case/seed trajectory equal weight. Sibling MPNN samples are not independent replicates. Missing measurements remain unknown.','',
            'See `paired_blocks.csv` and `analysis.json` for paired changes, coverage counts and selection comparisons.','',*report['limitations']]
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('--build-pool',action='store_true')
    args=parser.parse_args()
    if args.build_pool:build_pool(args.root)
    summarize(args.root)

if __name__=='__main__':main()
