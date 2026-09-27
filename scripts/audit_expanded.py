"""Independent CPU artifact and paired-stream audit of the expanded campaign."""
import argparse,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from baseline.result_io import write_json
from biopython_utils import hotspot_residues
from run_state import completed_run,sha256,stage_seed


def audit(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());jobs=protocol['jobs']
    parser=PDBParser(QUIET=True);count=predictions=protected=0;maximum=0.
    for job in jobs:
        folder=root/job['name']
        if not completed_run(folder):raise ValueError('Invalid source run: '+job['name'])
        original=parser.get_structure('trajectory',folder/'traj/traj_1.pdb')[0]
        original_binder=seq1(''.join(r.resname for r in original['B']))
        interface=hotspot_residues(str(folder/'traj/traj_1.pdb'),'B')
        history=json.loads((folder/'optimization/traj_1.json').read_text())
        assert len(history['records'])==220 and history['seed']==stage_seed(job['seed'],'design',1)
        for receipt in (folder/'designs').glob('*.validation.json'):
            info=json.loads(receipt.read_text());candidate=receipt.name.removesuffix('.validation.json');draw=int(candidate.rsplit('_',1)[1])
            target,binder=info['sequence'].split('/')
            for pos in interface:
                assert binder[pos-1]==original_binder[pos-1];protected+=1
            for name,record in info['models'].items():
                stem=candidate if name==info['primary_model'] else candidate+'.'+name
                structure=parser.get_structure('prediction',folder/'designs'/f'{stem}.pdb')
                assert len(structure)==1 and [c.id for c in structure[0]]==['A','B']
                assert [seq1(''.join(r.resname for r in structure[0][c])) for c in ('A','B')]==[target,binder]
                with (folder/'designs'/f'{stem}.pickle').open('rb') as stream:data=pickle.load(stream)
                length=len(target)+len(binder)
                assert data['pae'].shape==(1,length,length) and np.isfinite(data['pae']).all() and (data['pae']>=0).all()
                assert data['atom_positions'].shape==(1,length,37,3) and np.isfinite(data['atom_positions']).all()
                pdb_ca=np.asarray([r['CA'].coord for c in structure[0] for r in c])
                delta=float(np.max(np.abs(pdb_ca-data['atom_positions'][0,:,1,:])))
                assert delta<=.00055;maximum=max(maximum,delta)
                seed=stage_seed(job['seed'],'validation',1,draw)
                if name!=info['primary_model']:seed=stage_seed(seed,name)
                assert record['seed']==seed and record['actual_recycles']==3 and record['executed_models']==[name]
                predictions+=1
            count+=1
    assert count==96 and predictions==192
    result=dict(status='passed',runs=len(jobs),candidates=count,predictions=predictions,
                protected_interface_residues_checked=protected,max_pdb_rounding_delta=maximum,
                complete_histories=48,notes='All predicted arrays finite, chain/sequence/model/seed identity exact; PDB coordinates differ only by 0.001 A text rounding.')
    write_json(root/'artifact_audit.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
