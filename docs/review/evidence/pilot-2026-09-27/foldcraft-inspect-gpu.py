import json,pickle,sys
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from baseline.add_ipsae import single_model_pae
from baseline.ipsae import ipsae
from baseline.add_rmsd import template_ca_atoms,rmsd_to_template
from run_state import completed_run,sha256
from biopython_utils import hotspot_residues

root=Path(sys.argv[1]); parser=PDBParser(QUIET=True);out=[]
for case in (sys.argv[2:] or ('fixed','replay','bounded_sample','vhh','long_target')):
    folder=root/case
    state=json.loads((folder/'run.json').read_text())
    if state['status']=='complete':assert completed_run(folder),case
    elif case=='bounded_sample':assert state['status']=='exhausted'
    else:raise AssertionError((case,state['status']))
    for receipt in sorted((folder/'designs').glob('*.validation.json')):
        record=json.loads(receipt.read_text());candidate=receipt.name.removesuffix('.validation.json')
        target=parser.get_structure('target',folder/'inputs/target.pdb')
        target_seq=''.join(seq1(r.resname) for r in next(iter(target[0])))
        assert record['sequence'].split('/')[0]==target_seq
        trajectory=folder/'traj'/(candidate.rsplit('_',1)[0]+'.pdb')
        fixed=hotspot_residues(str(trajectory),'B') if state['config']['redesign_method']=='non-interface' else {}
        binder_seq=record['sequence'].split('/')[1]
        assert all(binder_seq[pos-1]==aa for pos,aa in fixed.items()),(case,candidate,'fixed MPNN residues changed')
        results={}
        for model in record['model_order']:
            suffix='' if model==record['primary_model'] else '.'+model
            path=folder/'designs'/(candidate+suffix+'.pdb');pk=path.with_suffix('.pickle')
            structure=parser.get_structure('prediction',path)
            assert len(structure)==1
            chains=list(structure[0]);assert [c.id for c in chains]==['A','B']
            sequence='/'.join(''.join(seq1(r.resname) for r in c) for c in chains)
            assert sequence==record['sequence'],(case,candidate,model,sequence,record['sequence'])
            with pk.open('rb') as f:data=pickle.load(f)
            pae=single_model_pae(data['pae']);lens=[len(c) for c in chains]
            assert pae.shape==(sum(lens),sum(lens)) and np.isfinite(pae).all() and (pae>=0).all()
            coords=np.asarray(data['atom_positions'])[0]
            assert np.isfinite(coords).all()
            # AlphaFold atom37 puts CA at index 1; PDB output rounds to .001 A.
            ca=np.asarray([r['CA'].coord for c in chains for r in c])
            delta=float(np.max(abs(ca-coords[:,1,:])))
            assert delta<.001,(case,candidate,model,delta)
            assert sha256(path)==record['models'][model]['pdb_sha256']
            assert sha256(pk)==record['models'][model]['pickle_sha256']
            assert record['models'][model]['actual_recycles']==record['models'][model]['requested_recycles']
            metrics=dict(chain_lengths=lens,pae_min=float(pae.min()),pae_max=float(pae.max()),
                         ipsae=float(ipsae(pae,*lens,10)),pdb_coordinate_max_delta=delta,
                         pickle_bytes=pk.stat().st_size,metrics=record['models'][model]['metrics'])
            if (folder/'template.pdb').exists():
                metrics['binder_template_ca_rmsd']=rmsd_to_template(str(path),template_ca_atoms(str(folder/'template.pdb')))
            results[model]=metrics
        out.append(dict(case=case,candidate=candidate,models=results,fixed_interface_positions_checked=len(fixed)))
report=dict(status='passed',candidates=len(out),predictions=sum(len(c['models']) for c in out),records=out)
(root/'artifact-inspection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'}))
