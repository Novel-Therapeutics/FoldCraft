import sys,json,time
from pathlib import Path
import openmm
from baseline import score_openmm as score
root=Path('/home/bizon/projects/foldcraft-gpu-20260927')
out=root/'scoring-qc';out.mkdir(exist_ok=True)
score._PLATFORM['p']=openmm.Platform.getPlatformByName('CUDA')
score._PLATFORM['p'].setPropertyDefaultValue('Precision','mixed')
score._PLATFORM['p'].setPropertyDefaultValue('DeterministicForces','true')
pdb=root/'full-default-01/designs/traj_1_0.pdb'
results=[]
for iterations in (0,1,5000):
    started=time.monotonic()
    try:
        de,energy,qc=score.interface_dE(str(pdb),iterations,seed=20260927,return_qc=True)
        assert iterations!=1,'One iteration unexpectedly passed convergence'
        record=dict(iterations=iterations,score=de,energy=energy,qc=qc)
    except score.ConvergenceError as exc:
        assert iterations==1,exc
        record=dict(iterations=iterations,rejected=True,qc=exc.qc)
    record['seconds']=time.monotonic()-started;results.append(record)
    (out/'openmm.json').write_text(json.dumps(dict(openmm=openmm.__version__,records=results),indent=2)+'\n')
    print(json.dumps(record),flush=True)
assert results[-1]['qc']['status']=='converged'
