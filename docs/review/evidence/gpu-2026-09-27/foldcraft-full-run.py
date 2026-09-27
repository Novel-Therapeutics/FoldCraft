import json,os,time,sys
from pathlib import Path
from scripts.gpu_smoke import run_owned
from run_state import completed_run
root=Path('/home/bizon/projects/foldcraft-gpu-20260927')
folder=root/'full-default-01'
cmd=[sys.executable,'FoldCraft.py','--output_folder',str(folder),'--data_dir',str(root/'weights'),
     '--seed','20260927','--num_designs','2','--mpnn_samples','2',
     '--target_template','examples/targets/pd-l1-1.pdb','--target_hotspots','30-34,50-54,69-76',
     '--binder_template','examples/templates/1qys1.pdb','--binder_hotspots','26-40,58-71']
started=time.monotonic()
try:
    with (root/'full-default-01.log').open('w') as log:
        code=run_owned(cmd,env=dict(os.environ),timeout=1800,stdout=log,stderr=-2)
    report=dict(command=cmd,seconds=time.monotonic()-started,returncode=code,completed=completed_run(folder))
    assert code==0 and report['completed'],report
finally:
    (root/'full-default-01.summary.json').write_text(json.dumps(locals().get('report',dict(status='failed',seconds=time.monotonic()-started)),indent=2)+'\n')
print(json.dumps(report))
