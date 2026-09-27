"""Export compact evidence without large inference pickles or model weights."""
import json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);dest.mkdir(exist_ok=False)
paths=set(root.glob('*.json'))|set(root.glob('*.csv'))|set(root.glob('*.log'))|set(root.glob('*.md'))
paths.update((root/'pool').glob('*.json'))
paths.update((root/'pool').glob('*.csv'))
paths.update((root/'pool/designs').glob('*.pdb'))
protocol=json.loads((root/'protocol.json').read_text())
for job in protocol['jobs']:
    folder=root/job['name']
    paths.update(folder.glob('*.json'))
    paths.update(folder.glob('results.csv'))
    paths.update((folder/'designs').glob('*.validation.json'))
paths.update((root/'boltz-work').rglob('selection.json'))
paths.update((root/'boltz-work').rglob('confidence_*.json'))
for path in sorted(paths):
    if not path.is_file():continue
    target=dest/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
    if target.suffix=='.log':
        target.write_text('\n'.join(line.rstrip() for line in target.read_text(errors='replace').splitlines())+'\n')
print(json.dumps(dict(files=len(paths),bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()))))
