"""Export compact immutable benchmark evidence; keep large raw outputs on the run host."""
import argparse,shutil
from pathlib import Path

def export(root,destination):
    destination.mkdir(parents=True,exist_ok=False)
    count=0
    for path in root.rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        parts=path.relative_to(root).parts
        if any(x.startswith(('cache-','audit-','parity-')) for x in parts) or 'optimization' in parts:continue
        if path.suffix not in ('.json','.csv','.log','.md','.tsv'):continue
        # Raw Boltz preprocessing metadata is redundant; preserve selected output receipts/confidence.
        if 'boltz-work' in parts and path.name!='selection.json' and not path.name.startswith('confidence_'):continue
        target=destination/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if target.suffix=='.log':target.write_text('\n'.join(line.rstrip() for line in target.read_text(errors='replace').splitlines())+'\n')
        count+=1
    print(f'Exported {count} compact evidence files')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();export(a.root,a.destination)
