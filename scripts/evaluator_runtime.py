"""Fingerprint the independent evaluator sources even without wheel metadata."""
import argparse,hashlib,importlib.util,json,sys
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from run_state import sha256
from baseline.result_io import write_json


def snapshot():
    result=dict(python=sys.version,executable=sys.executable,packages={},sources={})
    for name in ('torch','transformers','boltz','numpy','scipy','biopython','pytorch-lightning'):
        try:result['packages'][name]=version(name)
        except PackageNotFoundError:result['packages'][name]=None
    for module in ('boltz','transformers'):
        spec=importlib.util.find_spec(module)
        if spec is None or not spec.origin:raise ValueError('Missing evaluator module: '+module)
        root=Path(spec.origin).resolve().parent;files={}
        for p in sorted(root.rglob('*.py')):
            before=p.stat();files[str(p.relative_to(root))]=sha256(p);after=p.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('Evaluator source changed during capture')
        result['sources'][module]=dict(root=str(root),files=files,digest=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--compare',type=Path);a=p.parse_args();value=snapshot()
    if a.compare and value!=json.loads(a.compare.read_text()):raise ValueError('Independent evaluator runtime changed')
    write_json(a.output,value);print({k:v['digest'] for k,v in value['sources'].items()})
