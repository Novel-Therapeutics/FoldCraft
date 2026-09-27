"""CPU run lifecycle and immutable input provenance for the design CLI."""
from pathlib import Path
import csv
import hashlib
import json
import secrets
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone

from baseline.result_io import write_json


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stage_seed(seed, *identity):
    """Stable independent 32-bit streams, reproducible across Python processes."""
    payload = json.dumps([seed, *identity], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def start_run(folder, args, prepared):
    folder = Path(folder)
    # Exclusive creation; refuse existing directories, including stale/partial runs.
    folder.mkdir(parents=True, exist_ok=False)
    if args.seed is None:
        args.seed = secrets.randbits(32)
    target, binder, thot, bhot, mask = prepared
    inputs = folder / 'inputs'; inputs.mkdir()
    (inputs / 'target.pdb').write_text(target.pdb)
    if binder is not None:
        (inputs / 'binder.pdb').write_text(binder.pdb)
        (folder / 'template.pdb').write_text(binder.pdb)
    from input_validation import VHH_CONVENTION
    root = Path(__file__).resolve().parent
    code = {name:sha256(root / name) for name in ('FoldCraft.py','input_validation.py',
        'sequence_design.py','cmap_utils.py','biopython_utils.py','run_state.py','model_validation.py')}
    packages = {}
    for package in ('colabdesign','jax','jaxlib','numpy','biopython'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    manifest = dict(packages=packages, seed_protocol='sha256-stage-v1', created_at=datetime.now(timezone.utc).isoformat(), code=code,
                    vhh_convention=VHH_CONVENTION if args.vhh else None, schema=1, status='running', config=vars(args).copy(),
                    target=target.manifest(), binder=binder.manifest() if binder else None,
                    mapped_hotspots=dict(target=thot, binder=bhot, binder_mask=mask),
                    validation_artifacts='per-model-v1', validation_policy='all_models_pass',
                    validation_models=args.validation_models.split(','), validation_recycles=args.validation_recycles)
    write_json(folder / 'run.json', manifest)
    return manifest


def finish_run(folder, state, *, status='complete'):
    folder = Path(folder)
    state = dict(state, status=status, updated_at=datetime.now(timezone.utc).isoformat())
    if status == 'complete':
        with open(folder / 'results.csv', newline='') as f:
            rows = list(csv.DictReader(f))
            names = [r['name'] for r in rows]
        if not names or len(set(names)) != len(names) or any(not n or Path(n).name != n for n in names):
            raise ValueError('A completed run must have unique candidate records')
        artifacts = {}
        from model_validation import candidate_files
        for name in names:
            for relative in candidate_files(folder, name, require_receipt=state.get('validation_artifacts') == 'per-model-v1'):
                p = folder / relative
                if not p.is_file() or p.stat().st_size == 0:
                    raise ValueError(f'Missing candidate artifact: {p}')
                artifacts[relative] = sha256(p)
        for p in [folder / 'template.pdb', folder / 'attempts.json',
                  *sorted(folder.glob('fold_cond_cmap*.npy')), *sorted((folder / 'inputs').glob('*.pdb'))]:
            if p.is_file():
                artifacts[str(p.relative_to(folder))] = sha256(p)
        from model_validation import verify_receipt
        for row in rows:
            verify_receipt(folder, row['name'], artifacts, row)
        records = [{'name':r['name'], 'sequence':r.get('sequence')} for r in rows]
        state.update(candidate_names=names, candidate_records=records, artifacts=artifacts)
    write_json(folder / 'run.json', state)


def completed_run(folder):
    """Historical CSV-only or damaged output is not an executable completion."""
    folder = Path(folder)
    try:
        state = json.loads((folder / 'run.json').read_text())
        if state.get('schema') != 1 or state.get('status') != 'complete':
            return False
        with open(folder / 'results.csv', newline='') as f:
            rows = list(csv.DictReader(f))
            names = [r['name'] for r in rows]
        if not names or len(set(names)) != len(names) or any(not n or Path(n).name != n for n in names) or names != state.get('candidate_names'):
            return False
        if [{'name':r['name'], 'sequence':r.get('sequence')} for r in rows] != state.get('candidate_records'):
            return False
        artifacts = state.get('artifacts', {})
        from model_validation import candidate_files
        required = {p for n in names for p in candidate_files(folder, n, require_receipt=state.get('validation_artifacts') == 'per-model-v1')}
        if not required <= artifacts.keys():
            return False
        if any(Path(p).is_absolute() or '..' in Path(p).parts for p in artifacts):
            return False
        from model_validation import verify_receipt
        for row in rows:
            verify_receipt(folder, row['name'], artifacts, row)
        return all((folder / p).is_file() and sha256(folder / p) == artifacts[p] for p in artifacts)
    except (OSError, ValueError, KeyError, TypeError):
        return False
