"""Model-specific AF2 validation; no JAX import until the caller runs inference."""
from pathlib import Path
import json
import math
import pickle
import time
import numpy as np

from baseline.result_io import atomic_write, write_json

ALLOWED_MODELS = ('model_1_ptm', 'model_2_ptm')


def model_names(value):
    names = value.split(',') if isinstance(value, str) else list(value)
    if not names or len(set(names)) != len(names) or any(n not in ALLOWED_MODELS for n in names):
        raise ValueError('Validation models must be unique names from model_1_ptm,model_2_ptm')
    return names


def passes(metrics):
    return metrics['plddt'] > .8 and metrics['i_ptm'] > .5 and metrics['i_pae'] < .35


def model_stem(candidate, model, primary):
    if not candidate or Path(candidate).name != candidate:
        raise ValueError('Candidate must be a basename')
    return candidate if model == primary else f'{candidate}.{model}'


def candidate_files(folder, candidate, *, require_receipt=False):
    """Yield artifact paths relative to a run, including every validation model."""
    receipt = Path(folder) / 'designs' / f'{candidate}.validation.json'
    if not receipt.exists():
        if require_receipt:
            raise ValueError(f'Missing validation receipt: {receipt}')
        return [f'designs/{candidate}{ext}' for ext in ('.pdb', '.pickle')]
    data = json.loads(receipt.read_text())
    names = model_names(data['model_order'])
    if data.get('schema') != 1 or data['primary_model'] != names[0] or set(data['models']) != set(names):
        raise ValueError('Invalid model validation receipt')
    paths = [f'designs/{candidate}.validation.json']
    for model in names:
        stem = model_stem(candidate, model, names[0])
        paths.extend(f'designs/{stem}{ext}' for ext in ('.pdb', '.pickle'))
    return paths


def validate_candidate(predictor, sequence, binder_len, folder, candidate, models, recycles, seed):
    """Predict each model separately and publish its own structure/PAE/metrics.

    Primary columns/files describe the first model exactly. The separate
    all_models_pass decision requires every requested model to pass the same
    existing gate. No averaged score is attached to a single model's structure.
    A receipt is published only after every requested model has completed.
    """
    from run_state import sha256, stage_seed
    names = model_names(models)
    if recycles < 0:
        raise ValueError('Validation recycles must be nonnegative')
    available = list(predictor._model_names)
    if any(n not in available for n in names):
        raise ValueError(f'Requested model weights unavailable: requested={names}, loaded={available}')
    folder = Path(folder); (folder / 'designs').mkdir(parents=True, exist_ok=True)
    receipt_path = folder / 'designs' / f'{candidate}.validation.json'
    paths = [folder/'designs'/(model_stem(candidate,n,names[0])+ext)
             for n in names for ext in ('.pdb','.pickle')]
    if receipt_path.exists() or any(p.exists() for p in paths):
        raise FileExistsError(f'Refusing to overwrite validation artifacts for {candidate}')
    report = dict(schema=1, sequence=sequence, primary_model=names[0], model_order=names,
                  policy='all_models_pass', models={})
    for index, name in enumerate(names):
        # Preserve the historical first-model seed. Additional models get stable
        # separate streams, independent of execution order or RNG consumption.
        model_seed = seed if index == 0 else stage_seed(seed, name)
        predictor.set_seq(sequence[-binder_len:])
        started = time.perf_counter()
        predictor.predict(num_recycles=recycles, verbose=False, models=[name],
                          num_models=1, sample_models=False, seed=model_seed)
        prediction_seconds = time.perf_counter() - started
        log = predictor.aux['log']
        executed = [available[int(n)] for n in log['models']]
        if executed != [name]:
            raise RuntimeError(f'Model dispatch mismatch: requested {name}, executed {executed}')
        metrics = {key:float(log[key]) for key in ('plddt','i_pae','i_ptm','cmap_loss_binder')}
        if not all(math.isfinite(v) for v in metrics.values()):
            raise ValueError(f'{name}: nonfinite validation metrics')
        pae = np.asarray(predictor.aux['all']['pae'])
        expected_len = len(sequence.replace('/', ''))
        if pae.shape != (1, expected_len, expected_len) or not np.isfinite(pae).all() or (pae < 0).any():
            raise ValueError(f'{name}: invalid PAE shape/values; expected one {expected_len}x{expected_len} matrix')
        stem = folder / 'designs' / model_stem(candidate, name, names[0])
        pdb_path, pickle_path = str(stem)+'.pdb', str(stem)+'.pickle'
        atomic_write(pdb_path, lambda path: predictor.save_pdb(path, get_best=False))
        def dump(path):
            with open(path, 'wb') as stream:
                pickle.dump(predictor.aux['all'], stream, protocol=pickle.HIGHEST_PROTOCOL)
        atomic_write(pickle_path, dump)
        report['models'][name] = dict(metrics=metrics, seed=model_seed, prediction_seconds=prediction_seconds,
            requested_recycles=recycles, actual_recycles=int(log['recycles']),
            executed_models=executed, passed=bool(passes(metrics)),
            pdb_sha256=sha256(pdb_path), pickle_sha256=sha256(pickle_path))
    report['all_models_pass'] = all(r['passed'] for r in report['models'].values())
    report['model_disagreement'] = len({r['passed'] for r in report['models'].values()}) > 1
    write_json(receipt_path, report)
    return report


def model_artifacts(folder, candidate):
    """Enumerate model-specific PDB/PAE pairs; legacy runs have one unnamed model."""
    root = Path(folder)
    receipt = root/'designs'/f'{candidate}.validation.json'
    if receipt.exists():
        candidate_files(root, candidate, require_receipt=True)
        data = json.loads(receipt.read_text())
        primary = data['primary_model']
        names = data['model_order']
    else:
        names, primary = ['legacy_primary'], 'legacy_primary'
    return [(model, root/'designs'/(model_stem(candidate,model,primary)+'.pdb'),
             root/'designs'/(model_stem(candidate,model,primary)+'.pickle')) for model in names]


def verify_receipt(folder, candidate, hashes, row=None):
    receipt = Path(folder)/'designs'/f'{candidate}.validation.json'
    if not receipt.exists():
        return
    data = json.loads(receipt.read_text())
    if row is not None:
        if row.get('sequence') != data['sequence'] or row.get('primary_model') != data['primary_model']:
            raise ValueError('CSV and receipt sequence/model identity mismatch')
        primary = data['models'][data['primary_model']]['metrics']
        for column, key in [('plddt','plddt'),('iptm','i_ptm'),('ipae','i_pae'),('cmap_loss','cmap_loss_binder')]:
            if column in row and not math.isclose(float(row[column]),primary[key],rel_tol=0,abs_tol=1e-10):
                raise ValueError('CSV metrics do not describe the primary model')
        if str(row.get('validation_pass')).lower() != str(data['all_models_pass']).lower():
            raise ValueError('CSV and receipt acceptance disagree')
    for model, pdb, pk in model_artifacts(folder,candidate):
        record = data['models'][model]
        if record['executed_models'] != [model]:
            raise ValueError('Receipt model identity mismatch')
        for path, key in [(pdb,'pdb_sha256'),(pk,'pickle_sha256')]:
            if hashes.get(str(path.relative_to(folder))) != record[key]:
                raise ValueError(f'Receipt artifact hash mismatch: {path}')
