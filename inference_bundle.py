"""Content identity for the files used by the supported FoldCraft design path.

Discovery uses filesystem/package metadata only: no JAX, CUDA or ColabDesign
imports. The scheduler invokes this helper with its selected worker interpreter.
"""
import argparse
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path

from baseline.checkpoint_files import af2_checkpoint
from run_state import sha256

AF2_TEMPLATE_MODELS = ('model_1_ptm', 'model_2_ptm')
MPNN_MODEL_NAME = 'v_48_010'


def colabdesign_root():
    spec = importlib.util.find_spec('colabdesign')
    if spec is None or not spec.origin:
        raise ValueError('ColabDesign is not installed in the selected Python environment')
    return Path(spec.origin).resolve().parent


def file_state(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def resolve_files(request, package_root):
    root, package_root = Path(request['repo_root']), Path(package_root)
    files = {}
    # All main-path AF2 constructors use templates and attempt these two models.
    # Missing optional models are tracked through the effective set, so adding
    # a previously absent checkpoint changes the bundle too.
    for name in AF2_TEMPLATE_MODELS:
        try:
            files['af2:'+name] = af2_checkpoint(request['data_dir'], name)
        except FileNotFoundError:
            if name in request['validation_models']:
                raise
    if not any(k.startswith('af2:') for k in files):
        raise ValueError('No template-capable AF2 checkpoint is available')
    weights = {'soluble':'weights_soluble', 'original':'weights'}[request['mpnn_weight']]
    files['mpnn:'+MPNN_MODEL_NAME] = package_root/'mpnn'/weights/(MPNN_MODEL_NAME+'.pkl')
    for name in ('__init__.py', 'af/model.py', 'af/alphafold/model/data.py',
                 'mpnn/model.py', f'mpnn/{weights}/__init__.py'):
        files['loader:'+name] = package_root/name
    if request['vhh']:
        files['conditioning:vhh'] = root/'framework/vhh.npy'
    if any(not p.is_file() for p in files.values()):
        raise FileNotFoundError('Missing inference file: '+', '.join(str(p) for p in files.values() if not p.is_file()))
    return files


def capture_bundle(data_dir, *, mpnn_weight='soluble', vhh=False,
                   validation_models='model_1_ptm', repo_root=None):
    from model_validation import model_names
    request = dict(data_dir=str(Path(data_dir).expanduser().resolve()),
                   mpnn_weight=mpnn_weight, vhh=bool(vhh),
                   validation_models=model_names(validation_models),
                   repo_root=str(Path(repo_root or Path(__file__).parent).resolve()))
    if mpnn_weight not in ('original','soluble'):
        raise ValueError('Unknown MPNN weight variant')
    package = colabdesign_root()
    files = resolve_files(request, package)
    identity = dict(schema=1, request=request, package_root=str(package), files={}, packages={})
    observed = {}
    for label, path in sorted(files.items()):
        before = file_state(path)
        identity['files'][label] = dict(path=str(path), resolved_path=str(path.resolve()),
                                       sha256=sha256(path), size=before[2])
        if file_state(path) != before:
            raise ValueError('Inference file changed while hashing: '+str(path))
        observed[label] = before
    for name in ('colabdesign','jax','jaxlib','numpy','dm-haiku','optax','chex'):
        try:
            identity['packages'][name] = version(name)
        except PackageNotFoundError:
            pass
    bundle = dict(identity=identity, observed=observed)
    validate_snapshot(bundle)  # also detect a lookup/file-set change during capture
    return bundle


def validate_snapshot(bundle, *, full=False, worker_environment=False):
    """Reject mutation, new lookup winners and changed effective model sets.

    Fast checks are safe only within this process/campaign snapshot. Fresh runs
    always hash content again. Stat observations are not scientific identity.
    """
    identity = bundle['identity']
    if identity['schema'] != 1:
        raise ValueError('Unsupported inference bundle schema')
    package = colabdesign_root() if worker_environment else Path(identity['package_root'])
    if str(package) != identity['package_root']:
        raise ValueError('Worker ColabDesign environment differs from the planned bundle')
    current = resolve_files(identity['request'], package)
    if set(current) != set(identity['files']):
        raise ValueError('Effective inference file set changed')
    for label, path in current.items():
        saved = identity['files'][label]
        if (str(path) != saved['path'] or str(path.resolve()) != saved['resolved_path'] or
                file_state(path) != bundle['observed'][label]):
            raise ValueError('Inference file changed since capture: '+label)
        if full and sha256(path) != saved['sha256']:
            raise ValueError('Inference file contents changed: '+label)
        if file_state(path) != bundle['observed'][label]:
            raise ValueError('Inference file changed during verification: '+label)


def guarded_load(bundle, factory, *args, **kwargs):
    if bundle:
        validate_snapshot(bundle)
    model = factory(*args, **kwargs)
    if bundle:
        validate_snapshot(bundle)
    if 'model_names' in kwargs and list(model._model_names) != list(kwargs['model_names']):
        raise ValueError('AF2 loader did not load the fingerprinted model set')
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--mpnn-weight', choices=['original','soluble'], default='soluble')
    p.add_argument('--vhh', action='store_true')
    p.add_argument('--validation-models', default='model_1_ptm')
    args = p.parse_args()
    print(json.dumps(capture_bundle(args.data_dir, mpnn_weight=args.mpnn_weight,
                                  vhh=args.vhh, validation_models=args.validation_models)))


if __name__ == '__main__':
    main()
