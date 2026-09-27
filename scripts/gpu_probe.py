"""Explicit GPU runtime probe. Never imported by CPU preflight or tests."""
import argparse
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline.result_io import write_json
from run_state import sha256


def checkpoint_paths(root):
    root = Path(root)
    found = {}
    for model in ('model_1_ptm','model_2_ptm'):
        candidates = [root/'params'/f'params_{model}.npz', root/f'params_{model}.npz',
                      root/'params'/f'{model}.npz', root/f'{model}.npz']
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing checkpoint {model} under {root}')
        found[model] = path
    return found


def probe(data_dir):
    # Called only after explicit execution of this program, never during planning.
    weights = checkpoint_paths(data_dir)
    import jax
    import jax.numpy as jnp
    devices = [d for d in jax.devices() if d.platform == 'gpu']
    if not devices:
        raise RuntimeError('No JAX GPU backend; CPU fallback is not a valid GPU test')
    with jax.default_device(devices[0]):
        for dtype in (jnp.float32,jnp.bfloat16):
            x=jnp.ones((32,32),dtype=dtype)
            result=jax.jit(lambda a:a@a)(x)
            result.block_until_ready()
            if not bool(jnp.all(result == 32)):
                raise RuntimeError(f'GPU matrix multiplication failed for {dtype}')
        gradient=jax.jit(jax.grad(lambda x:jnp.sum(x*x)))(jnp.ones((32,)))
        gradient.block_until_ready()
        if not bool(jnp.all(gradient == 2)):
            raise RuntimeError('GPU autodiff failed')
    packages={}
    for package in ('jax','jaxlib','colabdesign','numpy','openmm','pdbfixer'):
        try: packages[package]=version(package)
        except PackageNotFoundError: pass
    smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used',
                        '--format=csv,noheader'],capture_output=True,text=True,check=True)
    return dict(status='passed',platform=platform.platform(),python=sys.version,packages=packages,
                devices=[str(d) for d in devices],nvidia_smi=smi.stdout.strip(),
                checkpoints={name:dict(path=str(p.resolve()),sha256=sha256(p)) for name,p in weights.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    try:
        report=probe(args.data_dir)
    except Exception as exc:
        write_json(args.report,dict(status='failed',error=f'{type(exc).__name__}: {exc}'))
        raise
    write_json(args.report,report)


if __name__=='__main__':main()
