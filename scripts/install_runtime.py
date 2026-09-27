"""Install the validated Linux x86_64 / Python 3.12 GPU runtime in a fresh venv.

Requires existing official AF2 model_1_ptm/model_2_ptm weights. Never downloads
weights, replaces an environment, or installs the experimental BindCraft stack.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_state import sha256
from baseline.result_io import write_json
from scripts.gpu_probe import checkpoint_paths

WEIGHTS = {
    'model_1_ptm': '5e564f79af5bcd54ccef6e2a6bb0ff01015d01650ebc41d4575e35f0de9ecc84',
    'model_2_ptm': '23645d9a82c4af2ed54cd48a7b3c1c2575dc6aa9fe931adb4d7203ca5f0dc398',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', type=Path, default=ROOT/'.venv')
    parser.add_argument('--python', default='python3.12', help='Python 3.12 interpreter used to create the venv')
    parser.add_argument('--data-dir', type=Path, required=True, help='Existing official AF2 weights or parent containing params/')
    args = parser.parse_args(argv)
    destination = args.env.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        parser.error('Environment path already exists; choose a new path')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('This validated runtime profile supports Linux x86_64 only')
    runtime = json.loads(subprocess.check_output([args.python, '-c',
        'import sys,json;print(json.dumps(list(sys.version_info[:2])))'], text=True))
    if runtime != [3, 12]:
        parser.error('--python must select Python 3.12')
    args.data_dir = args.data_dir.expanduser().resolve()
    weights = checkpoint_paths(args.data_dir)
    if any(sha256(path) != WEIGHTS[model] for model, path in weights.items()):
        parser.error('AF2 checkpoint hashes do not match the validated official weights')
    lock = ROOT/'requirements/bizon-linux-py312.txt'
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(status='installing', requirements_sha256=sha256(lock),
                  environment=str(destination), checkpoints=WEIGHTS)
    report_path = destination/'foldcraft-install.json'
    write_json(report_path, report)
    try:
        subprocess.run([args.python, '-m', 'venv', str(destination)], check=True)
        python = str(destination/'bin/python')
        subprocess.run([python, '-m', 'pip', 'install', '-r', str(lock)], check=True)
        subprocess.run([python, '-m', 'pip', 'check'], check=True)
        subprocess.run([python, '-c', 'import colabdesign; import jax; import haiku; import optax'], check=True)
        subprocess.run([python, str(ROOT/'scripts/gpu_probe.py'), '--data-dir', str(args.data_dir.resolve()),
                        '--report', str(destination/'foldcraft-gpu.json')], check=True,
                       env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false'))
        freeze = subprocess.check_output([python, '-m', 'pip', 'freeze'], text=True)
        (destination/'foldcraft-freeze.txt').write_text(freeze)
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(report_path, report)
    print(f'Runtime ready: {destination}/bin/python. See {report_path} and foldcraft-gpu.json.')
    print('A GPU probe is not a pipeline smoke test; run scripts/gpu_smoke.py before a campaign.')


if __name__ == '__main__':
    main()
