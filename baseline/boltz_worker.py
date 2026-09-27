"""Persistent, isolated Boltz CLI processes; model initialization remains unchanged.

Each worker executes one command at a time and retains imported modules only.
The original CLI reseeds and rebuilds its model/trainer for every request. Keeping
that lifecycle avoids silently changing diffusion RNG when model loading is skipped.
"""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import gc
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
import traceback


class _RequestStream:
    """A stable stream identity for log handlers retained across CLI calls."""
    def __init__(self, fallback):
        self.fallback = fallback
        self.target = None

    def write(self, text):
        return (self.target if self.target is not None else self.fallback).write(text)

    def flush(self):
        return (self.target if self.target is not None else self.fallback).flush()

    def __getattr__(self, name):
        return getattr(self.target if self.target is not None else self.fallback, name)


_stdout = _RequestStream(sys.stdout)
_stderr = _RequestStream(sys.stderr)


def _predict(args, log_path):
    # A spawned service worker otherwise forces its data-loader children to use
    # spawn too. Restore the platform's normal context, as in a fresh CLI process
    # (fork on the validated Linux/Python 3.12 runtime). The service itself is
    # always spawned; this setting is confined to that isolated child process.
    multiprocessing.set_start_method(multiprocessing.get_all_start_methods()[0], force=True)
    began = time.monotonic()
    code = 0
    with open(log_path, 'w') as log, redirect_stdout(_stdout), redirect_stderr(_stderr):
        _stdout.target = _stderr.target = log
        try:
            from boltz.main import cli
            cli.main(args=args, standalone_mode=False)
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            # Lightning trainers form cycles. Release request-owned models and
            # CUDA allocations before the next request, including failed requests.
            gc.collect()
            torch = sys.modules.get('torch')
            if torch is not None and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            _stdout.target = _stderr.target = None
    return code, time.monotonic() - began


class BoltzWorkerPool:
    """Spawn workers (never fork an initialized CUDA runtime). Close explicitly."""
    def __init__(self, workers=1):
        if workers < 1:
            raise ValueError('workers must be positive')
        self.executor = ProcessPoolExecutor(max_workers=workers,
                                            mp_context=multiprocessing.get_context('spawn'))

    def run(self, command, log_path):
        if command[1:4] != ['-m', 'boltz.main', 'predict']:
            raise ValueError('Worker accepts only boltz.main predict commands')
        code, seconds = self.executor.submit(_predict, command[3:], str(log_path)).result()
        # Read only on failure; successful logs remain alongside model artifacts.
        error = Path(log_path).read_text()[-4000:] if code else ''
        result = subprocess.CompletedProcess(command, code, stdout='', stderr=error)
        result.worker_seconds = seconds
        return result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.executor.shutdown(wait=True, cancel_futures=True)
