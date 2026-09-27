"""POSIX process supervision for the production CLI (no GPU imports)."""
import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DEFAULT_TIMEOUT_MINUTES = 360.


@dataclass
class Outcome:
    returncode: int
    reason: str
    seconds: float


class Interrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def terminate_group(process, grace_seconds=5.):
    """Terminate the new session we created, including children of an exited leader."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before a stubborn descendant. Give the whole group
    # the bounded grace period, then kill remaining members without killpg(0)
    # liveness probes (not portable across supported POSIX environments).
    time.sleep(max(0., deadline-time.monotonic()))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_owned(command, timeout_seconds, *, grace_seconds=5.):
    if os.name != 'posix':
        raise RuntimeError('Production supervision requires a POSIX host (Linux/macOS)')
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('Timeout must be finite and positive')
    started = time.monotonic()
    previous = {}
    process = None
    def interrupt(signum, frame):
        raise Interrupted(signum)
    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, interrupt)
        process = subprocess.Popen(command, start_new_session=True)
        code = process.wait(timeout=timeout_seconds)
        reason = 'exited'
    except subprocess.TimeoutExpired:
        code, reason = 124, 'timed_out'
    except Interrupted as exc:
        code, reason = 128 + exc.signum, 'interrupted'
    finally:
        # A second interruption must not strand the worker during cleanup.
        for signum in previous:
            signal.signal(signum, signal.SIG_IGN)
        try:
            if process is not None:
                terminate_group(process, grace_seconds)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    return Outcome(code if code >= 0 else 128-code, reason, time.monotonic()-started)


def supervise(args):
    """Run inference from the already-created manifest and canonical input files."""
    import json
    from run_state import completed_run, finish_run
    folder = Path(args.output_folder).resolve()
    result = run_owned([sys.executable, '-u', str(Path(__file__).resolve()), str(folder)],
                       args.timeout_minutes * 60)
    state = json.loads((folder/'run.json').read_text())
    state['supervision'] = dict(timeout_minutes=args.timeout_minutes,
                               elapsed_seconds=result.seconds, returncode=result.returncode,
                               reason=result.reason)
    if result.reason != 'exited':
        state['error'] = f'Worker {result.reason}; owned process group terminated'
        finish_run(folder, state, status=result.reason)
    elif result.returncode:
        if state['status'] not in ('failed', 'exhausted'):
            state['error'] = f'Worker exited with code {result.returncode}'
            finish_run(folder, state, status='failed')
        else:
            finish_run(folder, state, status=state['status'])
    elif not completed_run(folder):
        state['error'] = 'Worker exited successfully without a valid completed run'
        finish_run(folder, state, status='failed')
        result.returncode = 1
    else:
        finish_run(folder, state)
    if result.returncode:
        raise SystemExit(result.returncode)


def worker(folder):
    import json
    from FoldCraft import execute
    from run_state import finish_run
    state = json.loads((Path(folder)/'run.json').read_text())
    if state['status'] != 'running':
        raise ValueError('Worker requires a newly started run')
    args = argparse.Namespace(**state['config'])
    mapped = state['mapped_hotspots']
    # execute consumes canonical files from inputs/ and these mapped selections;
    # original source paths are never reopened by the worker.
    prepared = (None, None, mapped['target'], mapped['binder'], mapped['binder_mask'])
    try:
        from inference_bundle import validate_snapshot
        if not state.get('inference_bundle'):
            raise ValueError('Production worker requires an inference bundle')
        validate_snapshot(state['inference_bundle'], full=True, worker_environment=True)
        execute(args, prepared, state)
    except BaseException as exc:
        current = json.loads((Path(folder)/'run.json').read_text())
        if current['status'] == 'running':
            current['error'] = f'{type(exc).__name__}: {exc}'
            finish_run(folder, current, status='failed')
        raise


if __name__ == '__main__':
    worker(sys.argv[1])
