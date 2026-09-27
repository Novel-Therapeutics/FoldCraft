"""Atomic publication and serialized scorer updates (no model dependencies)."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import tempfile


def atomic_write(path, writer):
    """Keep the previous file until its replacement has been fully serialized."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    def write(tmp):
        with open(tmp, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    atomic_write(path, write)


@contextmanager
def file_lock(path):
    """Protect local/shared POSIX filesystem writers; no stale PID lock files."""
    with open(str(path) + '.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def publish_columns(csv_path, frame, columns, *, include_missing=False):
    """Re-read under a lock and merge only this scorer's columns by stable ID.

    Atomic replacement alone does not prevent lost updates from other scorers.
    Never overwrite immutable names/sequences using an old in-memory snapshot.
    """
    import pandas as pd
    with file_lock(csv_path):
        current = pd.read_csv(csv_path)
        if frame['name'].duplicated().any() or current['name'].duplicated().any():
            raise ValueError('Candidate names must be unique')
        if set(current['name']) != set(frame['name']):
            raise ValueError('Candidate set changed while scoring; use a new run')
        old = frame.set_index('name'); new = current.set_index('name')
        if 'sequence' in frame and 'sequence' in current and not old['sequence'].reindex(new.index).equals(new['sequence']):
            raise ValueError('Candidate sequences changed while scoring')
        for col in columns:
            if col in old:
                if col not in new:
                    new[col] = float('nan')
                valid = pd.Series(True, index=old.index) if include_missing else old[col].notna()
                new.loc[old.index[valid], col] = old.loc[valid, col]
        atomic_write(csv_path, lambda p: new.reset_index().to_csv(p, index=False))
