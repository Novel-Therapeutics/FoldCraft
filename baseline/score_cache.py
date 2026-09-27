"""Provenance-keyed scorer sidecars and serialized publication to results.csv."""
from pathlib import Path
from contextlib import ExitStack
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import math
import sys

try:
    from .result_io import file_lock, write_json, publish_columns
except ImportError:
    from result_io import file_lock, write_json, publish_columns


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class ScoreSession:
    """Only reuse sidecar records with identical inputs, code, settings and versions.

    A separate scorer lock prevents duplicate/stale writes by concurrent copies
    of the same scorer; the CSV lock merges different scorers safely. Historical
    CSV-only scores have no verifiable provenance and are recomputed on invocation.
    """
    def __init__(self, csv_path, source, columns, settings, extra_inputs=(), structure=True):
        self.csv = Path(csv_path)
        self.columns = columns
        self.structure = structure
        self.path = self.csv.with_name(Path(source).stem + '.scores.json')
        versions = {}
        for name in ('numpy','pandas','biopython','jax','jaxlib','colabdesign','torch','transformers','boltz','openmm','pdbfixer'):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                pass
        self.extra_inputs = list(extra_inputs)
        self.protocol = dict(settings=settings, source=digest(source), cache_code=digest(__file__),
                             helper_code=digest(Path(__file__).with_name('structure_checks.py')),
                             python=sys.version, packages=versions,
                             extra_inputs={str(Path(p).resolve()):digest(p) for p in extra_inputs})
        self.stack = ExitStack()

    def __enter__(self):
        self.stack.enter_context(file_lock(self.path))
        try:
            self.cache = json.loads(self.path.read_text()).get('records', {}) if self.path.exists() else {}
        except (ValueError, OSError):
            self.cache = {}
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def prepare(self, frame):
        import pandas as pd
        self.signatures = {}
        self.input_hashes = {Path(p):digest(p) for p in self.extra_inputs}
        frame = frame.copy()
        for col in self.columns:
            frame[col] = float('nan')
        if frame['name'].isna().any() or frame['name'].duplicated().any():
            raise ValueError('Scoring requires unique candidate names')
        for idx, row in frame.iterrows():
            name = str(row['name'])
            if Path(name).name != name:
                raise ValueError('Candidate names must be basenames')
            sequence = row.get('sequence')
            payload = dict(protocol=self.protocol, name=name, sequence=None if pd.isna(sequence) else sequence)
            if self.structure:
                path = self.csv.parent/'designs'/(name+'.pdb')
                payload['structure'] = digest(path)
                self.input_hashes[path] = payload['structure']
            signature = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
            self.signatures[name] = signature
            cached = self.cache.get(name, {})
            if cached.get('signature') == signature:
                for col, value in cached.get('values', {}).items():
                    if col in self.columns and value is not None:
                        frame.loc[idx,col] = value
        return frame

    def publish(self, frame):
        import pandas as pd
        if any(digest(p) != expected for p, expected in self.input_hashes.items()):
            raise ValueError('Scorer inputs changed while scoring')
        records = {}
        for _, row in frame.iterrows():
            name = str(row['name'])
            values = {c:None if pd.isna(row[c]) else float(row[c]) for c in self.columns}
            if any(v is not None and not math.isfinite(v) for v in values.values()):
                raise ValueError(f'{name}: nonfinite score; result remains unknown')
            records[name] = dict(signature=self.signatures[name], values=values,
                                 status='complete' if all(v is not None for v in values.values()) else 'incomplete')
        write_json(self.path, dict(protocol=self.protocol, records=records))
        # Also clear invalidated values for unscored rows, never leave old settings
        # masquerading as results from this protocol.
        publish_columns(self.csv, frame, self.columns, include_missing=True)
