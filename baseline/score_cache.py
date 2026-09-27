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
    def __init__(self, csv_path, source, columns, settings, extra_inputs=(), structure=True, model_inputs=None):
        self.csv = Path(csv_path)
        self.columns = columns
        self.structure = structure
        self.path = self.csv.with_name(Path(source).stem + '.scores.json')
        versions = {}
        for name in ('numpy','pandas','biopython','jax','jaxlib','colabdesign','torch','transformers','boltz','openmm','pdbfixer','rdkit','scipy','huggingface-hub','pytorch-lightning','dm-haiku','chex','optax'):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                pass
        self.extra_inputs = list(extra_inputs)
        self.protocol = dict(settings=settings, source=digest(source), cache_code=digest(__file__),
                             helper_code={name:digest(Path(__file__).with_name(name)) for name in ('structure_checks.py','checkpoint_files.py','gates.py')},
                             python=sys.version, packages=versions,
                             extra_inputs={str(Path(p).resolve()):digest(p) for p in extra_inputs})
        self.model_files = {}
        model_manifest = {}
        for label, location in sorted((model_inputs or {}).items()):
            path = Path(location).expanduser().resolve()
            paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
            if not paths or any(not p.is_file() for p in paths):
                raise ValueError(f'Missing model input: {label}={path}')
            model_manifest[label] = dict(path=str(path), files={})
            for item in paths:
                key = str(item.relative_to(path)) if path.is_dir() else item.name
                before = self.file_identity(item)
                model_manifest[label]['files'][key] = digest(item)
                if self.file_identity(item) != before:
                    raise ValueError('Model input changed while hashing')
                self.model_files[item] = before
        self.protocol['model_inputs'] = model_manifest
        self.model_roots = {Path(p).expanduser().resolve() for p in (model_inputs or {}).values() if Path(p).expanduser().is_dir()}
        self.diagnostics = {}
        self.stack = ExitStack()

    @staticmethod
    def file_identity(path):
        st = path.stat()
        return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def annotate(self, name, **record):
        """Persist QC or failure reasons alongside the numeric result."""
        self.diagnostics[str(name)] = record

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
        self.diagnostics = {}
        self.input_hashes = {Path(p):digest(p) for p in self.extra_inputs}
        if any(value != self.protocol['extra_inputs'][str(p.resolve())] for p,value in self.input_hashes.items()):
            raise ValueError('Scorer inputs changed before scoring')
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
                self.diagnostics[name] = cached.get('diagnostics', {})
                for col, value in cached.get('values', {}).items():
                    if col in self.columns and value is not None:
                        frame.loc[idx,col] = value
        return frame

    def publish(self, frame):
        import pandas as pd
        if any(digest(p) != expected for p, expected in self.input_hashes.items()):
            raise ValueError('Scorer inputs changed while scoring')
        if any(not p.is_file() or self.file_identity(p) != identity for p, identity in self.model_files.items()):
            raise ValueError('Model inputs changed while scoring')
        for root in self.model_roots:
            if {p for p in root.rglob('*') if p.is_file()} != {p for p in self.model_files if p.is_relative_to(root)}:
                raise ValueError('Model input files changed while scoring')
        records = {}
        for _, row in frame.iterrows():
            name = str(row['name'])
            values = {c:None if pd.isna(row[c]) else float(row[c]) for c in self.columns}
            if any(v is not None and not math.isfinite(v) for v in values.values()):
                raise ValueError(f'{name}: nonfinite score; result remains unknown')
            records[name] = dict(signature=self.signatures[name], values=values,
                                 status='complete' if all(v is not None for v in values.values()) else 'incomplete',
                                 diagnostics=self.diagnostics.get(name, {}))
        write_json(self.path, dict(protocol=self.protocol, records=records))
        # Also clear invalidated values for unscored rows, never leave old settings
        # masquerading as results from this protocol.
        publish_columns(self.csv, frame, self.columns, include_missing=True)
