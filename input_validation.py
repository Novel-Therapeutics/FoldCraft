"""CPU-only input contracts. Native PDB numbering is mapped before inference."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import re

import numpy as np

STANDARD = set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
VHH_HOTSPOTS = '26-35,55-59,102-116'
VHH_CONVENTION = 'foldcraft-127-current-v1'


def residue_range(value, *, allow_empty=False):
    """Positive inclusive ranges; preserve order and reject ambiguous duplicates."""
    if not isinstance(value, str):
        raise ValueError('Residue selection must be a string')
    if not value.strip():
        if allow_empty:
            return []
        raise ValueError('Residue selection must not be empty')
    result = []
    seen = set()
    for item in value.split(','):
        match = re.fullmatch(r'\s*([0-9]+)\s*(?:-\s*([0-9]+)\s*)?', item)
        if not match:
            raise ValueError(f'Invalid residue selection: {item!r}')
        lo = int(match[1]); hi = int(match[2] or match[1])
        if lo < 1 or hi < lo or hi > 9999:
            raise ValueError(f'Invalid PDB residue range {item!r}: expected 1 <= start <= end <= 9999')
        for r in range(lo, hi + 1):
            if r in seen:
                raise ValueError(f'Duplicate residue {r} in selection')
            seen.add(r); result.append(r)
    return result


def positions(value, length, *, allow_empty=False):
    chosen = residue_range(value, allow_empty=allow_empty)
    if any(r > length for r in chosen):
        raise ValueError(f'Residue selection {value!r} exceeds chain length {length}')
    return chosen


@dataclass(frozen=True)
class PreparedChain:
    source: str
    source_sha256: str
    chain: str
    residue_ids: tuple
    pdb: str

    @property
    def length(self):
        return len(self.residue_ids)

    def selection(self, value, *, allow_empty=False):
        selected = residue_range(value, allow_empty=allow_empty)
        mapping = {native: i + 1 for i, native in enumerate(self.residue_ids)}
        missing = [r for r in selected if r not in mapping]
        if missing:
            raise ValueError(f'{self.source}, chain {self.chain}: residues {missing} are absent')
        return ','.join(str(mapping[r]) for r in selected)

    def manifest(self):
        return dict(source=self.source, source_sha256=self.source_sha256,
                    chain=self.chain, residues=[dict(original=r, model_position=i + 1)
                                               for i, r in enumerate(self.residue_ids)],
                    alternate_location_policy='first atom record, matching reviewed ColabDesign')


def read_chain(filename, chain):
    """Normalize one complete, continuous protein chain to positions 1..N.

    Numbering offsets are supported. Gaps, insertion codes and incomplete
    backbones fail early rather than being interpreted differently by models.
    MSE is retained for ColabDesign's native modified-residue conversion.
    """
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f'PDB not found: {path}')
    if not isinstance(chain, str) or len(chain) != 1:
        raise ValueError('Select exactly one PDB chain identifier')
    raw = path.read_bytes()
    lines = raw.decode('utf-8').splitlines()
    if sum(line.startswith('MODEL ') for line in lines) > 1:
        raise ValueError(f'{path}: select a single PDB MODEL before design')
    groups = {}
    records = []
    for line in lines:
        if not line.startswith(('ATOM  ', 'HETATM')) or line[21:22] != chain:
            continue
        name = line[17:20].strip()
        if line.startswith('HETATM') and name != 'MSE':
            continue
        if name not in STANDARD | {'MSE'}:
            raise ValueError(f'{path}: unsupported polymer residue {name!r}; prepare a canonical protein PDB')
        if line[26:27].strip():
            raise ValueError(f'{path}: insertion codes require explicit preprocessing and residue mapping')
        try:
            native = int(line[22:26]); xyz = [float(line[a:b]) for a, b in [(30,38),(38,46),(46,54)]]
        except ValueError as exc:
            raise ValueError(f'{path}: malformed PDB atom record') from exc
        if native < 1 or not all(math.isfinite(v) for v in xyz):
            raise ValueError(f'{path}: nonpositive residue numbering or nonfinite coordinates')
        if native not in groups:
            groups[native] = (name, set())
        previous_name, atoms = groups[native]
        if previous_name != name:
            raise ValueError(f'{path}: ambiguous residue identity at {chain}{native}')
        atom = line[12:16].strip()
        if atom in atoms:  # match dependency's first-record alternate selection
            continue
        atoms.add(atom)
        records.append((native, line))
    ids = tuple(groups)
    if not ids:
        raise ValueError(f'{path}: no supported protein residues in chain {chain!r}')
    if ids != tuple(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f'{path}: chain {chain} has gaps or unordered residue IDs; prepare a continuous chain and update selections')
    for r, (_, atoms) in groups.items():
        missing = {'N','CA','C','O'} - atoms
        if missing:
            raise ValueError(f'{path}: residue {chain}{r} lacks backbone atoms {sorted(missing)}')
    mapping = {r:i+1 for i,r in enumerate(ids)}
    output = [line[:16] + ' ' + line[17:22] + f'{mapping[r]:4d}' + ' ' + line[27:]
              for r, line in records]
    return PreparedChain(str(path), hashlib.sha256(raw).hexdigest(), chain, ids,
                         '\n'.join(output + ['TER', 'END', '']))


def validate_controls(args):
    from model_validation import model_names
    model_names(args.validation_models)
    if args.validation_recycles < 0:
        raise ValueError("--validation_recycles must be nonnegative")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        raise ValueError('--seed must be a 32-bit nonnegative integer')
    for name in ('num_designs', 'target_success', 'mpnn_samples', 'max_trajectories'):
        if getattr(args, name) < 1:
            raise ValueError(f'--{name} must be positive')
    for name in ('mpnn_sampling_temp', 'mpnn_backbone_noise'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == 'mpnn_sampling_temp' and value == 0):
            raise ValueError(f'--{name} must be finite and {"positive" if name.endswith("temp") else "nonnegative"}')
    try:
        stages = [int(s) for s in args.design_stages.split(',')]
    except ValueError as exc:
        raise ValueError('--design_stages requires three positive integers') from exc
    if len(stages) != 3 or any(s < 1 for s in stages):
        raise ValueError('--design_stages requires three positive integers')
    if not args.vhh and not args.binder_template:
        raise ValueError('--binder_template is required unless --vhh is set')
    return args


def preflight(args):
    validate_controls(args)
    target = read_chain(args.target_template, args.target_chain)
    target_hotspots = target.selection(args.target_hotspots)
    if args.vhh:
        validate_cmap(np.load(Path(__file__).resolve().parent / 'framework' / 'vhh.npy', allow_pickle=False), size=127)
        binder = None
        hotspots, mask = VHH_HOTSPOTS, ''
    else:
        binder = read_chain(args.binder_template, args.binder_chain)
        hotspots = binder.selection(args.binder_hotspots, allow_empty=True)
        mask = binder.selection(args.binder_mask, allow_empty=True)
    return target, binder, target_hotspots, hotspots, mask


def validate_cmap(value, *, size=None):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or array.shape[0] < 1:
        raise ValueError('Contact map must be a nonempty square matrix')
    if size is not None and array.shape != (size, size):
        raise ValueError(f'Contact map shape {array.shape} must equal ({size}, {size})')
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError('Contact map must contain real probabilities')
    # Softmax sums in float32 (including the bundled VHH map) can exceed one
    # by a few ulps. Preserve those values to avoid changing historical maps.
    tolerance = min(1e-6, 4 * np.finfo(array.dtype).eps) if np.issubdtype(array.dtype, np.floating) else 0
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1 + tolerance).any():
        raise ValueError('Contact probabilities must be finite and in [0, 1]')
    return array
