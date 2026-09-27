"""Persist actual optimization metrics and model dispatch without changing inference."""
from pathlib import Path
import math
import numpy as np
from baseline.result_io import write_json

PROTOCOL = 'three-stage-log-v1'


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite optimization metric')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f'Unsupported optimization metric type: {type(value).__name__}')


def save_history(folder, trajectory, model, stages, seed):
    if not trajectory or Path(trajectory).name != trajectory:
        raise ValueError('Trajectory must be a basename')
    logs = model._tmp['log']
    if len(stages) != 3 or min(stages) < 1 or len(logs) != sum(stages):
        raise ValueError('Optimization history does not cover every requested iteration')
    records=[]
    offset=0
    for stage, count in zip(('logits', 'temperature', 'hard'), stages):
        for i, raw in enumerate(logs[offset:offset+count], 1):
            log=plain(raw)
            required=('loss','plddt','i_pae','i_ptm','cmap_loss_binder','models','recycles')
            if any(k not in log for k in required):
                raise ValueError('Incomplete optimization metrics or model dispatch')
            indices=log['models']
            if not indices or any(type(n) is not int or n < 0 or n >= len(model._model_names) for n in indices):
                raise ValueError('Invalid optimization model dispatch')
            records.append(dict(iteration=offset+i, stage=stage, stage_iteration=i,
                                model_names=[model._model_names[n] for n in indices], metrics=log))
        offset += count
    path=Path(folder)/'optimization'/f'{trajectory}.json'
    write_json(path,dict(schema=1,protocol=PROTOCOL,trajectory=trajectory,seed=seed,
                         requested_iterations=list(stages),records=records))
    return str(path.relative_to(folder))
