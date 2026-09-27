"""Resolve the actual files handed to each inference backend before cache reuse."""
from pathlib import Path
import json


def af2_checkpoint(root, model='model_1_ptm'):
    root = Path(root).expanduser().resolve()
    for path in (root/'params'/f'params_{model}.npz', root/f'params_{model}.npz',
                 root/'params'/f'{model}.npz', root/f'{model}.npz'):
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing {model} checkpoint under {root}')


def esm_snapshot(model_dir=None, revision='main'):
    if model_dir:
        path = Path(model_dir).expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download('facebook/esmfold_v1', revision=revision,
                    allow_patterns=['*.json','*.txt','*.safetensors','*.bin'])).resolve()
    if not (path/'config.json').is_file() or not any((path/name).is_file() for name in ('model.safetensors','pytorch_model.bin','model.safetensors.index.json','pytorch_model.bin.index.json')):
        raise FileNotFoundError(f'Incomplete ESMFold snapshot: {path}')
    for index in path.glob('*.index.json'):
        for shard in set(json.loads(index.read_text()).get('weight_map', {}).values()):
            relative = Path(shard)
            if relative.is_absolute() or '..' in relative.parts or not (path/relative).is_file():
                raise FileNotFoundError(f'Missing/invalid checkpoint shard: {shard}')
    return path
