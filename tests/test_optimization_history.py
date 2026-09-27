import json
from types import SimpleNamespace
import numpy as np
import pytest
from optimization_history import save_history


def model():
    logs=[dict(loss=np.float32(i),plddt=.8,i_pae=.2,i_ptm=.6,cmap_loss_binder=.2,
               models=[i%2],recycles=0) for i in range(6)]
    return SimpleNamespace(_model_names=['model_1_ptm','model_2_ptm'],_tmp={'log':logs})


def test_all_stages_and_actual_models_are_preserved(tmp_path):
    af=model();save_history(tmp_path,'traj_1',af,[2,3,1],42)
    result=json.loads((tmp_path/'optimization/traj_1.json').read_text())
    assert result['seed']==42
    assert [r['stage'] for r in result['records']]==['logits']*2+['temperature']*3+['hard']
    assert [r['iteration'] for r in result['records']]==list(range(1,7))
    assert [r['model_names'][0] for r in result['records']]==af._model_names*3
    assert result['records'][-1]['metrics']['loss']==5.


@pytest.mark.parametrize('fault',['short','nonfinite','dispatch','missing'])
def test_bad_history_cannot_replace_previous_checkpoint(tmp_path,fault):
    af=model();save_history(tmp_path,'traj_1',af,[2,3,1],42)
    path=tmp_path/'optimization/traj_1.json';original=path.read_bytes()
    if fault=='short':af._tmp['log'].pop()
    elif fault=='nonfinite':af._tmp['log'][0]['loss']=np.nan
    elif fault=='dispatch':af._tmp['log'][0]['models']=[2]
    else:del af._tmp['log'][0]['i_pae']
    with pytest.raises(ValueError):save_history(tmp_path,'traj_1',af,[2,3,1],42)
    assert path.read_bytes()==original
