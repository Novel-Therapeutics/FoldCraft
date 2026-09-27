import pickle
import numpy as np
import pytest
from prediction_artifacts import prediction_artifact
from model_validation import validate_candidate
from test_model_validation import Predictor


def test_compaction_is_lossless_and_drops_unused_tensors():
    original=dict(pae=np.arange(36,dtype=np.float32).reshape(1,6,6),
                  atom_positions=np.arange(666,dtype=np.float32).reshape(1,6,37,3),
                  plDDT='not part of contract',prev=np.zeros((256,256,128)))
    compact=prediction_artifact(original,'compact')
    for key in ('pae','atom_positions'):
        assert compact[key].dtype == original[key].dtype
        np.testing.assert_array_equal(compact[key],original[key])
    assert 'prev' not in compact
    assert len(pickle.dumps(compact))<len(pickle.dumps(original))/100
    assert prediction_artifact(original,'full') is original


def test_candidate_compact_and_full_predictions_match(tmp_path):
    reports=[]
    for mode in ('full','compact'):
        reports.append(validate_candidate(Predictor(),'AAA/GGG',3,tmp_path/mode,'candidate',
                    'model_1_ptm,model_2_ptm',3,42,artifact_mode=mode))
    for name in ('candidate','candidate.model_2_ptm'):
        with (tmp_path/'full/designs'/f'{name}.pickle').open('rb') as f:a=pickle.load(f)
        with (tmp_path/'compact/designs'/f'{name}.pickle').open('rb') as f:b=pickle.load(f)
        for key in ('pae','atom_positions'):np.testing.assert_array_equal(a[key],b[key])
    assert reports[0]['all_models_pass']==reports[1]['all_models_pass']


def test_missing_compact_arrays_fail():
    with pytest.raises(ValueError):prediction_artifact({'pae':np.ones((1,2,2))},'compact')
