import numpy as np
import pytest
from design_objective import contact_objective


def test_legacy_matches_the_previous_arithmetic_exactly():
    rng=np.random.default_rng(8);p=rng.random((7,7));target=rng.random((7,7));mask=rng.integers(0,2,(7,7))
    assert contact_objective(p,target,mask,3)==np.sqrt(np.square(p*mask-target).sum(-1).mean())


def test_normalized_objective_ignores_unknown_target_padding():
    p=np.full((4,4),.2);target=np.ones((4,4));target[:2,:2]=0;mask=target.copy()
    base=contact_objective(p,target,mask,2,'normalized_pairs')
    padded=[np.pad(a,((3,0),(3,0))) for a in (p,target,mask)]
    assert contact_objective(*padded,2,'normalized_pairs')==base
    assert contact_objective(p,target,mask,2)!=contact_objective(*padded,2)


def test_normalized_objective_ignores_off_mask_predictions_and_stays_finite():
    target=np.zeros((4,4));mask=target.copy();mask[3,3]=1;target[3,3]=.5
    p=target.copy();p[0,0]=100
    assert np.isfinite(contact_objective(p,target,mask,2,'normalized_pairs'))
    with pytest.raises(ValueError):contact_objective(p,target,mask,2,'unknown')


def test_pilot_pairs_seeds_inputs_and_validation_and_changes_one_factor(tmp_path):
    from scripts.paired_pilot import plan,ARMS,CASES,SEEDS
    jobs=plan(tmp_path/'pilot',tmp_path/'weights')
    assert len(jobs)==12
    assert jobs==plan(tmp_path/'pilot',tmp_path/'weights')
    for case in CASES:
        for seed in SEEDS:
            selected=[j for j in jobs if j['case']==case and j['seed']==seed]
            assert {j['arm'] for j in selected}==set(ARMS)
            for j in selected:
                options=dict(zip(j['command'][2::2],j['command'][3::2]))
                assert options['--validation_models']=='model_1_ptm,model_2_ptm'
                assert options['--seed']==str(seed)
    assert not (tmp_path/'pilot').exists()
