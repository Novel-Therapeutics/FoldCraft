import numpy as np
import pandas as pd
import pytest
from scripts.expanded_benchmark import protocol
from scripts.analyze_expanded import analyze,ranking_choices,bootstrap_mean
from scripts.score_boltz_paired import paired_seed


def test_expanded_protocol_has_balanced_pairs_and_separate_holdout(tmp_path):
    plan=protocol(tmp_path/'out',tmp_path/'weights')
    assert len(plan['jobs'])==48
    assert {j['family'] for j in plan['jobs'] if j['split']=='holdout'}=={'IFNAR'}
    assert not (tmp_path/'out').exists()
    assert all(j['command'][j['command'].index('--artifact_mode')+1]=='compact' for j in plan['jobs'])
    assert plan['promotion']['min_unseen_target_families']==2
    for j in plan['jobs']:
        other=dict(case=j['case'],trajectory='traj_1',seed=j['seed'],draw=0,arm='other')
        row=dict(other,arm=j['arm'])
        assert paired_seed(row,0)==paired_seed(other,0)
        assert paired_seed(row,0)!=paired_seed(row,1)


def fixture():
    rows=[];jobs=[]
    for split in ('development','holdout'):
        for case in (split+'1',split+'2'):
            for seed in (1,2):
                for arm in ('baseline','mpnn_temp_02'):
                    jobs.append(dict(case=case,seed=seed,arm=arm))
                    for draw in (0,1):
                        rows.append(dict(name=f'{case}_{seed}_{arm}_{draw}',split=split,case=case,seed=seed,arm=arm,
                            target=case,family=split,scaffold='fold',binder_sequence='AAA',rmsd=2.,esmfold_rmsd=2.,
                            esmfold_plddt=80.,boltz2_iptm=.8,epitope_coverage=.2,interchain_clashes=0,iptm=.7,
                            single_model_pass=True,two_model_pass=True))
    return pd.DataFrame(rows),dict(jobs=jobs,promotion=dict(min_unseen_target_families=2,primary_gain_pp=10,
            max_template_rmsd_regression_A=.5,max_clash_regression=1))


def test_siblings_do_not_inflate_replication_or_allow_default_promotion():
    frame,plan=fixture();report,blocks,picks=analyze(frame,plan)
    assert report['paired_blocks']==8 and len(frame)==32
    assert report['temperature_deltas']['holdout']['proxy_pass']['n_paired_trajectories']==4
    assert report['temperature_deltas']['holdout']['proxy_pass']['n_case_clusters']==2
    assert not report['temperature_promotion']['promote']
    assert not report['ranking_promotion']['promote']
    assert len(picks)==16
    bad=frame.copy();bad.loc[0,'boltz2_iptm']=np.nan
    with pytest.raises(ValueError):analyze(bad,plan)
    with pytest.raises(ValueError):analyze(frame.iloc[:-1],plan)


def test_ranking_never_uses_independent_oracle_outcomes():
    frame,_=fixture();frame.loc[0,'rmsd']=99;frame.loc[0,'iptm']=.99
    a=ranking_choices(frame)
    frame['boltz2_iptm']=1-frame.boltz2_iptm;frame['esmfold_rmsd']=100
    b=ranking_choices(frame)
    assert a.name.tolist()==b.name.tolist()
    assert not a[a.policy=='fold_contact_rank'].name.eq(frame.iloc[0]['name']).any()
