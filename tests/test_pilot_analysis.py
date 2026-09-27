import json
import pandas as pd
import pytest
from scripts.analyze_pilot import summarize
from run_state import sha256


def test_pilot_aggregates_trajectories_not_unequal_sibling_counts(tmp_path):
    pool=tmp_path/'pool';(pool/'designs').mkdir(parents=True)
    rows=[];sources={}
    for name,seed,score in [('a',1,0.),('b',1,0.),('c',2,1.)]:
        pdb=pool/'designs'/f'{name}.pdb';pdb.write_text(name)
        rows.append(dict(name=name,sequence='AAA/GGG',binder_sequence='GGG',case='case',arm='baseline',seed=seed,
                         single_model_pass=bool(score),two_model_pass=False,rmsd=score,ipsae=score,
                         epitope_coverage=score,interchain_clashes=0))
        sources[name]=dict(sequence='AAA/GGG',pdb_sha256=sha256(pdb))
    (pool/'pool_manifest.json').write_text(json.dumps(dict(sources=sources)))
    pd.DataFrame(rows).to_csv(pool/'results.csv',index=False)
    summarize(tmp_path)
    report=json.loads((tmp_path/'analysis.json').read_text())
    entry=report['summaries'][0]
    assert entry['metrics']['single_model_pass']['mean']==.5
    assert entry['metrics']['boltz2_iptm']==dict(mean=None,measured=0,missing=3)
    (pool/'designs/a.pdb').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):summarize(tmp_path)
