"""Golden scientific inputs: historical bytes are never rewritten by migration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from FoldCraft import parse_args
from input_validation import preflight, VHH_CONVENTION, VHH_CONVENTIONS
from cmap_utils import assemble_fold_conditioned_cmap, binarize_cmap
from run_state import sha256, start_run

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT/'tests/fixtures/vhh_conventions.json').read_text())


def array_sha(value):
    return hashlib.sha256(np.asarray(value,dtype='<f8').tobytes(order='C')).hexdigest()


@pytest.mark.parametrize('example', GOLDEN['examples'], ids=lambda e:e['name'])
def test_archive_exact_reproduction_and_current_golden(example,tmp_path):
    assert VHH_CONVENTIONS == GOLDEN['conventions']
    assert sha256(ROOT/'framework/vhh.npy') == GOLDEN['framework_sha256']
    for field in ('target','archive','archive_mask'):
        assert sha256(ROOT/example[field]) == example[field+'_sha256']
    framework = np.load(ROOT/'framework/vhh.npy')
    original = np.load(ROOT/example['archive'])
    maps={}
    for convention in VHH_CONVENTIONS:
        out=tmp_path/convention
        args=parse_args(['--output_folder',str(out),'--vhh','--vhh_convention',convention,
            '--target_template',str(ROOT/example['target']),'--target_hotspots',example['target_hotspots']])
        prepared=preflight(args)
        target,_,thot,bhot,mask=prepared
        assert target.length == example['target_length']
        maps[convention]=assemble_fold_conditioned_cmap(framework,target.length,127,thot,bhot,mask)
        state=start_run(out,args,prepared)
        assert state['vhh_convention']==convention
        assert state['mapped_hotspots']['binder']==VHH_CONVENTIONS[convention]
    archived=maps[example['archive_convention']]
    np.testing.assert_array_equal(archived,original)
    np.testing.assert_array_equal(binarize_cmap(archived),np.load(ROOT/example['archive_mask']))
    current=maps[VHH_CONVENTION]
    assert array_sha(current)==example['current_array_sha256']
    assert array_sha(binarize_cmap(current))==example['current_mask_array_sha256']
    assert np.count_nonzero(current!=original)==example['changed_cells']


def test_historical_convention_requires_vhh():
    with pytest.raises(SystemExit):
        parse_args(['--output_folder','unused','--target_template','unused','--target_hotspots','1',
                    '--binder_template','unused','--vhh_convention','foldcraft-127-historical-v0'])
