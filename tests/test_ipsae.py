"""Tests for the ipSAE core (baseline/ipsae.py).

The key test is differential: ``_ref_asym`` is a literal loop translation of the
reference algorithm (DunbrackLab/IPSAE), and we assert the vectorized
``ipsae_d0res_asym`` matches it across random PAE matrices. Plus sanity bounds
and the symmetry property.
"""
import importlib.util
import os

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("ipsae_mod", os.path.join(_REPO, "baseline", "ipsae.py"))
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)


def _ref_asym(pae, mask_align, mask_scored, cutoff):
    """Literal loop translation of ipsae_d0res (reference IPSAE)."""
    valid = np.outer(mask_align, mask_scored) & (pae < cutoff)
    d0 = ip.calc_d0_array(valid.sum(axis=1))
    best = 0.0
    for i in range(pae.shape[0]):
        if not mask_align[i]:
            continue
        vrow = valid[i]
        val = (1.0 / (1.0 + (pae[i][vrow] / d0[i]) ** 2)).mean() if vrow.any() else 0.0
        best = max(best, float(val))
    return best


def _random_pae(n, seed):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0, 30, (n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m


@pytest.mark.parametrize("n,la,cut,seed", [
    (8, 4, 10.0, 0), (12, 5, 10.0, 1), (20, 13, 15.0, 2), (6, 3, 5.0, 3), (30, 7, 10.0, 4),
])
def test_vectorized_matches_reference_loop(n, la, cut, seed):
    pae = _random_pae(n, seed)
    a = np.zeros(n, bool); a[:la] = True; b = ~a
    assert ip.ipsae_d0res_asym(pae, a, b, cut) == pytest.approx(_ref_asym(pae, a, b, cut))
    assert ip.ipsae_d0res_asym(pae, b, a, cut) == pytest.approx(_ref_asym(pae, b, a, cut))


def test_perfect_pae_scores_high():
    # all interchain PAE = 0 -> ptm ~ 1 -> ipSAE near 1.
    pae = np.zeros((10, 10))
    assert ip.ipsae(pae, 5, 5, pae_cutoff=10.0) > 0.99


def test_hopeless_pae_scores_zero():
    # all PAE above the cutoff -> no valid interface pairs -> ipSAE = 0.
    pae = np.full((10, 10), 25.0)
    assert ip.ipsae(pae, 5, 5, pae_cutoff=10.0) == 0.0


def test_symmetric_in_chain_labels():
    pae = _random_pae(14, 7)
    # splitting [A|B] vs scoring direction: ipsae() takes the max of both, so it
    # equals the max of the two asyms regardless of which block is "first".
    a = np.zeros(14, bool); a[:6] = True; b = ~a
    expected = max(ip.ipsae_d0res_asym(pae, a, b, 10.0), ip.ipsae_d0res_asym(pae, b, a, 10.0))
    assert ip.ipsae(pae, 6, 8, 10.0) == pytest.approx(expected)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ip.ipsae(np.zeros((10, 10)), 4, 5)  # 4+5 != 10
