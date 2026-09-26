"""ipSAE: a PAE-derived interface-confidence score that fixes ipTM's size bias.

Pure-numpy core, faithful to the reference implementation:
  R. Dunbrack, "Res ipSAE loquunt: what's wrong with AlphaFold's ipTM score and
  how to fix it", bioRxiv 2025.02.10.637595; https://github.com/DunbrackLab/IPSAE
  (MIT license).

We compute the canonical **ipSAE = ipsae_d0res**: for each residue i in the
aligned chain, average ``ptm(PAE[i,j], d0_i)`` over scored-chain residues j with
``PAE[i,j] < pae_cutoff``, where ``d0_i`` comes from the number of such valid j;
the asymmetric value is ``max_i`` of that, and ipSAE is the max of the two chain
directions. It needs only the PAE matrix (in Angstroms) and the chain split.
"""
import numpy as np


def calc_d0_array(L):
    """d0 from a residue count L (Yang & Skolnick), floored at 1.0; L floored at 26."""
    L = np.maximum(26.0, np.asarray(L, dtype=float))
    return np.maximum(1.0, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def ptm_func(x, d0):
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def ipsae_d0res_asym(pae, mask_align, mask_scored, pae_cutoff):
    """Asymmetric ipSAE (d0res) for aligned-chain -> scored-chain."""
    valid = mask_align[:, None] & mask_scored[None, :] & (pae < pae_cutoff)
    n0_byres = valid.sum(axis=1)                      # valid j per aligned residue i
    d0_byres = calc_d0_array(n0_byres)
    ptm = ptm_func(pae, d0_byres[:, None])            # row i normalized by its own d0
    cnt = valid.sum(axis=1)
    byres = np.where(cnt > 0, (ptm * valid).sum(axis=1) / np.maximum(cnt, 1), 0.0)
    byres = np.where(mask_align, byres, 0.0)          # only aligned-chain residues score
    return float(byres.max()) if mask_align.any() else 0.0


def ipsae(pae, len_a, len_b, pae_cutoff=10.0):
    """Canonical ipSAE for a two-chain complex whose PAE is ordered [chainA, chainB].

    Symmetric in the chain labels (max of both directions), so only the split
    point ``len_a`` matters, not which chain is target vs binder.
    """
    pae = np.asarray(pae, dtype=float)
    n = pae.shape[0]
    if n != len_a + len_b:
        raise ValueError(f"PAE is {n}x{n} but len_a+len_b={len_a}+{len_b}={len_a + len_b}")
    a = np.zeros(n, dtype=bool)
    a[:len_a] = True
    b = ~a
    return max(
        ipsae_d0res_asym(pae, a, b, pae_cutoff),
        ipsae_d0res_asym(pae, b, a, pae_cutoff),
    )
