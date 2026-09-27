"""Pure (GPU-free) construction of the fold-conditioned contact map.

The fold-conditioned cmap is the artifact the design loss conditions on, so it
is the single most accuracy-relevant *deterministic* step in the pipeline. This
module extracts its assembly out of ``FoldCraft.py``'s ``main()`` so it can be
unit-tested and shared by both the standard and VHH paths (which previously
duplicated the same logic inline).

Index conventions: hotspot/mask range strings are parsed by ``set_range`` into
1-based residue numbers (inclusive of both endpoints), which map to 0-based
array positions as ``residue R -> index R-1``. An empty binder selection includes
all binder positions. The CLI maps original PDB IDs to these positions before
calling this module; direct callers must already supply model positions.

These semantics decide which residues the design loss conditions on, so changes
here are accuracy-relevant and should be measured against the design baseline.
See tests/test_cmap_utils.py for the differential cross-check against an
independent re-implementation of the assembly.
"""
import numpy as np

from biopython_utils import set_range
from input_validation import positions, validate_cmap


def apply_binder_mask(binder_cmap, binder_mask):
    """Zero out masked binder rows/columns in the binder's intra-chain cmap.

    ``binder_mask`` is a residue-range string (e.g. ``"14-30"``); ``set_range``
    yields 1-based residue numbers, which map to 0-based positions as
    ``residue R -> index R-1`` -- the same convention the hotspots use in
    ``assemble_fold_conditioned_cmap``. A residue outside ``1..binder_len``
    raises ``ValueError`` (rather than silently wrapping a negative index or
    raising a bare IndexError). Operates on a copy; the input is not mutated.
    """
    binder_cmap = np.array(validate_cmap(binder_cmap), copy=True)
    if binder_mask != '':
        n = binder_cmap.shape[0]
        for r in set_range(binder_mask):
            if not 1 <= r <= n:
                raise ValueError(
                    f"binder_mask residue {r} is out of range for a "
                    f"{n}-residue binder (expected 1..{n})."
                )
            binder_cmap[r - 1, :] = 0.
            binder_cmap[:, r - 1] = 0.
    return binder_cmap


def assemble_fold_conditioned_cmap(binder_cmap, target_len, binder_len,
                                   target_hotspots, binder_hotspots,
                                   binder_mask=''):
    """Build the ``(target_len + binder_len)`` square fold-conditioned cmap.

    Parameters
    ----------
    binder_cmap : array (binder_len x binder_len)
        The binder's intra-chain contact map. In the standard path this comes
        from an AlphaFold2 prediction of the binder monomer; in the VHH path it
        is loaded from ``framework/vhh.npy``.
    target_len, binder_len : int
        Chain lengths. The binder block is placed in the bottom-right corner.
    target_hotspots, binder_hotspots : str
        Residue-range strings (e.g. ``"36-41,84-88"``). If ``binder_hotspots``
        is the empty string, all binder residues are used (the original
        default).
    binder_mask : str, optional
        Residue ranges in the binder to zero out before assembly. The VHH path
        does not use a binder mask; pass ``''`` to match it.

    Returns
    -------
    np.ndarray
        The fold-conditioned cmap (float; target<->binder hotspot contacts
        marked ``1.0``).
    """
    if not isinstance(target_len, (int, np.integer)) or target_len < 1 or not isinstance(binder_len, (int, np.integer)) or binder_len < 1:
        raise ValueError('Target and binder lengths must be positive integers')
    validate_cmap(binder_cmap, size=binder_len)
    binder_cmap = apply_binder_mask(binder_cmap, binder_mask)
    targets = np.asarray(positions(target_hotspots, target_len), dtype=int) - 1
    binders = positions(binder_hotspots, binder_len, allow_empty=True) or list(range(1, binder_len + 1))
    binders = np.asarray(binders, dtype=int) - 1 + target_len
    fc_cmap = np.zeros((target_len + binder_len, target_len + binder_len))
    fc_cmap[-binder_len:, -binder_len:] = binder_cmap
    fc_cmap[np.ix_(binders, targets)] = 1.0
    fc_cmap[np.ix_(targets, binders)] = 1.0

    return fc_cmap


def binarize_cmap(fc_cmap):
    """Return the binary contact mask (any positive entry -> 1).

    This is the ``cond_cmap_mask`` the design loop loads alongside the cmap.
    Returns a copy; does not mutate the input.
    """
    mask = np.array(fc_cmap, copy=True)
    mask[mask > 0] = 1
    return mask
