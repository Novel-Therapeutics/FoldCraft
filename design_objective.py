"""Opt-in, testable design objectives. Legacy remains the production default."""
import numpy as np


def contact_objective(prediction, target, mask, binder_len, mode='legacy', xp=np):
    error = prediction * mask - target
    if mode == 'legacy':
        return xp.sqrt(xp.square(error).sum(-1).mean())
    if mode != 'normalized_pairs':
        raise ValueError(f'Unknown contact objective: {mode}')
    # Fold and interface each contribute one RMSE, normalized by observed pairs.
    # Unknown/off-mask entries are not introduced as negative restraints.
    def rmse(values, observed):
        return xp.sqrt(xp.square(values).sum() / xp.maximum(observed.sum(), 1.) + 1e-8)
    fold = rmse(error[-binder_len:, -binder_len:], mask[-binder_len:, -binder_len:])
    cross_sq = xp.square(error[:-binder_len, -binder_len:]).sum() + xp.square(error[-binder_len:, :-binder_len]).sum()
    cross_n = mask[:-binder_len, -binder_len:].sum() + mask[-binder_len:, :-binder_len].sum()
    interface = xp.sqrt(cross_sq / xp.maximum(cross_n, 1.) + 1e-8)
    return fold + interface
