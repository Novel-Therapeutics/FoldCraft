"""Exploratory paired loss ablation with shared stage seeds and randomized arm order.

Report confidence gates and combined fold/interface contact loss separately from
fold retention or binding accuracy. Analyze uncertainty over trajectories, not
sibling MPNN sequences; historical CSVs from unpaired runs remain historical.
"""
import math
import argparse
import os
import sys

import numpy as np
import pandas as pd
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.loss import get_contact_map
from colabdesign.mpnn import mk_mpnn_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_get_best as A


def cmap_loss_fp(inputs, outputs, opt):
    """FoldCraft's recall cmap loss PLUS a separate false-positive term.

    Returns two named losses; the caller weights them via opt['weights']
    (cmap_loss_binder = 1.0 always; fp_loss = the swept weight). Keeping them
    separate means aux['log']['cmap_loss_binder'] stays the pure recall metric
    (combined fold/interface loss) regardless of the fp weight.
    """
    conditioned_array = opt['cond_cmap']
    conditioned_mask = opt['cond_cmap_mask']
    binder_len = inputs['seq']['input'].shape[1]
    target_len = conditioned_array.shape[0] - binder_len

    i_cmap = get_contact_map(outputs, inputs["opt"]["i_con"]["cutoff"])
    cmap = get_contact_map(outputs, inputs["opt"]["con"]["cutoff"])
    i_cmap = i_cmap.at[-binder_len:, -binder_len:].set(cmap[-binder_len:, -binder_len:])

    # recall: verbatim FoldCraft loss
    out = i_cmap * conditioned_mask
    recall = jnp.sqrt(jnp.square(out - conditioned_array).sum(-1).mean())

    # false positives: predicted contacts OFF the conditioned support, in the
    # binder + cross blocks only (zero the target-target block top-left corner).
    fp = i_cmap * (1.0 - conditioned_mask)
    fp = fp.at[:target_len, :target_len].set(0.0)
    fp_loss = fp.sum(-1).mean()

    return {"cmap_loss_binder": recall, "fp_loss": fp_loss}


def design_model(cond_cmap, cond_cmap_mask, binder_len, fp_w):
    m = mk_afdesign_model(data_dir=A._ROOT, protocol="binder", loss_callback=cmap_loss_fp, use_templates=True)
    m.opt['cond_cmap'] = cond_cmap.copy()
    m.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
    m.prep_inputs(pdb_filename=A.TARGET, chain=A.CHAIN, binder_len=binder_len,
                  hotspot=A.TARGET_HOTSPOTS, rm_aa=A.RM_AA)
    m.opt["weights"].update({"cmap_loss_binder": 1.0, "fp_loss": fp_w, "rmsd": 0.0,
                             "fape": 0.0, "plddt": 0.0, "con": 0.0, "i_con": 0.0,
                             "i_pae": 0.0})
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="top7", choices=list(A.FOLDS))
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--fp", type=str, default="0.1,0.3",
                    help="comma-separated false-positive weights (sweep)")
    ap.add_argument("--mpnn-samples", type=int, default=5)
    ap.add_argument("--mpnn-temp", type=float, default=0.1)
    ap.add_argument("--out", default="runs/ab_loss_fp")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.seed < 2**32:
        ap.error("--seed must be a 32-bit nonnegative integer")
    if os.path.exists(args.out):
        ap.error('Output already exists; choose a fresh --out directory')
    if args.n < 1 or args.mpnn_samples < 1:
        ap.error('Trajectory and sample counts must be positive')

    weights = [float(w) for w in args.fp.split(",")]
    if not all(math.isfinite(w) and w >= 0 for w in weights):
        ap.error("Loss weights must be finite and nonnegative")
    binder_template, binder_hotspots = A.FOLDS[args.fold]
    cond_cmap, cond_cmap_mask, binder_len = A.build_cond_cmap(binder_template, binder_hotspots, seed=args.seed)
    print(f"fold={args.fold} binder_len={binder_len} n={args.n} fp_weights={weights}", flush=True)

    mpnn = mk_mpnn_model(A.MODEL_NAME, backbone_noise=0.0, weights="soluble")
    arms = {"fp0": 0.0}
    for w in weights:
        arms[f"fp{w}"] = w
    rows = {a: [] for a in arms}
    recall_loss = {a: [] for a in arms}   # combined fold/interface cmap loss

    A.record_protocol(args.out, args, arms, binder_template, binder_hotspots, A.TARGET, A.TARGET_HOTSPOTS)
    for i in range(1, args.n + 1):
        for arm in A.arm_order(arms, args.seed, i):
            fp_w = arms[arm]
            adir = os.path.join(args.out, arm)
            os.makedirs(os.path.join(adir, "traj"), exist_ok=True)
            clear_mem()
            name = f"traj_{i}"
            m = design_model(cond_cmap, cond_cmap_mask, binder_len, fp_w)
            m.restart(seed=A.stage_seed(args.seed, "design", i), reset_opt=False)
            m.design_3stage(*A.DESIGN_STAGES)
            recall_loss[arm].append(float(m.aux['log']['cmap_loss_binder']))
            tpdb = os.path.join(adir, "traj", f"{name}.pdb")
            m.save_pdb(tpdb, get_best=False)
            pred = design_model(cond_cmap, cond_cmap_mask, binder_len, 0.0)
            r = A.mpnn_and_predict(tpdb, name, mpnn, pred, cond_cmap, cond_cmap_mask,
                                   binder_len, args.mpnn_samples, args.mpnn_temp, adir, seed=args.seed)
            rows[arm].extend(r)
            npass = sum(A.passes(x) for x in r)
            A.atomic_write(os.path.join(adir, "results.csv"), lambda p: pd.DataFrame(rows[arm]).to_csv(p, index=False))
            print(f"[{arm}][{i}/{args.n}] {name}: recall cmap_loss="
                  f"{recall_loss[arm][-1]:.3f} passes={npass}/{args.mpnn_samples}", flush=True)

    print("\n=== SUMMARY (paired stage seeds, n={} trajectories/arm) ===".format(args.n))
    for arm in arms:
        df = pd.DataFrame(rows[arm])
        npass = int(df.apply(A.passes, axis=1).sum())
        rl = np.array(recall_loss[arm])
        print(f"  {arm:8s}: {len(df)} designs, {npass} pass "
              f"({100*npass/len(df):.0f}%); recall cmap_loss median {np.median(rl):.3f}; "
              f"design iptm {df['iptm'].median():.3f} ipae {df['ipae'].median():.3f}")
    print("  Exploratory confidence metrics only. Analyze paired trajectory-level outcomes "
          "and independent structural/experimental evidence before promoting an arm.")


if __name__ == "__main__":
    main()
