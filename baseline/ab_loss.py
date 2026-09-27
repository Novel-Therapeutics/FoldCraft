"""Exploratory paired loss ablation with shared stage seeds and randomized arm order.

Report confidence gates and combined fold/interface contact loss separately from
fold retention or binding accuracy. Analyze uncertainty over trajectories, not
sibling MPNN sequences; historical CSVs from unpaired runs remain historical.
"""
import math
import argparse
import os
import sys

import pandas as pd
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_get_best as A   # cmap_loss_binder, build_cond_cmap, mpnn_and_predict, passes, FOLDS, ...


def design_model(cond_cmap, cond_cmap_mask, binder_len, ipae_w):
    """FoldCraft's design model, but with a configurable i_pae weight (0 = the
    validated cmap-only baseline)."""
    m = mk_afdesign_model(data_dir=A._ROOT, protocol="binder", loss_callback=A.cmap_loss_binder,
                          use_templates=True)
    m.opt['cond_cmap'] = cond_cmap.copy()
    m.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
    m.prep_inputs(pdb_filename=A.TARGET, chain=A.CHAIN, binder_len=binder_len,
                  hotspot=A.TARGET_HOTSPOTS, rm_aa=A.RM_AA)
    m.opt["weights"].update({"cmap_loss_binder": 1.0, "rmsd": 0.0, "fape": 0.0,
                             "plddt": 0.0, "con": 0.0, "i_con": 0.0, "i_pae": ipae_w})
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="top7", choices=list(A.FOLDS))
    ap.add_argument("--n", type=int, default=15, help="trajectories per arm")
    ap.add_argument("--ipae", type=str, default="0.1",
                    help="comma-separated i_pae weights for the variant arms (sweep), "
                         "e.g. 0.05,0.1,0.2")
    ap.add_argument("--mpnn-samples", type=int, default=5)
    ap.add_argument("--mpnn-temp", type=float, default=0.1)
    ap.add_argument("--out", default="runs/ab_loss")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.seed < 2**32:
        ap.error("--seed must be a 32-bit nonnegative integer")
    if os.path.exists(args.out):
        ap.error('Output already exists; choose a fresh --out directory')
    if args.n < 1 or args.mpnn_samples < 1:
        ap.error('Trajectory and sample counts must be positive')

    weights = [float(w) for w in args.ipae.split(",")]
    if not all(math.isfinite(w) and w >= 0 for w in weights):
        ap.error("Loss weights must be finite and nonnegative")
    binder_template, binder_hotspots = A.FOLDS[args.fold]
    cond_cmap, cond_cmap_mask, binder_len = A.build_cond_cmap(binder_template, binder_hotspots, seed=args.seed)
    print(f"fold={args.fold} binder_len={binder_len} n={args.n} ipae_weights={weights} "
          f"mpnn_samples={args.mpnn_samples}", flush=True)

    mpnn = mk_mpnn_model(A.MODEL_NAME, backbone_noise=0.0, weights="soluble")
    # 'cmap' is the validated baseline; one variant arm per swept i_pae weight.
    arms = {"cmap": 0.0}
    for w in weights:
        arms[f"ipae{w}"] = w
    rows = {a: [] for a in arms}
    design_loss = {a: [] for a in arms}   # design-trajectory cmap_loss (fold fidelity)

    A.record_protocol(args.out, args, arms, binder_template, binder_hotspots, A.TARGET, A.TARGET_HOTSPOTS)
    for i in range(1, args.n + 1):
        for arm in A.arm_order(arms, args.seed, i):
            ipae_w = arms[arm]
            adir = os.path.join(args.out, arm)
            os.makedirs(os.path.join(adir, "traj"), exist_ok=True)
            clear_mem()
            name = f"traj_{i}"
            m = design_model(cond_cmap, cond_cmap_mask, binder_len, ipae_w)
            m.restart(seed=A.stage_seed(args.seed, "design", i), reset_opt=False)
            m.design_3stage(*A.DESIGN_STAGES)
            d_cmaploss = float(m.aux['log']['cmap_loss_binder'])
            design_loss[arm].append(d_cmaploss)
            tpdb = os.path.join(adir, "traj", f"{name}.pdb")
            m.save_pdb(tpdb, get_best=False)
            # prediction model: loss weights are irrelevant for predict (forward only)
            pred = design_model(cond_cmap, cond_cmap_mask, binder_len, 0.0)
            r = A.mpnn_and_predict(tpdb, name, mpnn, pred, cond_cmap, cond_cmap_mask,
                                   binder_len, args.mpnn_samples, args.mpnn_temp, adir, seed=args.seed)
            rows[arm].extend(r)
            npass = sum(A.passes(x) for x in r)
            A.atomic_write(os.path.join(adir, "results.csv"), lambda p: pd.DataFrame(rows[arm]).to_csv(p, index=False))
            print(f"[{arm}][{i}/{args.n}] {name}: design cmap_loss={d_cmaploss:.3f} "
                  f"passes={npass}/{args.mpnn_samples}", flush=True)

    print("\n=== SUMMARY (paired stage seeds, n={} trajectories/arm) ===".format(args.n))
    import numpy as np
    for arm in arms:
        df = pd.DataFrame(rows[arm])
        npass = int(df.apply(A.passes, axis=1).sum())
        dl = np.array(design_loss[arm])
        print(f"  {arm:10s}: {len(df)} designs, {npass} pass "
              f"({100*npass/len(df):.0f}%); design combined cmap_loss "
              f"median {np.median(dl):.3f}; design metrics iptm "
              f"{df['iptm'].median():.3f} ipae {df['ipae'].median():.3f}")
    print("  Exploratory confidence metrics only. Analyze paired trajectory-level outcomes "
          "and independent structural/experimental evidence before promoting an arm.")


if __name__ == "__main__":
    main()
