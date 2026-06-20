"""A/B for Accuracy #3: does adding a small i_pae term to FoldCraft's cmap-only
design loss raise the design success rate -- WITHOUT sacrificing fold fidelity?

The validated loss is cmap_loss_binder=1.0 only (all other weights 0), so the
hallucination gets no gradient toward the quantities the success gate actually
checks (plddt / i_pae / i_ptm). This tests adding an interface-PAE term.

UNPAIRED (unlike get_best): the loss change alters the design trajectory from the
first step, so the two arms are independent runs.
  arm 'cmap'      : cmap_loss_binder = 1.0                  (current baseline)
  arm 'cmap_ipae' : cmap_loss_binder = 1.0 + i_pae = <w>    (default w = 0.1)

Per trajectory: design_3stage(arm weights) -> save (get_best=False) -> ProteinMPNN
non-interface redesign -> AF2-ptm predict mpnn_samples. Writes results.csv +
designs/ per arm (scoreable by score_openmm.py).

THE COMPARISON HAS TWO HALVES, both reported:
  1. success rate (plddt>0.8, iptm>0.5, ipae<0.35)  -- does i_pae help?
  2. fold fidelity = the design-trajectory cmap_loss  -- if i_pae pulls designs
     OFF the conditioned fold, this rises and any success gain is illusory (the
     whole point of FoldCraft is fold-conditioning). A win requires (1) up with
     (2) NOT regressing.

Reuses the proven helpers from ab_get_best.py (cmap_loss_binder, build_cond_cmap,
mpnn_and_predict -- incl. the clear_mem + set_seed MPNN handling).

    python baseline/ab_loss.py --fold top7 --n 15 --ipae 0.1
"""
import argparse
import os
import sys

import pandas as pd
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model

sys.path.insert(0, ".")
sys.path.insert(0, "baseline")
import ab_get_best as A   # cmap_loss_binder, build_cond_cmap, mpnn_and_predict, passes, FOLDS, ...


def design_model(cond_cmap, cond_cmap_mask, binder_len, ipae_w):
    """FoldCraft's design model, but with a configurable i_pae weight (0 = the
    validated cmap-only baseline)."""
    m = mk_afdesign_model(protocol="binder", loss_callback=A.cmap_loss_binder,
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
    ap.add_argument("--ipae", type=float, default=0.1, help="i_pae weight for the variant arm")
    ap.add_argument("--mpnn-samples", type=int, default=5)
    ap.add_argument("--mpnn-temp", type=float, default=0.1)
    ap.add_argument("--out", default="baseline/ab_loss")
    args = ap.parse_args()

    binder_template, binder_hotspots = A.FOLDS[args.fold]
    cond_cmap, cond_cmap_mask, binder_len = A.build_cond_cmap(binder_template, binder_hotspots)
    print(f"fold={args.fold} binder_len={binder_len} n={args.n} ipae_w={args.ipae} "
          f"mpnn_samples={args.mpnn_samples}", flush=True)

    mpnn = mk_mpnn_model(A.MODEL_NAME, backbone_noise=0.0, weights="soluble")
    arms = {"cmap": 0.0, "cmap_ipae": args.ipae}
    rows = {a: [] for a in arms}
    design_loss = {a: [] for a in arms}   # design-trajectory cmap_loss (fold fidelity)

    for arm, ipae_w in arms.items():
        adir = os.path.join(args.out, arm)
        os.makedirs(os.path.join(adir, "traj"), exist_ok=True)
        for i in range(1, args.n + 1):
            clear_mem()
            mpnn.set_seed(None)             # clear_mem() deletes the hoisted mpnn's key
            name = f"traj_{i}"
            m = design_model(cond_cmap, cond_cmap_mask, binder_len, ipae_w)
            m.design_3stage(*A.DESIGN_STAGES)
            d_cmaploss = float(m.aux['log']['cmap_loss_binder'])
            design_loss[arm].append(d_cmaploss)
            tpdb = os.path.join(adir, "traj", f"{name}.pdb")
            m.save_pdb(tpdb, get_best=False)
            # prediction model: loss weights are irrelevant for predict (forward only)
            pred = design_model(cond_cmap, cond_cmap_mask, binder_len, 0.0)
            r = A.mpnn_and_predict(tpdb, name, mpnn, pred, cond_cmap, cond_cmap_mask,
                                   binder_len, args.mpnn_samples, args.mpnn_temp, adir)
            rows[arm].extend(r)
            npass = sum(A.passes(x) for x in r)
            pd.DataFrame(rows[arm]).to_csv(os.path.join(adir, "results.csv"), index=False)
            print(f"[{arm}][{i}/{args.n}] {name}: design cmap_loss={d_cmaploss:.3f} "
                  f"passes={npass}/{args.mpnn_samples}", flush=True)

    print("\n=== SUMMARY (unpaired, n={} trajectories/arm) ===".format(args.n))
    import numpy as np
    for arm in arms:
        df = pd.DataFrame(rows[arm])
        npass = int(df.apply(A.passes, axis=1).sum())
        dl = np.array(design_loss[arm])
        print(f"  {arm:10s}: {len(df)} designs, {npass} pass "
              f"({100*npass/len(df):.0f}%); design fold-fidelity cmap_loss "
              f"median {np.median(dl):.3f}; design metrics iptm "
              f"{df['iptm'].median():.3f} ipae {df['ipae'].median():.3f}")
    print("  WIN iff cmap_ipae pass-rate UP and its design cmap_loss NOT worse "
          "(else i_pae traded fold-conditioning for confidence). Then score_openmm both arms.")


if __name__ == "__main__":
    main()
