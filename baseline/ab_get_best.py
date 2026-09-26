"""Paired A/B for Accuracy #2: does saving the BEST stage-3 trajectory iterate
(get_best=True) instead of the LAST one (get_best=False, the current default)
yield better final designs?

Paired per trajectory: design_3stage is run ONCE, then saved BOTH ways and each
backbone is sent through the same MPNN-redesign + AF2-ptm evaluation. Because
both arms share the trajectory, the get_best effect is isolated from the design's
seed noise (much stronger than two independent --num_designs runs).

For each of N trajectories on a fold:
  * design_3stage (cmap-only loss, exactly as FoldCraft.py)
  * save get_best=False (last) and get_best=True (best); record both traj cmap_loss
  * for EACH arm: ProteinMPNN non-interface redesign -> mpnn_samples sequences ->
    AF2-ptm predict (model_1/2_ptm, 3 recycles), keep every design + its metrics
Outputs baseline/ab_get_best/<arm>/{results.csv,designs/<name>.pdb} in the same
layout the oracle scorers expect, so `python baseline/score_openmm.py
baseline/ab_get_best/<arm>` adds the family-neutral interface energy afterward.

Pass criterion (per design) is FoldCraft's gate: plddt>0.8, i_ptm>0.5, i_pae<0.35.
The headline comparison is the per-trajectory paired delta (best arm minus last
arm) in pass count and in best-design metrics.

Run in the FoldCraft env on a GPU box, from the repo root:
    python baseline/ab_get_best.py --fold top7 --n 15
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.loss import get_contact_map
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.mpnn import mk_mpnn_model

sys.path.insert(0, ".")
from cmap_utils import assemble_fold_conditioned_cmap, binarize_cmap
from biopython_utils import hotspot_residues

# Fold configs from examples/scripts/design_*_pd_l1.sh (target + hotspots fixed
# across folds; binder template + binder hotspots vary).
FOLDS = {
    "top7":     ("examples/templates/1qys1.pdb", "26-40,58-71"),
    "barrel":   ("examples/templates/6d0t1.pdb", "30-44,90-104"),
    "iglike":   ("examples/templates/3sd21.pdb", "1-9,40-54"),
    "solenoid": ("examples/templates/3jx81.pdb", "28-42,70-84"),
}
TARGET = "examples/targets/pd-l1-1.pdb"
TARGET_HOTSPOTS = "30-34,50-54,69-76"
CHAIN = "A"
RM_AA = "C"
MODEL_NAME = "v_48_010"
DESIGN_STAGES = (100, 100, 20)
GATE = dict(plddt=0.8, i_ptm=0.5, i_pae=0.35)


def cmap_loss_binder(inputs, outputs, opt):
    """Verbatim from FoldCraft.py."""
    conditioned_array = opt['cond_cmap']
    conditioned_mask = opt['cond_cmap_mask']
    binder_len = inputs['seq']['input'].shape[1]
    i_cmap = get_contact_map(outputs, inputs["opt"]["i_con"]["cutoff"])
    cmap = get_contact_map(outputs, inputs["opt"]["con"]["cutoff"])
    i_cmap = i_cmap.at[-binder_len:, -binder_len:].set(cmap[-binder_len:, -binder_len:])
    out = i_cmap * conditioned_mask
    return {"cmap_loss_binder": jnp.sqrt(jnp.square(out - conditioned_array).sum(-1).mean())}


def _native_seq(m):
    return "".join(residue_constants.restypes[a] if a < residue_constants.restype_num
                   else "X" for a in m._wt_aatype)


def build_cond_cmap(binder_template, binder_hotspots):
    clear_mem()
    afb = mk_afdesign_model(protocol="fixbb", use_templates=True)
    afb.prep_inputs(pdb_filename=binder_template, ignore_missing=False, chain=CHAIN,
                    rm_template_seq=False, rm_template_sc=False)
    binder_len = afb._len
    afb.set_seq(_native_seq(afb))
    afb.predict(num_recycles=3, verbose=False)
    binder_cmap = afb.aux['cmap']
    aft = mk_afdesign_model(protocol="fixbb", use_templates=True)
    aft.prep_inputs(pdb_filename=TARGET, ignore_missing=False, chain=CHAIN)
    target_len = aft._len
    fc = assemble_fold_conditioned_cmap(binder_cmap, target_len, binder_len,
                                        TARGET_HOTSPOTS, binder_hotspots)
    return fc, binarize_cmap(fc), binder_len


def design_model(cond_cmap, cond_cmap_mask, binder_len):
    m = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder, use_templates=True)
    m.opt['cond_cmap'] = cond_cmap.copy()
    m.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
    m.prep_inputs(pdb_filename=TARGET, chain=CHAIN, binder_len=binder_len,
                  hotspot=TARGET_HOTSPOTS, rm_aa=RM_AA)
    m.opt["weights"].update({"cmap_loss_binder": 1.0, "rmsd": 0.0, "fape": 0.0,
                             "plddt": 0.0, "con": 0.0, "i_con": 0.0, "i_pae": 0.0})
    return m


def mpnn_and_predict(traj_pdb, name, mpnn, pred, cond_cmap, cond_cmap_mask,
                     binder_len, mpnn_samples, mpnn_temp, out_dir):
    """Replicate FoldCraft.py: non-interface MPNN redesign of traj_pdb, then
    AF2-ptm predict each sample. Returns list of per-design metric dicts and
    writes accepted+rejected designs to out_dir/designs (so all are scoreable)."""
    design_pos = range(1, binder_len + 1)   # incl. the fix: cover the C-terminal residue
    interface = list(hotspot_residues(traj_pdb, 'B').keys())
    fix_pos = ','.join(f'B{i}' for i in design_pos if i not in interface)
    mpnn.prep_inputs(pdb_filename=traj_pdb, chain='A,B', fix_pos=fix_pos,
                     rm_aa=RM_AA, inverse=True)
    samples = mpnn.sample_parallel(temperature=mpnn_temp, batch=mpnn_samples)
    os.makedirs(os.path.join(out_dir, "designs"), exist_ok=True)
    rows = []
    for num, seq in enumerate(samples['seq']):
        pred.set_seq(seq[-binder_len:])
        pred.predict(num_recycles=3, verbose=False, models=["model_1_ptm", "model_2_ptm"])
        log = pred.aux['log']
        dname = f"{name}_{num}"
        pred.save_pdb(os.path.join(out_dir, "designs", f"{dname}.pdb"), get_best=False)
        rows.append(dict(name=dname, sequence=seq,
                         plddt=round(float(log['plddt']), 4),
                         iptm=round(float(log['i_ptm']), 4),
                         ipae=round(float(log['i_pae']), 4),
                         cmap_loss=round(float(log['cmap_loss_binder']), 4)))
    return rows


def passes(r):
    return r['plddt'] > GATE['plddt'] and r['iptm'] > GATE['i_ptm'] and r['ipae'] < GATE['i_pae']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default="top7", choices=list(FOLDS))
    ap.add_argument("--n", type=int, default=15, help="trajectories")
    ap.add_argument("--mpnn-samples", type=int, default=5)
    ap.add_argument("--mpnn-temp", type=float, default=0.1)
    ap.add_argument("--out", default="baseline/ab_get_best")
    args = ap.parse_args()

    binder_template, binder_hotspots = FOLDS[args.fold]
    cond_cmap, cond_cmap_mask, binder_len = build_cond_cmap(binder_template, binder_hotspots)
    print(f"fold={args.fold} binder_len={binder_len} n={args.n} "
          f"mpnn_samples={args.mpnn_samples}")

    mpnn = mk_mpnn_model(MODEL_NAME, backbone_noise=0.0, weights="soluble")
    arms = {"last": dict(get_best=False), "best": dict(get_best=True)}
    for arm in arms:
        os.makedirs(os.path.join(args.out, arm), exist_ok=True)
    rows = {a: [] for a in arms}
    traj_loss = []   # (cmap_loss_last, cmap_loss_best) per trajectory

    for i in range(1, args.n + 1):
        clear_mem()
        mpnn.set_seed(None)   # clear_mem() deletes the hoisted mpnn model's RNG key
        name = f"traj_{i}"
        m = design_model(cond_cmap, cond_cmap_mask, binder_len)
        m.design_3stage(*DESIGN_STAGES)
        # mechanism check: how far the best stage-3 iterate beats the last one
        loss_last = float(m.aux['log']['cmap_loss_binder'])
        best_aux = m._tmp.get('best', {}).get('aux')
        loss_best = (float(best_aux['log']['cmap_loss_binder'])
                     if best_aux and 'cmap_loss_binder' in best_aux.get('log', {}) else loss_last)
        traj_loss.append((loss_last, loss_best))
        # the prediction model is reused across both arms' samples (Perf #1)
        pred = design_model(cond_cmap, cond_cmap_mask, binder_len)
        per_traj = {}
        for arm, opt in arms.items():
            tdir = os.path.join(args.out, arm)
            os.makedirs(os.path.join(tdir, "traj"), exist_ok=True)
            tpdb = os.path.join(tdir, "traj", f"{name}.pdb")
            m.save_pdb(tpdb, get_best=opt["get_best"])
            r = mpnn_and_predict(tpdb, name, mpnn, pred, cond_cmap, cond_cmap_mask,
                                 binder_len, args.mpnn_samples, args.mpnn_temp, tdir)
            rows[arm].extend(r)
            per_traj[arm] = sum(passes(x) for x in r)
            pd.DataFrame(rows[arm]).to_csv(os.path.join(tdir, "results.csv"), index=False)
        print(f"[{i}/{args.n}] {name}: traj cmap_loss last={loss_last:.3f} "
              f"best={loss_best:.3f} | passes last={per_traj['last']} "
              f"best={per_traj['best']} (of {args.mpnn_samples})")

    # summary
    print("\n=== SUMMARY (paired, n={} trajectories) ===".format(args.n))
    for arm in arms:
        df = pd.DataFrame(rows[arm])
        npass = int(df.apply(passes, axis=1).sum())
        print(f"  {arm:5s}: {len(df)} designs, {npass} pass gate "
              f"({100*npass/len(df):.0f}%), median iptm {df['iptm'].median():.3f}, "
              f"median ipae {df['ipae'].median():.3f}, median cmap_loss {df['cmap_loss'].median():.3f}")
    tl = np.array(traj_loss)
    print(f"  traj cmap_loss (mechanism): last {tl[:,0].mean():.3f} vs best "
          f"{tl[:,1].mean():.3f}, mean gap {(tl[:,0]-tl[:,1]).mean():.3f} "
          f"(a small gap => get_best can't help much)")
    print("  -> next: score both arms with baseline/score_openmm.py for the "
          "family-neutral interface energy, then compare the paired deltas.")


if __name__ == "__main__":
    main()
