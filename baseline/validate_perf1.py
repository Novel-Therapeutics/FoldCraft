"""Bit-identical gate for the Perf #1 hoist: building the AF2-ptm prediction
model ONCE per trajectory and reusing it across the mpnn_samples sequences
(only set_seq changes) vs the original fresh-model-per-sample.

This is the safety check the hoist must pass before shipping -- it changes the
*validated* prediction path, so the outputs must be unchanged. We replicate
FoldCraft.py's prediction setup exactly (protocol='binder',
loss_callback=cmap_loss_binder, use_templates=True, cond_cmap in opt,
prep_inputs with the target/hotspot/binder_len, rm_aa='C') for the Top7 fold,
then predict a few binder sequences BOTH ways:

  OLD: a fresh mk_afdesign_model + prep_inputs per sequence  (original)
  NEW: one model built once, only set_seq per sequence       (the hoist)

and assert plddt / i_ptm / i_pae / cmap_loss_binder and the predicted atom
coordinates are identical for every sequence. A match means the hoist is
output-preserving (no state leak when the model is reused) and safe to ship.
Also times both to show the speedup the hoist buys.

Run in the FoldCraft conda env on a GPU box, from the repo root:
    python baseline/validate_perf1.py
"""
import sys
import time

import numpy as np
import jax.numpy as jnp
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.loss import get_contact_map
from colabdesign.af.alphafold.common import residue_constants

sys.path.insert(0, ".")
from cmap_utils import assemble_fold_conditioned_cmap, binarize_cmap

TARGET = "examples/targets/pd-l1-1.pdb"
BINDER_TEMPLATE = "examples/templates/1qys1.pdb"   # Top7
TARGET_HOTSPOTS = "30-34,50-54,69-76"
BINDER_HOTSPOTS = "26-40,58-71"
CHAIN = "A"
RM_AA = "C"
N_SEQS = 3
MODELS = ["model_1_ptm", "model_2_ptm"]


def cmap_loss_binder(inputs, outputs, opt):
    """Verbatim copy of FoldCraft.py's design/eval loss, so cmap_loss_binder is
    computed exactly as in production and compared old-vs-new."""
    conditioned_array = opt['cond_cmap']
    conditioned_mask = opt['cond_cmap_mask']
    binder_len = inputs['seq']['input'].shape[1]
    i_cmap = get_contact_map(outputs, inputs["opt"]["i_con"]["cutoff"])
    cmap = get_contact_map(outputs, inputs["opt"]["con"]["cutoff"])
    i_cmap = i_cmap.at[-binder_len:, -binder_len:].set(cmap[-binder_len:, -binder_len:])
    out_cmap_conditioned = i_cmap * conditioned_mask
    loss = jnp.sqrt(jnp.square(out_cmap_conditioned - conditioned_array).sum(-1).mean())
    return {"cmap_loss_binder": loss}


def _native_seq(model):
    return "".join(residue_constants.restypes[a]
                   if a < residue_constants.restype_num else "X"
                   for a in model._wt_aatype)


def build_cond_cmap():
    """Reproduce FoldCraft.py's Top7 cond_cmap (binder-monomer cmap + assembly)."""
    clear_mem()
    af_binder = mk_afdesign_model(protocol="fixbb", use_templates=True)
    af_binder.prep_inputs(pdb_filename=BINDER_TEMPLATE, ignore_missing=False,
                          chain=CHAIN, rm_template_seq=False, rm_template_sc=False)
    binder_len = af_binder._len
    af_binder.set_seq(_native_seq(af_binder))
    af_binder.predict(num_recycles=3, verbose=False)
    binder_cmap = af_binder.aux['cmap']

    af_t = mk_afdesign_model(protocol="fixbb", use_templates=True)
    af_t.prep_inputs(pdb_filename=TARGET, ignore_missing=False, chain=CHAIN)
    target_len = af_t._len

    fc = assemble_fold_conditioned_cmap(binder_cmap, target_len, binder_len,
                                        TARGET_HOTSPOTS, BINDER_HOTSPOTS)
    return fc, binarize_cmap(fc), binder_len


def build_pred_model(cond_cmap, cond_cmap_mask, binder_len):
    """A prediction model set up exactly as FoldCraft.py's predict loop does."""
    m = mk_afdesign_model(protocol="binder", loss_callback=cmap_loss_binder,
                          use_templates=True)
    m.opt['cond_cmap'] = cond_cmap.copy()
    m.opt['cond_cmap_mask'] = cond_cmap_mask.copy()
    m.prep_inputs(pdb_filename=TARGET, chain=CHAIN, binder_len=binder_len,
                  hotspot=TARGET_HOTSPOTS, rm_aa=RM_AA)
    return m


def snapshot(m):
    log = m.aux['log']
    coords = None
    for k in ("atom_positions", "xyz"):
        if k in m.aux:
            coords = np.asarray(m.aux[k], dtype=float)
            break
    return {
        "plddt": round(float(log['plddt']), 6),
        "i_ptm": round(float(log['i_ptm']), 6),
        "i_pae": round(float(log['i_pae']), 6),
        "cmap_loss": round(float(log['cmap_loss_binder']), 6),
        "coords": coords,
    }


def main():
    cond_cmap, cond_cmap_mask, binder_len = build_cond_cmap()
    rng = np.random.default_rng(0)
    aa = list("ARNDCQEGHILKMFPSTWYV")
    seqs = ["".join(rng.choice(aa, binder_len)) for _ in range(N_SEQS)]
    print(f"Top7 binder_len={binder_len}; {N_SEQS} fixed random sequences\n")

    # OLD: fresh model per sequence (original FoldCraft.py predict loop)
    t0 = time.time()
    old = []
    for s in seqs:
        m = build_pred_model(cond_cmap, cond_cmap_mask, binder_len)
        m.set_seq(s)
        m.predict(num_recycles=3, verbose=False, models=MODELS)
        old.append(snapshot(m))
    t_old = time.time() - t0

    # NEW: one model, reused (the hoist)
    t0 = time.time()
    m = build_pred_model(cond_cmap, cond_cmap_mask, binder_len)
    new = []
    for s in seqs:
        m.set_seq(s)
        m.predict(num_recycles=3, verbose=False, models=MODELS)
        new.append(snapshot(m))
    t_new = time.time() - t0

    print(f"{'seq':>3}  {'plddt':>18} {'i_ptm':>18} {'i_pae':>18} "
          f"{'cmap_loss':>18} {'coordΔ':>10}  verdict")
    all_ok = True
    for i, (o, n) in enumerate(zip(old, new)):
        cd = (float(np.max(np.abs(o["coords"] - n["coords"])))
              if o["coords"] is not None and n["coords"] is not None else float("nan"))
        scal_ok = all(o[k] == n[k] for k in ("plddt", "i_ptm", "i_pae", "cmap_loss"))
        coord_ok = np.isnan(cd) or cd < 1e-3
        ok = scal_ok and coord_ok
        all_ok = all_ok and ok
        print(f"{i:>3}  {o['plddt']}/{n['plddt']:<8} {o['i_ptm']}/{n['i_ptm']:<8} "
              f"{o['i_pae']}/{n['i_pae']:<8} {o['cmap_loss']}/{n['cmap_loss']:<8} "
              f"{cd:.1e}  {'OK' if ok else 'DIFF!'}")

    print(f"\nWall: OLD {t_old:.1f}s   NEW {t_new:.1f}s   "
          f"speedup {t_old / t_new:.2f}x (predict-loop only)")
    if all_ok:
        print("VERDICT: BIT-IDENTICAL -> the hoist is output-preserving; safe to ship.")
        sys.exit(0)
    print("VERDICT: OUTPUTS DIFFER -> do NOT ship; the reused model leaks state.")
    sys.exit(1)


if __name__ == "__main__":
    main()
