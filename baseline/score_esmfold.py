"""Independent fold-fidelity oracle: predict each designed binder *monomer* from
its sequence with ESMFold and check (a) how confidently it folds and (b) whether
that fold matches the binder chain in the design complex.

ESMFold is single-sequence and architecturally independent of both AF2 (FoldCraft
family) and Boltz (BoltzProt family), so it is a neutral fold judge. Two columns:
  esmfold_plddt -- mean pLDDT of the ESMFold monomer (0-100); does the sequence
                   fold into a confident structure at all?
  esmfold_rmsd  -- CA-RMSD between the ESMFold monomer and the design's chain B,
                   superposed; do the two models *agree* on the fold? (For
                   FoldCraft this complements RMSD-to-template; for BoltzProt,
                   which has no intended fold, it is the fold check.)

Layout/idempotency/usage mirror score_boltz2.py. GPU-only -- run on reg-box-1.
Requires `pip install transformers accelerate torch` (downloads facebook/
esmfold_v1, ~2.8 GB on first use).

Usage:  python baseline/score_esmfold.py <design_dir> [--sample N]
"""
import argparse
import os
import sys

import numpy as np
try:
    from .score_cache import ScoreSession
    from .checkpoint_files import esm_snapshot
except ImportError:
    from score_cache import ScoreSession
    from checkpoint_files import esm_snapshot

import pandas as pd
try:
    from .gates import af2_mask
except ImportError:
    from gates import af2_mask
from Bio.PDB import PDBParser, Superimposer
from Bio.SeqUtils import seq1
try:
    from .structure_checks import chain_ca, matching_ca, fractional_plddt_percent
except ImportError:
    from structure_checks import chain_ca, matching_ca, fractional_plddt_percent

_parser = PDBParser(QUIET=True)
_MODEL = None
_TOK = None


def _load_model(model_dir):
    """Lazy-load ESMFold once (kept out of import so the module is light)."""
    global _MODEL, _TOK
    if _MODEL is None:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
        _TOK = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        _MODEL = EsmForProteinFolding.from_pretrained(str(model_dir), local_files_only=True)
        _MODEL = _MODEL.cuda().eval()
        _MODEL.esm = _MODEL.esm.half()           # fp16 ESM trunk -> fits 24 GB
        torch.backends.cuda.matmul.allow_tf32 = True
    return _MODEL, _TOK


def esmfold_predict(seq, model_dir):
    """Return (mean_plddt 0-100, CA coords array Nx3) for the ESMFold monomer."""
    import torch
    model, tok = _load_model(model_dir)
    ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.no_grad():
        out = model(ids)
    plddt = fractional_plddt_percent(out["plddt"][0, :, 1].detach().cpu().numpy())          # CA-atom pLDDT, mean
    # CA coords: positions[-1] is the final recycle; atom index 1 = CA
    pos = out["positions"][-1, 0]                         # (L, atoms, 3)
    ca = pos[:, 1, :].detach().cpu().numpy()
    return plddt, ca


def binder_seq_and_ca(pdb, chain="B"):
    """(binder sequence, CA coords Nx3) of the design's binder chain, in order.

    Read straight from the structure (not results.csv, whose 'sequence' column is
    the full 'target/binder' complex for FoldCraft) so the ESMFold input and the
    RMSD reference come from one consistent source.
    """
    model = _parser.get_structure("d", pdb)[0]
    atoms = chain_ca(model[chain])
    seq = seq1("".join(a.parent.resname for a in atoms), custom_map={"MSE": "M"})
    ca = np.array([a.coord for a in atoms])
    return seq, ca


def ca_rmsd(a, b):
    """RMSD between two equal-order CA coordinate sets after superposition."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    matching_ca(a, b)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('RMSD requires finite matching Nx3 coordinates')
    from Bio.PDB.qcprot import QCPSuperimposer
    sup = QCPSuperimposer()
    sup.set(a, b)
    sup.run()
    return round(sup.get_rms(), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv + designs/<name>.pdb")
    ap.add_argument("--sample", type=int, default=0, help="score only N random rows")
    ap.add_argument("--af2-pass-only", action="store_true",
                    help="score only designs that clear the AF2 gate (consensus candidates)")
    ap.add_argument('--model-dir',help='Local immutable ESMFold snapshot (otherwise resolve/download from Hugging Face)')
    ap.add_argument('--revision',default='main',help='Hugging Face revision resolved to a local snapshot before scoring')
    args = ap.parse_args()
    if args.sample < 0:
        ap.error('--sample must be nonnegative')

    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    global _MODEL, _TOK
    _MODEL = _TOK = None
    model_dir = esm_snapshot(args.model_dir, args.revision)
    with ScoreSession(csvf, __file__, ['esmfold_plddt', 'esmfold_rmsd'],
                      dict(model='facebook/esmfold_v1', tf32=True, trunk='fp16', confidence_units='percent'), extra_inputs=(), structure=True, model_inputs={'snapshot':model_dir}) as session:
        df = pd.read_csv(csvf)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        for col in ("esmfold_plddt", "esmfold_rmsd"):
            if col not in df.columns:
                df[col] = pd.NA

        df = session.prepare(df)
        session.publish(df)  # publish invalidation before expensive inference
        todo = df[df["esmfold_plddt"].isna()]
        if args.af2_pass_only:
            todo = todo[af2_mask(todo)]
        if args.sample and len(todo) > args.sample:
            todo = todo.sample(args.sample, random_state=0)
        print(f"{args.design_dir}: {len(todo)} to score "
              f"({len(df) - len(todo)} already done / skipped)")

        for i, (idx, row) in enumerate(todo.iterrows(), 1):
            pdb = os.path.join(args.design_dir, "designs", f"{row['name']}.pdb")
            if not os.path.exists(pdb):
                sys.exit(f"design PDB missing: {pdb}")
            binder_seq, ca_design = binder_seq_and_ca(pdb)
            try:
                plddt, ca_pred = esmfold_predict(binder_seq, model_dir)
                rmsd = ca_rmsd(ca_pred, ca_design)
            except (RuntimeError, ValueError) as exc:
                session.annotate(row['name'], status='failed', error=str(exc))
                session.publish(df)
                raise
            df.loc[idx, ["esmfold_plddt", "esmfold_rmsd"]] = [round(plddt, 1), rmsd]
            if i % 20 == 0 or i == len(todo):
                session.publish(df)      # checkpoint for resumability
                print(f"  [{i}/{len(todo)}] {row['name']}: plddt={plddt:.1f} rmsd={rmsd}")
        session.publish(df)
        print(f"wrote esmfold_* columns -> {csvf}")


if __name__ == "__main__":
    main()
