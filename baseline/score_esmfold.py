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
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

_parser = PDBParser(QUIET=True)
_MODEL = None
_TOK = None


def _load_model():
    """Lazy-load ESMFold once (kept out of import so the module is light)."""
    global _MODEL, _TOK
    if _MODEL is None:
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
        _TOK = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        _MODEL = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        _MODEL = _MODEL.cuda().eval()
        _MODEL.esm = _MODEL.esm.half()           # fp16 ESM trunk -> fits 24 GB
        torch.backends.cuda.matmul.allow_tf32 = True
    return _MODEL, _TOK


def esmfold_predict(seq):
    """Return (mean_plddt 0-100, CA coords array Nx3) for the ESMFold monomer."""
    import torch
    model, tok = _load_model()
    ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.no_grad():
        out = model(ids)
    plddt = float(out["plddt"][0, :, 1].mean())          # CA-atom pLDDT, mean
    # CA coords: positions[-1] is the final recycle; atom index 1 = CA
    pos = out["positions"][-1, 0]                         # (L, atoms, 3)
    ca = pos[:, 1, :].detach().cpu().numpy()
    return plddt, ca


def binder_ca(pdb, chain="B"):
    """CA coords (Nx3) of the design's binder chain, in sequence order."""
    model = _parser.get_structure("d", pdb)[0]
    return np.array([r["CA"].coord for r in model[chain]
                     if r.id[0] == " " and "CA" in r])


def ca_rmsd(a, b):
    """RMSD between two equal-order CA coordinate sets after superposition."""
    n = min(len(a), len(b))
    if n < 3:
        raise ValueError("fewer than 3 comparable CA atoms")
    from Bio.PDB.qcprot import QCPSuperimposer
    sup = QCPSuperimposer()
    sup.set(a[:n].astype(float), b[:n].astype(float))
    sup.run()
    return round(sup.get_rms(), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv + designs/<name>.pdb")
    ap.add_argument("--sample", type=int, default=0, help="score only N random rows")
    args = ap.parse_args()

    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    df = pd.read_csv(csvf)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in ("esmfold_plddt", "esmfold_rmsd"):
        if col not in df.columns:
            df[col] = pd.NA

    todo = df[df["esmfold_plddt"].isna()]
    if args.sample and len(todo) > args.sample:
        todo = todo.sample(args.sample, random_state=0)
    print(f"{args.design_dir}: {len(todo)} to score "
          f"({len(df) - len(todo)} already done / skipped)")

    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        if pd.isna(row.get("sequence")):
            sys.exit(f"row {row['name']} has no binder sequence")
        plddt, ca_pred = esmfold_predict(str(row["sequence"]))
        pdb = os.path.join(args.design_dir, "designs", f"{row['name']}.pdb")
        if not os.path.exists(pdb):
            sys.exit(f"design PDB missing: {pdb}")
        rmsd = ca_rmsd(ca_pred, binder_ca(pdb))
        df.loc[idx, ["esmfold_plddt", "esmfold_rmsd"]] = [round(plddt, 1), rmsd]
        if i % 20 == 0 or i == len(todo):
            df.to_csv(csvf, index=False)      # checkpoint for resumability
            print(f"  [{i}/{len(todo)}] {row['name']}: plddt={plddt:.1f} rmsd={rmsd}")
    df.to_csv(csvf, index=False)
    print(f"wrote esmfold_* columns -> {csvf}")


if __name__ == "__main__":
    main()
