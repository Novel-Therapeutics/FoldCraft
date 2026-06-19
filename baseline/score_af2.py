"""AF2 interface oracle for designs that lack AF2 scores (i.e. the BoltzProt-1
arm). FoldCraft designs already carry AF2 metrics from their design-time
evaluation (plddt/iptm/ipae in results.csv); BoltzProt-1 is Boltz-family, so AF2
is its *independent* judge and must be run here.

Replicates FoldCraft.py's own evaluation exactly so the AF2 leg is comparable:
colabdesign binder protocol, predict with model_1_ptm + model_2_ptm, 3 recycles,
reading the same aux['log'] metrics (plddt, i_ptm, i_pae -- i_pae normalised 0-1,
matching the iPAE<0.35 gate). Writes af2_plddt / af2_iptm / af2_ipae columns.

The binder sequence comes from results.csv; the target is threaded fresh against
the same PD-L1 structure FoldCraft used, so the score is epitope-agnostic and
self-contained. Idempotent; GPU-only -- run on reg-box-1 in the FoldCraft env.

Usage:  python baseline/score_af2.py <design_dir> [--target PDB] [--sample N]
"""
import argparse
import os
import sys

import pandas as pd

DEFAULT_TARGET = "examples/targets/pd-l1-1.pdb"
_AF = {"model": None}


def af2_score(target_pdb, binder_seq, target_chain="A"):
    """(plddt, i_ptm, i_pae) for the target+binder complex via colabdesign AF2,
    matching FoldCraft.py's predict() call."""
    from colabdesign import mk_afdesign_model, clear_mem
    clear_mem()                                   # avoids the RuntimeError FoldCraft guards
    af = mk_afdesign_model(protocol="binder", use_templates=False)
    af.prep_inputs(pdb_filename=target_pdb, chain=target_chain,
                   binder_len=len(binder_seq))
    af.set_seq(binder_seq)
    af.predict(num_recycles=3, verbose=False,
               models=["model_1_ptm", "model_2_ptm"])
    log = af.aux["log"]
    return round(log["plddt"], 3), round(log["i_ptm"], 3), round(log["i_pae"], 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv (needs a 'sequence' col)")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--sample", type=int, default=0, help="score only N random rows")
    args = ap.parse_args()

    if not os.path.exists(args.target):
        sys.exit(f"target not found: {args.target}")
    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    df = pd.read_csv(csvf)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in ("af2_plddt", "af2_iptm", "af2_ipae"):
        if col not in df.columns:
            df[col] = pd.NA

    todo = df[df["af2_plddt"].isna()]
    if args.sample and len(todo) > args.sample:
        todo = todo.sample(args.sample, random_state=0)
    print(f"{args.design_dir}: {len(todo)} to score "
          f"({len(df) - len(todo)} already done / skipped)")

    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        if pd.isna(row.get("sequence")):
            sys.exit(f"row {row['name']} has no binder sequence")
        plddt, iptm, ipae = af2_score(args.target, str(row["sequence"]))
        df.loc[idx, ["af2_plddt", "af2_iptm", "af2_ipae"]] = [plddt, iptm, ipae]
        if i % 10 == 0 or i == len(todo):
            df.to_csv(csvf, index=False)       # checkpoint for resumability
            print(f"  [{i}/{len(todo)}] {row['name']}: "
                  f"plddt={plddt} iptm={iptm} ipae={ipae}")
    df.to_csv(csvf, index=False)
    print(f"wrote af2_* columns -> {csvf}")


if __name__ == "__main__":
    main()
