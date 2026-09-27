"""AF2 interface oracle for designs that lack AF2 scores (i.e. the BoltzProt-1
arm). FoldCraft designs already carry AF2 metrics from their design-time
evaluation (plddt/iptm/ipae in results.csv); BoltzProt-1 is Boltz-family, so AF2
is its *independent* judge and must be run here.

Replicates FoldCraft.py's own evaluation exactly so the AF2 leg is comparable:
colabdesign binder protocol, predict with model_1_ptm (explicit historical single-model protocol), 3 recycles,
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
from pathlib import Path
from tempfile import TemporaryDirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from input_validation import read_chain
from run_state import stage_seed
from baseline.checkpoint_files import af2_checkpoint

try:
    from .score_cache import ScoreSession
except ImportError:
    from score_cache import ScoreSession

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = str(ROOT / "examples/targets/pd-l1-1.pdb")
_AF = {"model": None}


def af2_score(target_pdb, binder_seq, target_chain="A", hotspot=None, data_dir=ROOT, seed=0):
    """(plddt, i_ptm, i_pae) for the target+binder complex via colabdesign AF2,
    matching FoldCraft.py's predict() exactly so the AF2 leg is comparable across
    methods: protocol='binder', use_templates=True (templates the TARGET, which is
    what lets AF2 place the binder -- without it ipTM is systematically ~0.3 lower
    for everyone), the epitope hotspot, model_1_ptm, 3 recycles."""
    from colabdesign import mk_afdesign_model, clear_mem
    clear_mem()                                   # avoids the RuntimeError FoldCraft guards
    target = read_chain(target_pdb, target_chain)
    mapped = target.selection(hotspot) if hotspot else None
    af = mk_afdesign_model(data_dir=str(data_dir), protocol="binder", use_templates=True, model_names=["model_1_ptm"])
    with TemporaryDirectory(prefix='foldcraft-af2-target-') as folder:
        canonical = Path(folder) / 'target.pdb'
        canonical.write_text(target.pdb)
        af.prep_inputs(pdb_filename=str(canonical), chain=target_chain,
                       binder_len=len(binder_seq), hotspot=mapped)
    af.set_seq(binder_seq)
    af.predict(num_recycles=3, verbose=False,
               models=["model_1_ptm"], num_models=1, sample_models=False, seed=seed)
    log = af.aux["log"]
    return round(log["plddt"], 3), round(log["i_ptm"], 3), round(log["i_pae"], 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv (needs a 'sequence' col)")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--hotspot", default="30-34,50-54,69-76",
                    help="target epitope residues (PD-L1 default, matching FoldCraft)")
    ap.add_argument("--sample", type=int, default=0, help="score only N random rows")
    ap.add_argument('--data-dir',default=str(ROOT))
    ap.add_argument('--seed',type=int,default=0)
    args = ap.parse_args()
    if not 0 <= args.seed < 2**32:
        ap.error('--seed must be a 32-bit nonnegative integer')
    if args.sample < 0:
        ap.error('--sample must be nonnegative')

    if not os.path.exists(args.target):
        sys.exit(f"target not found: {args.target}")
    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    checkpoint = af2_checkpoint(args.data_dir)
    with ScoreSession(csvf, __file__, ['af2_plddt', 'af2_iptm', 'af2_ipae'],
                      dict(target=str(args.target), hotspot=args.hotspot, model='model_1_ptm', recycles=3, seed=args.seed), extra_inputs=[args.target], structure=False, model_inputs={'checkpoint':checkpoint}) as session:
        df = pd.read_csv(csvf)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        for col in ("af2_plddt", "af2_iptm", "af2_ipae"):
            if col not in df.columns:
                df[col] = pd.NA

        df = session.prepare(df)
        session.publish(df)  # publish invalidation before expensive inference
        todo = df[df["af2_plddt"].isna()]
        if args.sample and len(todo) > args.sample:
            todo = todo.sample(args.sample, random_state=0)
        print(f"{args.design_dir}: {len(todo)} to score "
              f"({len(df) - len(todo)} already done / skipped)")

        for i, (idx, row) in enumerate(todo.iterrows(), 1):
            if pd.isna(row.get("sequence")):
                sys.exit(f"row {row['name']} has no binder sequence")
            # FoldCraft's 'sequence' is the full 'target/binder' complex; take the
            # binder (BoltzProt's is binder-only, so the split is a no-op there).
            binder_seq = str(row["sequence"]).split("/")[-1]
            try:
                plddt, iptm, ipae = af2_score(args.target, binder_seq, hotspot=args.hotspot, data_dir=args.data_dir, seed=stage_seed(args.seed, row["name"]))
            except (RuntimeError, ValueError) as exc:
                session.annotate(row['name'], status='failed', error=str(exc))
                session.publish(df)
                raise
            df.loc[idx, ["af2_plddt", "af2_iptm", "af2_ipae"]] = [plddt, iptm, ipae]
            if i % 10 == 0 or i == len(todo):
                session.publish(df)       # checkpoint for resumability
                print(f"  [{i}/{len(todo)}] {row['name']}: "
                      f"plddt={plddt} iptm={iptm} ipae={ipae}")
        session.publish(df)
        print(f"wrote af2_* columns -> {csvf}")


if __name__ == "__main__":
    main()
