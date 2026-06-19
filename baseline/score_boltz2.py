"""Independent interface-confidence oracle: score binder-target design complexes
with the open, MIT-licensed Boltz-2 (NOT the closed Boltz-2.1/API).

This is the cross-family check the benchmark turns on: FoldCraft/BindCraft/
RFdiffusion design with AF2, so AF2 self-scores are biased toward them; Boltz-2
is an architecturally independent judge. (BoltzProt-1 is Boltz-family, so AF2 is
its independent judge -- run baseline/score_af2.py for that leg.) A design
"passes consensus" iff BOTH legs clear threshold.

Note: open Boltz-2 has no protein-protein *affinity* head (small-molecule only),
so this reports interface *confidence* -- ipTM, the A<->B pair ipTM, and complex
pLDDT -- not a Kd.

Layout: a design dir has results.csv (with a 'name' column) + designs/<name>.pdb,
complex = chain A (target) + chain B (binder). FoldCraft: loop over
baseline/repro/<fold>/. BoltzProt: pass baseline/boltzprot/ directly.

Idempotent: rows already carrying boltz2_iptm are skipped, so a killed run resumes.
GPU-only -- run on reg-box-1. Requires `pip install boltz[cuda]` (MIT) + a GPU.

Usage:  python baseline/score_boltz2.py <design_dir> [--sample N] [--no-msa]
        [--diffusion-samples K] [--workdir DIR]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

_parser = PDBParser(QUIET=True)


def chain_sequences(pdb, chains=("A", "B")):
    """(target_seq, binder_seq) from the design complex PDB."""
    model = _parser.get_structure("c", pdb)[0]
    seqs = []
    for cid in chains:
        if cid not in model:
            raise ValueError(f"{pdb}: missing chain {cid}")
        seqs.append(seq1("".join(r.resname for r in model[cid] if r.id[0] == " ")))
    return seqs


def run_boltz2(target_seq, binder_seq, workdir, use_msa=True, diffusion_samples=1):
    """Predict the A(target)+B(binder) complex; return (iptm, pair_iptm, plddt).

    Writes a minimal Boltz YAML and shells out to `boltz predict`, then reads the
    best sample's confidence JSON. `--use_msa_server` fetches the target MSA from
    Boltz's public server (the de novo binder is effectively single-sequence) --
    fine for the public PD-L1 target; pass --no-msa to keep everything local.
    """
    yml = os.path.join(workdir, "complex.yaml")
    with open(yml, "w") as fh:
        fh.write(
            "version: 1\nsequences:\n"
            f"  - protein: {{id: A, sequence: {target_seq}}}\n"
            f"  - protein: {{id: B, sequence: {binder_seq}}}\n"
        )
    # --no_kernels uses the pure-torch triangular-update path; the optimized
    # cuequivariance kernels are an optional dep that is finicky to match to the
    # CUDA build, and the speed cost is acceptable for a few hundred complexes.
    cmd = ["boltz", "predict", yml, "--out_dir", workdir, "--override",
           "--no_kernels", "--diffusion_samples", str(diffusion_samples)]
    if use_msa:
        cmd.append("--use_msa_server")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"boltz predict failed:\n{r.stderr[-2000:]}")

    confs = glob.glob(os.path.join(workdir, "**", "confidence_*.json"), recursive=True)
    if not confs:
        raise RuntimeError(f"no confidence_*.json produced under {workdir}")
    # highest confidence_score sample
    best = max((json.load(open(c)) for c in confs),
               key=lambda d: d.get("confidence_score", d.get("iptm", 0.0)))
    # pair_chains_iptm is keyed by integer chain index as strings ("0"=target A,
    # "1"=binder B); the A<->B interface value is [0][1] (== overall iptm for a
    # 2-chain complex, but kept explicit).
    pair = best.get("pair_chains_iptm", {})
    pair_iptm = (pair.get("0", {}).get("1")
                 if isinstance(pair.get("0"), dict) else None)
    return best.get("iptm"), pair_iptm, best.get("complex_plddt")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("design_dir", help="dir with results.csv + designs/<name>.pdb")
    ap.add_argument("--sample", type=int, default=0,
                    help="score only a random N rows (0 = all); for a fast pass")
    ap.add_argument("--af2-pass-only", action="store_true",
                    help="score only designs that already clear the AF2 gate "
                         "(the consensus candidates), to spare the heavy Boltz-2 pass")
    ap.add_argument("--no-msa", action="store_true",
                    help="single-sequence (no MSA server); keeps data local")
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    csvf = os.path.join(args.design_dir, "results.csv")
    if not os.path.exists(csvf):
        sys.exit(f"no results.csv in {args.design_dir}")
    df = pd.read_csv(csvf)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for col in ("boltz2_iptm", "boltz2_pair_iptm", "boltz2_plddt"):
        if col not in df.columns:
            df[col] = pd.NA

    todo = df[df["boltz2_iptm"].isna()]
    if args.af2_pass_only:
        # AF2 gate, using whichever AF2 columns the dir carries: FoldCraft's
        # design-time plddt/iptm/ipae, or score_af2.py's af2_* for BoltzProt.
        p, i, e = (("plddt", "iptm", "ipae") if "iptm" in df.columns
                   else ("af2_plddt", "af2_iptm", "af2_ipae"))
        todo = todo[(todo[p] > 0.8) & (todo[i] > 0.5) & (todo[e] < 0.35)]
    if args.sample and len(todo) > args.sample:
        todo = todo.sample(args.sample, random_state=0)
    print(f"{args.design_dir}: {len(todo)} to score "
          f"({len(df) - len(todo)} already done / skipped)")

    work_root = args.workdir or tempfile.mkdtemp(prefix="boltz2_")
    for i, (idx, row) in enumerate(todo.iterrows(), 1):
        pdb = os.path.join(args.design_dir, "designs", f"{row['name']}.pdb")
        if not os.path.exists(pdb):
            sys.exit(f"design PDB missing: {pdb}")
        tgt, binder = chain_sequences(pdb)
        wd = os.path.join(work_root, str(row["name"]))
        os.makedirs(wd, exist_ok=True)
        iptm, pair, plddt = run_boltz2(tgt, binder, wd, use_msa=not args.no_msa,
                                       diffusion_samples=args.diffusion_samples)
        df.loc[idx, ["boltz2_iptm", "boltz2_pair_iptm", "boltz2_plddt"]] = [
            iptm, pair, plddt]
        if i % 10 == 0 or i == len(todo):
            df.to_csv(csvf, index=False)   # checkpoint for resumability
            print(f"  [{i}/{len(todo)}] {row['name']}: iptm={iptm}")
    df.to_csv(csvf, index=False)
    print(f"wrote boltz2_* columns -> {csvf}")


if __name__ == "__main__":
    main()
