"""Drive a BoltzProt-1 de novo binder-design job against PD-L1 and normalise its
output into the baseline scoring layout (so the oracle re-scoring treats it
exactly like a FoldCraft fold).

BoltzProt-1 is API-only (Boltz API, ~$0.025/prediction, $2k company launch
credit) and was released 2026-06-16, so the SDK schema below is built from the
documented BoltzGen/Tamarind workflow it inherits and MUST be confirmed against
the live API once a key exists. The two pieces that are *not* speculative and
that this file pins down are:

  1. build_request() -- the job spec, matched to FoldCraft's PD-L1 run so the two
     methods design against the *same target and epitope* (fairness rule #1).
  2. ingest() -- convert downloaded BoltzProt complexes (CIF) + sequences into
     ``<out>/designs/<name>.pdb`` (chain A = target, chain B = binder) + a
     ``results.csv``, the same layout add_rmsd.py / add_ipsae.py / score.py read.

submit() is intentionally a stub: it prints the exact request and the steps,
rather than guessing SDK calls. Fill it in once the API key + SDK are available.

Usage:
  python baseline/boltzprot_design.py request --n 100        # print the job spec
  python baseline/boltzprot_design.py ingest --raw <dir> --out baseline/boltzprot
"""
import csv
import json
import os
import sys

# --- fairness: match FoldCraft's PD-L1 run exactly --------------------------
# Same target + epitope FoldCraft conditioned on (see baseline/repro_config.tsv,
# target_hotspots 30-34,50-54,69-76 on pd-l1-1.pdb, renumbered-from-1).
TARGET_PDB = "examples/targets/pd-l1-1.pdb"
EPITOPE_RESIDUES = "30-34,50-54,69-76"
BINDER_TYPE = "protein"      # de novo mini-protein binder (not nanobody) for the
                             # closest comparison to FoldCraft's fold-conditioned
                             # mini-binders; switch to "nanobody" for a VHH arm.
BINDER_LENGTH = [60, 130]    # spans FoldCraft's binder sizes (77-184); keep wide.


def build_request(n_designs):
    """The BoltzProt-1 job spec. Fields follow the documented BoltzGen/Tamarind
    schema; confirm names against the live SDK before submitting."""
    return {
        "task": "de_novo_binder_design",
        "target_structure": TARGET_PDB,      # API wants the PDB uploaded/inlined
        "target_chains": ["A"],
        "binder_type": BINDER_TYPE,
        "binder_length": BINDER_LENGTH,
        "epitope_residues": EPITOPE_RESIDUES,  # targeted design at FoldCraft's site
        "num_designs": n_designs,
        # return raw designs; we apply our own gate, not BoltzProt's pass_filters
        "return_all": True,
    }


def submit(req):
    """STUB: submit the job to the Boltz API and return a job id.

    Pending an API key + confirmed SDK. Two paths once available:
      - pip the Boltz SDK and call its design endpoint, or
      - POST req to api.boltz.bio with the API key header.
    Snapshot the returned model/version string + date into the output dir for
    reproducibility (it is a moving, closed target).
    """
    raise NotImplementedError(
        "BoltzProt-1 submission needs a Boltz API key + confirmed SDK schema.\n"
        "Run `python baseline/boltzprot_design.py request` to print the exact job "
        "spec, submit it via the SDK / Tamarind, download the results, then\n"
        "`python baseline/boltzprot_design.py ingest --raw <dir> --out baseline/boltzprot`."
    )


# --- ingest downloaded results into the baseline scoring layout -------------
def ingest(raw_dir, out_dir, target_chain="A", binder_chain="B"):
    """Convert a directory of BoltzProt complex structures (.cif) + a sequence
    table into ``<out>/designs/<name>.pdb`` + ``<out>/results.csv``.

    Assumptions (override if the real output differs): each design is a CIF
    complex with the target as ``target_chain`` and the binder as
    ``binder_chain``; binder sequences are in ``<raw>/sequences.csv`` (columns
    name,sequence) or recovered from the CIF. Raises loudly on a missing/empty
    input rather than silently producing an empty benchmark.
    """
    from Bio.PDB import MMCIFParser, PDBIO
    from Bio.SeqUtils import seq1

    cifs = sorted(f for f in os.listdir(raw_dir) if f.endswith(".cif"))
    if not cifs:
        sys.exit(f"no .cif designs found in {raw_dir}")
    designs_out = os.path.join(out_dir, "designs")
    os.makedirs(designs_out, exist_ok=True)
    parser, io = MMCIFParser(QUIET=True), PDBIO()

    rows = []
    for cif in cifs:
        name = os.path.splitext(cif)[0]
        struct = parser.get_structure(name, os.path.join(raw_dir, cif))
        model = struct[0]
        for cid in (target_chain, binder_chain):
            if cid not in model:
                sys.exit(f"{cif}: missing chain {cid} (expected target="
                         f"{target_chain}, binder={binder_chain})")
        binder_seq = seq1("".join(r.resname for r in model[binder_chain]
                                  if r.id[0] == " "))
        io.set_structure(struct)
        io.save(os.path.join(designs_out, f"{name}.pdb"))
        rows.append({"name": name, "sequence": binder_seq})

    # carry through any metrics file BoltzProt provides (reference only -- not
    # used for the gate, which comes from the external oracle)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "sequence"])
        w.writeheader()
        w.writerows(rows)
    # record the binder template for RMSD scoring is N/A here (no intended fold);
    # fold fidelity for BoltzProt is judged by ESMFold self-consistency instead.
    print(f"ingested {len(rows)} BoltzProt designs -> {out_dir}/")


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("request", help="print the job spec (no submission)")
    r.add_argument("--n", type=int, default=100)
    s = sub.add_parser("submit", help="submit to the Boltz API (stub)")
    s.add_argument("--n", type=int, default=100)
    g = sub.add_parser("ingest", help="normalise downloaded results")
    g.add_argument("--raw", required=True, help="dir of downloaded .cif designs")
    g.add_argument("--out", default="baseline/boltzprot")
    args = p.parse_args(argv)

    if args.cmd == "request":
        print(json.dumps(build_request(args.n), indent=2))
    elif args.cmd == "submit":
        submit(build_request(args.n))
    elif args.cmd == "ingest":
        ingest(args.raw, args.out)


if __name__ == "__main__":
    main()
