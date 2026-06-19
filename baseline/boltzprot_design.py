"""Drive a BoltzProt-1 de novo binder-design job against PD-L1 and normalise its
output into the baseline scoring layout (so the oracle re-scoring treats it
exactly like a FoldCraft fold).

BoltzProt-1 is API-only (Boltz API, $0.05/protein design, $2k company launch
credit; released 2026-06-16). The request body and mmCIF requirements below are
validated against the live `protein:design` endpoint (boltz-api CLI v0.31.1).

  1. build_request() -- the validated job spec, matched to FoldCraft's PD-L1 run
     so the two methods design against the *same target and epitope* (fairness #1).
  2. build_target_cif_b64() -- the non-obvious part: the API requires PDBx/mmCIF
     polymer metadata that naive PDB->CIF drops; gemmi produces a conformant file.
  3. ingest() -- convert downloaded BoltzProt complexes (CIF) into
     ``<out>/designs/<name>.pdb`` + ``results.csv``, the layout add_rmsd.py /
     add_ipsae.py / score.py read. (Adjust chain ids to the real output.)

Submission is via the `boltz-api` CLI (estimate-cost is free; run is metered) --
see run_command(). Requires `gemmi` (target prep) and an authenticated boltz-api.

Usage:
  python baseline/boltzprot_design.py payload --n 200      # write JSON + run cmds
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
MODALITY = "custom_protein"   # de novo mini-protein binder (closest to FoldCraft's
                              # fold-conditioned mini-binders); "nanobody" for a VHH arm.
BINDER_LENGTH_DSL = "70..185"  # variable-length de novo segment spanning all six
                               # FoldCraft fold sizes (77-184); each design samples a
                               # length in-range (recover it post-hoc per design).


def _hotspots_to_0based(hotspots_1based):
    """FoldCraft's 1-based target_hotspots string -> the 0-based residue-index list
    the Boltz API's ``epitope_residues`` expects. Valid because pd-l1-1.pdb is
    contiguously numbered from 1 (verified)."""
    out = []
    for part in hotspots_1based.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [r - 1 for r in out]


def build_target_cif_b64(pdb=TARGET_PDB):
    """base64 PDBx/mmCIF the API accepts. It requires polymer metadata
    (_entity_poly_seq, _struct_asym) that naive PDB->mmCIF converters drop, and it
    keys chain_selection on ``label_asym_id``. gemmi -- with the entity
    full_sequence populated and the subchain forced to a clean 'A' -- produces a
    conformant file (Biopython's MMCIFIO does not)."""
    import base64
    import gemmi
    st = gemmi.read_structure(pdb)
    st.setup_entities()
    seq = [r.name for r in st[0]["A"].get_polymer()]
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Polymer:
            ent.full_sequence = seq
            ent.subchains = ["A"]
    for res in st[0]["A"]:
        res.subchain = "A"                 # clean label_asym_id ('A'), not gemmi's 'Axp'
    st.assign_label_seq_id()
    cif = st.make_mmcif_document(gemmi.MmcifOutputGroups(True)).as_string()
    for tag in ("_entity_poly_seq", "_struct_asym"):
        assert tag in cif, f"gemmi mmCIF missing {tag}"
    return base64.b64encode(cif.encode()).decode()


def build_request(n_designs):
    """The validated `protein:design` request body, matched to FoldCraft's PD-L1
    run: same target + epitope, de novo no_template mini-protein binder."""
    return {
        "binder_specification": {
            "type": "no_template",
            "modality": MODALITY,
            "entities": [{"type": "designed_protein", "chain_ids": ["B"],
                          "value": BINDER_LENGTH_DSL, "modifications": []}],
            "bonds": [],
        },
        "num_proteins": n_designs,
        "target": {
            "type": "structure_template",
            "structure": {"type": "base64", "media_type": "chemical/x-cif",
                          "data": build_target_cif_b64()},
            "chain_selection": {
                "A": {"chain_type": "polymer", "crop_residues": "all",
                      "epitope_residues": _hotspots_to_0based(EPITOPE_RESIDUES)},
            },
        },
    }


def run_command(payload_path, run_dir="baseline/boltzprot/run1",
                idem="foldcraft-pdl1-baseline-70to185-n200-v1"):
    """The exact CLI flow. Estimate first (free, no GPU); keep the idempotency key
    stable so retries don't double-bill. The run record stores ``engine_version``
    for reproducibility (BoltzProt is a moving, closed target)."""
    return (
        f"boltz-api protein:design estimate-cost --input @json://{payload_path}\n"
        f"boltz-api protein:design run --input @json://{payload_path} \\\n"
        f"  --idempotency-key '{idem}' --run-dir {run_dir} --download-mode everything"
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
    w = sub.add_parser("payload", help="write the validated request JSON + print "
                                       "the boltz-api estimate/run commands")
    w.add_argument("--n", type=int, default=200)
    w.add_argument("--out", default="/tmp/boltzprot_payload.json")
    g = sub.add_parser("ingest", help="normalise downloaded results")
    g.add_argument("--raw", required=True, help="dir of downloaded .cif designs")
    g.add_argument("--out", default="baseline/boltzprot")
    args = p.parse_args(argv)

    if args.cmd == "payload":
        with open(args.out, "w") as fh:
            json.dump(build_request(args.n), fh)
        print(f"wrote payload ({args.n} designs) -> {args.out}\n")
        print(run_command(args.out))
    elif args.cmd == "ingest":
        ingest(args.raw, args.out)


if __name__ == "__main__":
    main()
