"""Prepare the 6 fold-conditioning binder templates for the FoldCraft baseline.

For each fold we take the chain-A residues of the source PDB within a trim range
(from the paper's Supp. Table 1), renumber them sequentially to 1..N (so chain
position == residue number, no gaps/ambiguity), and write the trimmed template.
The Supp. Table "cmap hotspots" are PDB residue numbers; we map them to the new
1..N positions (which is what FoldCraft.py consumes as --binder_hotspots).

Run from the repo root:  python baseline/prep_templates.py
Source PDBs are expected at baseline/templates/<PDBID>.pdb (fetch from RCSB).
Writes baseline/templates/<fold>.pdb and (re)writes baseline/config.tsv.

NOTE: this produces OUR frozen, self-consistent baseline config, not a
bit-faithful reproduction of the paper (the paper's exact template numbering /
hotspot frame isn't recoverable from the repo). See README.md.
"""
import os
import copy
import warnings

warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPL = os.path.join(HERE, "templates")

# fold -> (pdb_id, trim_lo, trim_hi, hotspots_in_PDB_resnums)   [Supp. Table 1]
FOLDS = {
    "top7":     ("1QYS", 3,   94,  "29-42,61-75"),
    "barrel":   ("6D0T", 1,   111, "39-45,26-30"),
    "iglike":   ("3SD2", 42,  118, "91-99,48-53,69-77"),
    "tim":      ("5BVL", 1,   184, "140-153,164-174,3-15"),
    "solenoid": ("3JX8", 172, 269, "241-247,219-223,199-203"),
    "ankyrin":  ("5AAO", 12,  136, "61-68,73-94,127-134"),
}

# Target (PD-L1) used for all 6 folds. target_hotspots are Supp. Table 2's
# PD-L1 values applied to the shipped framework/test/pd_l1.pdb (1..112).
TARGET = "framework/test/pd_l1.pdb"
TARGET_HOTSPOTS = "34-39,43-49,11-17"


def main():
    p = PDBParser(QUIET=True)
    cfg = open(os.path.join(HERE, "config.tsv"), "w")
    cfg.write("fold\ttemplate\tbinder_hotspots\tbinder_len\n")
    print(f"{'fold':9} {'len':>4}  {'binder_hotspots':24} gaps_in_hotspots")
    for fold, (pid, lo, hi, hot) in FOLDS.items():
        s = p.get_structure(pid, os.path.join(TEMPL, f"{pid}.pdb"))
        chain_a = list(s[0])[0]
        sel = sorted(
            (r for r in chain_a if r.id[0] == " " and lo <= r.id[1] <= hi),
            key=lambda r: r.id[1],
        )
        new_of = {r.id[1]: i + 1 for i, r in enumerate(sel)}

        ns = Structure("x"); nm = Model(0); nc = Chain("A")
        ns.add(nm); nm.add(nc)
        for i, r in enumerate(sel):
            nr = copy.deepcopy(r); nr.id = (" ", i + 1, " "); nc.add(nr)
        io = PDBIO(); io.set_structure(ns)
        io.save(os.path.join(TEMPL, f"{fold}.pdb"))

        def mapp(tok):
            a, b = (int(tok), int(tok)) if "-" not in tok else map(int, tok.split("-"))
            pres = [r for r in range(a, b + 1) if r in new_of]
            return f"{new_of[pres[0]]}-{new_of[pres[-1]]}", (len(pres) != (b - a + 1))

        parts, gap = [], False
        for tok in hot.split(","):
            m, g = mapp(tok); parts.append(m); gap = gap or g
        bh = ",".join(parts)
        print(f"{fold:9} {len(sel):>4}  {bh:24} {'yes(present-only)' if gap else 'no'}")
        cfg.write(f"{fold}\tbaseline/templates/{fold}.pdb\t{bh}\t{len(sel)}\n")
    cfg.close()
    print(f"\nTarget: {TARGET}  target_hotspots={TARGET_HOTSPOTS}")
    print("wrote baseline/config.tsv")


if __name__ == "__main__":
    main()
