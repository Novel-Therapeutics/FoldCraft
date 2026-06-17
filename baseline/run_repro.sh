#!/usr/bin/env bash
# Reproduction run against the AUTHOR's exact published configs
# (examples/scripts/design_*.sh in upstream 9a7a094): PD-L1 target pd-l1-1.pdb,
# the author's per-fold templates + target/binder hotspots, num_designs=40,
# default mpnn_samples (5) and design_stages (100,100,20).
#
# Run with the INCLUSIVE set_range fix (verified to reproduce the author's
# shipped examples/cmaps/*.npy). Outputs one scoreable dir per fold:
#   baseline/repro/<fold>/{results.csv,designs/,traj/}
# The FoldCraft conda env must already be ACTIVE (this script does not activate
# it -- see baseline/README.md), keeping it portable across machines.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1

TARGET="examples/targets/pd-l1-1.pdb"
OUT="baseline/repro"
mkdir -p "$OUT"

# fold  template                        target_hotspots      binder_hotspots
CONFIG=$(cat <<'EOF'
top7      examples/templates/1qys1.pdb  30-34,50-54,69-76    26-40,58-71
barrel    examples/templates/6d0t1.pdb  30-34,50-54,69-76    12-15,25-31,40-43
iglike    examples/templates/3sd21.pdb  30-34,50-54,69-76    17-24,39-46
solenoid  examples/templates/3jx81.pdb  30-34,50-54,69-76    7-13,26-33,46-53,64-71,83-88
ankyrin   examples/templates/5aao1.pdb  30-34,50-54,69-76    15-26,48-58,81-91,116-124
tim       examples/templates/5bvl1.pdb  29-35,49-55,68-77    42-65,68-87,89-113
EOF
)

echo "REPRO_START $(date +%H:%M:%S)"
while read -r fold tmpl thot bhot; do
    [ -z "$fold" ] && continue
    echo "===== FOLD $fold (tmpl=$tmpl thot=$thot bhot=$bhot) $(date +%H:%M:%S) ====="
    python FoldCraft.py \
        --output_folder "$OUT/$fold" \
        --binder_template "$tmpl" \
        --target_template "$TARGET" \
        --target_hotspots "$thot" \
        --binder_hotspots "$bhot" \
        --num_designs 40
    # record the template used so RMSD-to-template scoring is unambiguous later
    cp "$tmpl" "$OUT/$fold/template.pdb"
    rows=$(($(wc -l < "$OUT/$fold/results.csv") - 1))
    echo "FOLD_OK $fold rows=$rows $(date +%H:%M:%S)"
done <<< "$CONFIG"
echo "REPRO_DONE $(date +%H:%M:%S)"
