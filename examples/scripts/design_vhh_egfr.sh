#!/bin/bash
# FoldCraft VHH nanobody design (VHH framework, binder template ignored)
# Current convention; see examples/VHH_CONVENTIONS.md for historical reproduction.
# Target: EGFR  |  Target hotspots from custom_template/vhh/vhh_egfr.npy (renumbered-from-1)
cd "$(dirname "$0")/../.." || exit 1

python FoldCraft.py \
    --vhh \
    --vhh_convention foldcraft-127-current-v1 \
    --output_folder examples/outputs/design_vhh_egfr \
    --target_template examples/targets/egfr.pdb \
    --target_hotspots '106-107,128-131,138-153' \
    --num_designs 40
