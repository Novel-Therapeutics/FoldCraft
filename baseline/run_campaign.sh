#!/bin/bash
# usage: run_campaign.sh STAGES NDES NSAMP OUTROOT
source /opt/conda/etc/profile.d/conda.sh
conda activate /opt/conda/envs/FoldCraft
cd /root/FoldCraft
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
STAGES=${1:-100,100,20}; NDES=${2:-10}; NSAMP=${3:-10}; OUT=${4:-baseline/runs}
TARGET=framework/test/pd_l1.pdb; THOT="34-39,43-49,11-17"
tail -n +2 baseline/config.tsv | while IFS=$'\t' read fold tmpl bhot blen; do
  [ -z "$fold" ] && continue
  echo "===== FOLD $fold (len=$blen hotspots=$bhot) $(date +%H:%M:%S) ====="
  python -u FoldCraft.py --output_folder "$OUT/$fold" \
    --binder_template "$tmpl" --target_template "$TARGET" \
    --target_hotspots "$THOT" --binder_hotspots "$bhot" \
    --design_stages "$STAGES" --num_designs "$NDES" --mpnn_samples "$NSAMP" \
    && echo "FOLD_OK $fold rows=$(($(wc -l < $OUT/$fold/results.csv 2>/dev/null || echo 1)-1))" \
    || echo "FOLD_FAIL $fold"
done
echo "CAMPAIGN_DONE $(date +%H:%M:%S)"
