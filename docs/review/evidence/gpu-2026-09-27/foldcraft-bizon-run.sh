#!/usr/bin/env bash
set -euo pipefail
root=/home/bizon/projects/foldcraft-gpu-20260927
run_name=${1:?Run name required}
recycles=${2:?Recycle count required}
cd "$root/repo"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="$root/jax-cache"
"$root/venv/bin/pip" freeze > "$root/environment.freeze.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 2 > "$root/$run_name.gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
"$root/venv/bin/python" scripts/gpu_smoke.py --output-root "$root/$run_name" --data-dir "$root/weights" --recycles "$recycles" --execute > "$root/$run_name.runner.log" 2>&1
