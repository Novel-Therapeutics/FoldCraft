#!/bin/bash
# Turnkey FoldCraft setup for a fresh Lambda A100 (x86_64) instance.
#
# Brings up the two conda envs the design + scoring work needs:
#   * FoldCraft  -- jax[cuda12]==0.4.30 + ColabDesign + AF2 params (via the repo's
#                   install_foldcraft.sh); runs FoldCraft.py and validate_perf1.py.
#   * mm         -- OpenMM + pdbfixer for the family-neutral interface energy
#                   (baseline/score_openmm.py), CUDA pinned <= the driver.
#
# Usage (from the repo root, after cloning the fork):
#   git clone -b pure-tier-refactor https://github.com/Novel-Therapeutics/FoldCraft
#   cd FoldCraft && bash baseline/lambda_setup.sh
#
# NOTE: install_foldcraft.sh also installs PyRosetta (graylab academic build).
# FoldCraft.py / validate_perf1.py do NOT use it, but it is pulled in regardless;
# PyRosetta needs a commercial licence for commercial use -- it stays unused here.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || {
  echo "no nvidia-smi -- is this a GPU instance?"; exit 1; }
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
CUDA_MAJOR=${CUDA_VER%%.*}
echo "driver CUDA: ${CUDA_VER:-unknown}"

# --- conda --- (resolve CONDA_BASE robustly: conda may be installed but not on
# PATH after a batch miniconda install, and a re-run must not reinstall over an
# existing miniconda dir)
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [ -d "$HOME/miniconda3" ]; then
  CONDA_BASE="$HOME/miniconda3"
else
  echo "=== installing miniconda ==="
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$HOME/miniconda3"
  CONDA_BASE="$HOME/miniconda3"
fi
export PATH="$CONDA_BASE/condabin:$CONDA_BASE/bin:$PATH"   # conda binary for subshells
source "$CONDA_BASE/etc/profile.d/conda.sh"                # conda activate in this shell

# --- FoldCraft env (design): env + AF2 params + ColabDesign, via the repo installer ---
if conda env list | grep -qw FoldCraft; then
  echo "=== FoldCraft env already present, skipping install ==="
else
  echo "=== install_foldcraft.sh (FoldCraft env + AF2 params + ColabDesign) ==="
  bash install_foldcraft.sh -c "${CUDA_VER:-12.4}"
fi
echo "=== verify jax sees the GPU ==="
conda run -n FoldCraft python -c "import jax; d=jax.devices(); print('jax devices:', d); assert any('cuda' in str(x).lower() or 'gpu' in str(x).lower() for x in d), 'no GPU visible to jax'"

# --- mm env (scoring): OpenMM + pdbfixer, CUDA pinned to the driver (<= avoids the
#     PTX-too-new error: openmm's default build can target CUDA newer than the driver) ---
if conda env list | grep -qw '^mm '; then
  echo "=== mm env already present, skipping ==="
else
  echo "=== mm env (OpenMM + pdbfixer), cuda-version<=${CUDA_VER:-12.4} ==="
  conda create -y -n mm -c conda-forge python=3.11 \
      "cuda-version<=${CUDA_VER:-12.4}" openmm pdbfixer numpy pandas biopython
fi
conda run -n mm python -c "from openmm import Platform; print('openmm platforms:', [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())])"

cat <<EOF

=== DONE ===
  conda activate FoldCraft   # design / prediction (FoldCraft.py, validate_perf1.py)
  conda activate mm          # OpenMM interface-energy scoring (score_openmm.py)

Next:
  conda activate FoldCraft && python baseline/validate_perf1.py   # Perf #1 bit-identical gate
EOF
