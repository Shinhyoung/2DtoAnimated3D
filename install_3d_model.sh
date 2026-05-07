#!/usr/bin/env bash
# 3D mesh generation model installer.
#
# Selection rationale (12GB VRAM, RTX 5070 target):
#   1) TripoSR  (CHOSEN) — ~6GB VRAM, MIT, trimesh export, single-image direct.
#   2) InstantMesh        — ~16-20GB VRAM, multi-view diffusion. FALLBACK if TripoSR fails.
#   3) Unique3D           — ~18-22GB VRAM. Unsuitable for 12GB.
#
# If TripoSR exhausts VRAM on a particular input (large resolution, mc_resolution>=512),
# fall back to InstantMesh with the script below (commented out).

set -e

ENV_NAME="img2mesh"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
TRIPO_DIR="${PROJ_DIR}/TripoSR"

echo "[install] Activating conda env '${ENV_NAME}'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

if [ ! -d "${TRIPO_DIR}" ]; then
    echo "[install] Cloning TripoSR..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git "${TRIPO_DIR}"
else
    echo "[install] TripoSR already cloned."
fi

echo "[install] Installing TripoSR dependencies..."
# These are also pinned in environment.yml; this script is idempotent
# in case TripoSR is added to an existing env without re-running setup.
pip install --upgrade \
    "transformers==4.46.3" \
    "tokenizers<0.21" \
    "huggingface-hub<0.27" \
    "trimesh>=4.0.5" \
    "omegaconf>=2.3.0" \
    "einops>=0.7.0" \
    "imageio[ffmpeg]" \
    "xatlas>=0.0.9" \
    "moderngl>=5.10.0" \
    "rich>=13.7.0"

# Marching cubes: torchmcubes (CUDA build) requires CUDA-VS integration.
# On systems where the integration is missing (No CUDA toolset found),
# fall back to PyMCubes (CPU). generate_mesh.py installs a torchmcubes
# shim that delegates to PyMCubes when the CUDA build is unavailable.
pip install --upgrade "PyMCubes>=0.1.4"

echo "[install] Done. TripoSR installed at: ${TRIPO_DIR}"
echo "[install] Test: python generate_mesh.py --input output/nobg/robot.png"

# ---------------------------------------------------------------------------
# FALLBACK: InstantMesh (uncomment if TripoSR is insufficient)
# ---------------------------------------------------------------------------
# git clone https://github.com/TencentARC/InstantMesh.git "${PROJ_DIR}/InstantMesh"
# cd "${PROJ_DIR}/InstantMesh"
# pip install -r requirements.txt
# # InstantMesh needs >=16GB VRAM at default settings; reduce
# #   `--diffusion_steps 50 -> 30` and `--view_size 320 -> 256` in run.py
# #   to fit 12GB, but quality drops.
