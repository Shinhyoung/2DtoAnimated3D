#!/usr/bin/env bash
# 3D mesh generation model installer.
#
# Two backends are supported (selectable via `--backend` in main.py / app.py):
#   1) TripoSR  (default) — ~6GB VRAM, MIT, vertex-color trimesh export, single-image direct.
#                           Uses regression-style triplane NeRF + marching cubes.
#   2) TripoSG  (optional) — ~8GB VRAM, flow-based diffusion (rectified flow),
#                            geometry-only (no vertex colors), higher topology quality.
#                            Slower (~50 inference steps) but produces cleaner meshes.
#
# Both repos are cloned side-by-side under this directory:
#   img2mesh_pipeline/TripoSR/
#   img2mesh_pipeline/TripoSG/
#
# Weights are downloaded lazily on first run via huggingface_hub.

set -e

ENV_NAME="img2mesh"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
TRIPOSR_DIR="${PROJ_DIR}/TripoSR"
TRIPOSG_DIR="${PROJ_DIR}/TripoSG"

echo "[install] Activating conda env '${ENV_NAME}'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---------------------------------------------------------------------------
# TripoSR
# ---------------------------------------------------------------------------
if [ ! -d "${TRIPOSR_DIR}" ]; then
    echo "[install] Cloning TripoSR..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git "${TRIPOSR_DIR}"
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

# ---------------------------------------------------------------------------
# TripoSG
# ---------------------------------------------------------------------------
if [ ! -d "${TRIPOSG_DIR}" ]; then
    echo "[install] Cloning TripoSG..."
    git clone https://github.com/VAST-AI-Research/TripoSG.git "${TRIPOSG_DIR}"
else
    echo "[install] TripoSG already cloned."
fi

echo "[install] Installing TripoSG dependencies..."
# TripoSG specific deps from its requirements.txt. Notes:
#  - `numpy==1.22.3` pin from upstream is intentionally NOT applied; it
#    conflicts with rembg/onnxruntime in this env. Modern numpy (1.26.x or
#    2.x) works with TripoSG's actual code paths.
#  - `diso` provides differentiable iso-surface extraction used by the
#    TripoSG pipeline. It needs a CUDA build.
#  - `pymeshlab` is required for optional face-count simplification.
#  - `diffusers`, `peft`, `jaxtyping`, `typeguard` are the diffusion pipeline
#    runtime; pin diffusers to a version known to be compatible with the
#    TripoSGPipeline API (>=0.30 is safe at time of writing).
pip install --upgrade \
    "diffusers>=0.30.0" \
    "peft>=0.11.0" \
    "jaxtyping>=0.2.34" \
    "typeguard>=4.3.0" \
    "scikit-image>=0.22.0" \
    "opencv-python>=4.8.0" \
    "pymeshlab>=2023.12" \
    "diso>=0.1.4"

echo "[install] Done."
echo "[install]   TripoSR -> ${TRIPOSR_DIR}"
echo "[install]   TripoSG -> ${TRIPOSG_DIR}"
echo "[install]"
echo "[install] Quick test:"
echo "[install]   python generate_mesh.py --backend triposr --input output/nobg/robot.png"
echo "[install]   python generate_mesh.py --backend triposg --input output/nobg/robot.png"

# ---------------------------------------------------------------------------
# FALLBACK: InstantMesh (uncomment if both above are insufficient)
# ---------------------------------------------------------------------------
# git clone https://github.com/TencentARC/InstantMesh.git "${PROJ_DIR}/InstantMesh"
# cd "${PROJ_DIR}/InstantMesh"
# pip install -r requirements.txt
# # InstantMesh needs >=16GB VRAM at default settings; reduce
# #   `--diffusion_steps 50 -> 30` and `--view_size 320 -> 256` in run.py
# #   to fit 12GB, but quality drops.
