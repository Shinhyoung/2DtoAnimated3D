#!/usr/bin/env bash
# Generate Mixamo-named BVH motion files into motions/.
# Note: real CMU/Mixamo motions require account login + non-trivial bone mapping;
# we generate procedural motions with bone names matching skeleton_config.json
# so retargeting is a 1:1 name match.

set -e

ENV_NAME="img2mesh"
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python "${PROJ_DIR}/motions/generate_bvh.py"
echo "[motions] available: $(ls ${PROJ_DIR}/motions/*.bvh | xargs -n1 basename | tr '\n' ' ')"
