#!/usr/bin/env bash
set -e

ENV_NAME="img2mesh"

if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "[setup] '${ENV_NAME}' already exists. Updating..."
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
    echo "[setup] Creating conda env '${ENV_NAME}'..."
    conda env create -f environment.yml
fi

echo "[setup] Done. Activate with: conda activate ${ENV_NAME}"
