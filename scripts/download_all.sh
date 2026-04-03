#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HUGGINGFACE_CLI="${HUGGINGFACE_CLI:-${ROOT_DIR}/.venv/bin/hf}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_MAX_RETRIES="${HF_MAX_RETRIES:-20}"

bash "${ROOT_DIR}/scripts/download_lerobot.sh"
bash "${ROOT_DIR}/scripts/download_fineweb.sh"
bash "${ROOT_DIR}/scripts/download_laion10m.sh"
bash "${ROOT_DIR}/scripts/download_openvid.sh"
