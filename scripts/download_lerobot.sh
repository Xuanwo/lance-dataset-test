#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
HF_CLI="${HUGGINGFACE_CLI:-}"
if [[ -z "${HF_CLI}" && -x "${ROOT_DIR}/.venv/bin/hf" ]]; then
  HF_CLI="${ROOT_DIR}/.venv/bin/hf"
fi
HF_CLI="${HF_CLI:-hf}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

mkdir -p "${DATA_DIR}"

"${HF_CLI}" download lerobot/pusht --repo-type dataset --max-workers 1 --local-dir "${DATA_DIR}/lerobot-pusht"
"${HF_CLI}" download lerobot/pusht_image --repo-type dataset --max-workers 1 --local-dir "${DATA_DIR}/lerobot-pusht_image"
