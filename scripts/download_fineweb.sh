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

# FineWeb full dataset is very large. Download a single public parquet shard.
# For benchmarking, cap ingestion with `--limit-rows 10000000`.
FILES=(sample/10BT/000_00000.parquet)

for attempt in $(seq 1 "${HF_MAX_RETRIES:-20}"); do
  set +e
  "${HF_CLI}" download HuggingFaceFW/fineweb "${FILES[@]}" \
    --repo-type dataset \
    --max-workers 1 \
    --local-dir "${DATA_DIR}/fineweb"
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    exit 0
  fi

  sleep_sec=$((attempt * 10))
  echo "hf download failed (attempt=${attempt}, status=${status}); sleeping ${sleep_sec}s then retrying..." 1>&2
  sleep "${sleep_sec}"
done

exit 1
