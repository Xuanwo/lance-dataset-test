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

RAW_DIR="${DATA_DIR}/openvid_raw"
OUT_DIR="${DATA_DIR}/openvid"
CSV_REL_PATH="data/train/OpenVid-1M.csv"
TARGET_ROWS="${OPENVID_ROWS:-1000000}"

for attempt in $(seq 1 "${HF_MAX_RETRIES:-20}"); do
  set +e
  rm -rf "${RAW_DIR}" "${OUT_DIR}" || true
  mkdir -p "${RAW_DIR}" "${OUT_DIR}"

  "${HF_CLI}" download nkp37/OpenVid-1M "${CSV_REL_PATH}" \
    --repo-type dataset \
    --max-workers 1 \
    --local-dir "${RAW_DIR}"
  status=$?
  if [[ $status -ne 0 ]]; then
    set -e
    sleep_sec=$((attempt * 10))
    echo "hf download failed (attempt=${attempt}, status=${status}); sleeping ${sleep_sec}s then retrying..." 1>&2
    sleep "${sleep_sec}"
    continue
  fi

  set -e
  set +e
  cargo run -q -p bench-cli -- prepare-openvid \
    --input "${RAW_DIR}/${CSV_REL_PATH}" \
    --out "${OUT_DIR}" \
    --limit-rows "${TARGET_ROWS}" \
    --batch-size 8192
  status=$?
  set -e
  rm -rf "${RAW_DIR}" || true
  if [[ $status -eq 0 ]]; then
    exit 0
  fi

  sleep_sec=$((attempt * 10))
  echo "hf download failed (attempt=${attempt}, status=${status}); sleeping ${sleep_sec}s then retrying..." 1>&2
  sleep "${sleep_sec}"
done

exit 1
