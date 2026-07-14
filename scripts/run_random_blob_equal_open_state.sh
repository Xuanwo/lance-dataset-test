#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-random-blob-equal-open-state-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/random-blob-equal-open-state/${RUN_ID}}"
ITERS="${ITERS:-1000}"
REPETITIONS="${REPETITIONS:-3}"
SEED_BASE="${SEED_BASE:-2026071400}"
OPENVID_ROWS="${OPENVID_ROWS:-1000000}"
LAION_ROWS="${LAION_ROWS:-200000}"
LANCE_FILE_VERSION="2.2"

BIN="${ROOT_DIR}/target/release/bench"
DATA_DIR="${ROOT_DIR}/data"
DATASET_DIR="${RUN_ROOT}/datasets"
JSON_DIR="${RUN_ROOT}/json"
LOG_DIR="${RUN_ROOT}/logs"

mkdir -p "${DATASET_DIR}" "${JSON_DIR}" "${LOG_DIR}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "benchmark checkout must be clean" >&2
  exit 1
fi

source "${HOME}/.cargo/env"
export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
cargo build --release --locked

if [[ ! -x "${ROOT_DIR}/.venv/bin/hf" ]]; then
  python3 -m venv "${ROOT_DIR}/.venv"
  "${ROOT_DIR}/.venv/bin/python" -m pip install --upgrade pip
  "${ROOT_DIR}/.venv/bin/python" -m pip install 'huggingface_hub[cli]'
fi

export HUGGINGFACE_CLI="${ROOT_DIR}/.venv/bin/hf"
export BENCH_BIN="${BIN}"
export OPENVID_ROWS
bash scripts/download_openvid.sh
bash scripts/download_laion10m.sh

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'benchmark_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'lance_commit=%s\n' '09174bc9f49e372c1e8c13b73c4e21207150faa6'
  printf 'arrow_parquet_version=%s\n' '58.3.0'
  printf 'lance_file_version=%s\n' "${LANCE_FILE_VERSION}"
  printf 'iters=%s\n' "${ITERS}"
  printf 'repetitions=%s\n' "${REPETITIONS}"
  printf 'seed_base=%s\n' "${SEED_BASE}"
  printf 'openvid_rows=%s\n' "${OPENVID_ROWS}"
  printf 'laion_rows=%s\n' "${LAION_ROWS}"
  printf 'rustflags=%s\n' "${RUSTFLAGS}"
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  rustc --version
  cargo --version
  lscpu
  lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS
  findmnt -T "${ROOT_DIR}"
} > "${RUN_ROOT}/ENVIRONMENT.txt"

declare -A inputs=(
  [openvid]="${DATA_DIR}/openvid"
  [laion10m]="${DATA_DIR}/laion10m"
)
declare -A limits=(
  [openvid]="${OPENVID_ROWS}"
  [laion10m]="${LAION_ROWS}"
)
declare -A columns=(
  [openvid]="video_blob"
  [laion10m]="image"
)
declare -A dataset_args=(
  [openvid]="open-vid"
  [laion10m]="laion10m"
)

for dataset in openvid laion10m; do
  parquet_path="${DATASET_DIR}/${dataset}.parquet"
  lance_path="${DATASET_DIR}/${dataset}.lance.v2_2"
  rm -rf "${parquet_path}" "${lance_path}"

  "${BIN}" ingest \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --input "${inputs[${dataset}]}" \
    --out "${parquet_path}" \
    --batch-size 8192 \
    --limit-rows "${limits[${dataset}]}" \
    --result-out "${JSON_DIR}/ingest.${dataset}.parquet.json" \
    > "${LOG_DIR}/ingest.${dataset}.parquet.log" 2>&1

  "${BIN}" ingest \
    --engine lance \
    --dataset "${dataset_args[${dataset}]}" \
    --input "${inputs[${dataset}]}" \
    --out "${lance_path}" \
    --batch-size 8192 \
    --limit-rows "${limits[${dataset}]}" \
    --lance-file-version "${LANCE_FILE_VERSION}" \
    --result-out "${JSON_DIR}/ingest.${dataset}.lance.json" \
    > "${LOG_DIR}/ingest.${dataset}.lance.log" 2>&1

  parquet_rows="$(jq -r '.rows' "${JSON_DIR}/ingest.${dataset}.parquet.json")"
  lance_rows="$(jq -r '.rows' "${JSON_DIR}/ingest.${dataset}.lance.json")"
  if [[ "${parquet_rows}" != "${lance_rows}" ]]; then
    printf '%s row count mismatch: parquet=%s lance=%s\n' \
      "${dataset}" "${parquet_rows}" "${lance_rows}" >&2
    exit 1
  fi
done

drop_page_cache() {
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

run_one() {
  local dataset="$1"
  local mode="$2"
  local repetition="$3"
  local engine="$4"
  local seed="$5"
  local path

  if [[ "${engine}" == "parquet" ]]; then
    path="${DATASET_DIR}/${dataset}.parquet"
  else
    path="${DATASET_DIR}/${dataset}.lance.v2_2"
  fi

  local result="${JSON_DIR}/blob.${dataset}.${mode}.${engine}.r${repetition}.json"
  local log="${LOG_DIR}/blob.${dataset}.${mode}.${engine}.r${repetition}.log"
  drop_page_cache
  "${BIN}" --seed "${seed}" blob \
    --engine "${engine}" \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${path}" \
    --column "${columns[${dataset}]}" \
    --iters "${ITERS}" \
    --open-mode "${mode}" \
    --result-out "${result}" \
    > "${log}" 2>&1
}

for dataset in openvid laion10m; do
  for mode in opened reopen; do
    for repetition in $(seq 1 "${REPETITIONS}"); do
      seed=$((SEED_BASE + repetition))
      if (( repetition % 2 == 1 )); then
        engines=(parquet lance)
      else
        engines=(lance parquet)
      fi
      for engine in "${engines[@]}"; do
        run_one "${dataset}" "${mode}" "${repetition}" "${engine}" "${seed}"
      done
    done
  done
done

python3 scripts/summarize_random_blob_equal_open_state.py \
  --run-root "${RUN_ROOT}"

{
  printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  du -sh "${DATASET_DIR}"/*
  sha256sum "${DATA_DIR}/openvid/openvid.parquet"
  find "${DATA_DIR}/laion10m" -maxdepth 1 -name '*.tar' -print0 | sort -z | xargs -0 sha256sum
} > "${RUN_ROOT}/ARTIFACTS.txt"

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
