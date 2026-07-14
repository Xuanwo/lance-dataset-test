#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-random-blob-parquet-variants-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/random-blob-parquet-variants/${RUN_ID}}"
ITERS="${ITERS:-1000}"
REPETITIONS="${REPETITIONS:-3}"
SEED_BASE="${SEED_BASE:-2026071400}"
OPENVID_ROWS="${OPENVID_ROWS:-1000000}"
LAION_ROWS="${LAION_ROWS:-200000}"
VERIFY_SAMPLES="${VERIFY_SAMPLES:-32}"
PARQUET_DATA_PAGE_SIZE_LIMIT="${PARQUET_DATA_PAGE_SIZE_LIMIT:-65536}"
PARQUET_WRITE_BATCH_SIZE="${PARQUET_WRITE_BATCH_SIZE:-1}"
PARQUET_MAX_ROW_GROUP_BYTES="${PARQUET_MAX_ROW_GROUP_BYTES:-134217728}"
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
  printf 'verify_samples=%s\n' "${VERIFY_SAMPLES}"
  printf 'parquet_data_page_size_limit=%s\n' "${PARQUET_DATA_PAGE_SIZE_LIMIT}"
  printf 'parquet_write_batch_size=%s\n' "${PARQUET_WRITE_BATCH_SIZE}"
  printf 'parquet_max_row_group_bytes=%s\n' "${PARQUET_MAX_ROW_GROUP_BYTES}"
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
  parquet_default_path="${DATASET_DIR}/${dataset}.default.parquet"
  parquet_random_blob_path="${DATASET_DIR}/${dataset}.random-blob.parquet"
  lance_path="${DATASET_DIR}/${dataset}.lance.v2_2"
  rm -rf "${parquet_default_path}" "${parquet_random_blob_path}" "${lance_path}"

  "${BIN}" ingest \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --input "${inputs[${dataset}]}" \
    --out "${parquet_default_path}" \
    --batch-size 8192 \
    --limit-rows "${limits[${dataset}]}" \
    --parquet-writer-profile default \
    --result-out "${JSON_DIR}/ingest.${dataset}.parquet-default.json" \
    > "${LOG_DIR}/ingest.${dataset}.parquet-default.log" 2>&1

  "${BIN}" ingest \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --input "${inputs[${dataset}]}" \
    --out "${parquet_random_blob_path}" \
    --batch-size 8192 \
    --limit-rows "${limits[${dataset}]}" \
    --parquet-writer-profile random-blob \
    --parquet-data-page-size-limit "${PARQUET_DATA_PAGE_SIZE_LIMIT}" \
    --parquet-write-batch-size "${PARQUET_WRITE_BATCH_SIZE}" \
    --parquet-max-row-group-bytes "${PARQUET_MAX_ROW_GROUP_BYTES}" \
    --result-out "${JSON_DIR}/ingest.${dataset}.parquet-random-blob.json" \
    > "${LOG_DIR}/ingest.${dataset}.parquet-random-blob.log" 2>&1

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

  default_rows="$(jq -r '.rows' "${JSON_DIR}/ingest.${dataset}.parquet-default.json")"
  random_blob_rows="$(jq -r '.rows' "${JSON_DIR}/ingest.${dataset}.parquet-random-blob.json")"
  lance_rows="$(jq -r '.rows' "${JSON_DIR}/ingest.${dataset}.lance.json")"
  if [[ "${default_rows}" != "${random_blob_rows}" || "${default_rows}" != "${lance_rows}" ]]; then
    printf '%s row count mismatch: parquet-default=%s parquet-random-blob=%s lance=%s\n' \
      "${dataset}" "${default_rows}" "${random_blob_rows}" "${lance_rows}" >&2
    exit 1
  fi

  "${BIN}" --seed "$((SEED_BASE + 99))" verify-blob \
    --dataset "${dataset_args[${dataset}]}" \
    --lance-path "${lance_path}" \
    --parquet-path "${parquet_default_path}" \
    --column "${columns[${dataset}]}" \
    --samples "${VERIFY_SAMPLES}" \
    --parquet-read-mode sequential \
    --result-out "${JSON_DIR}/verify.${dataset}.parquet-default-sequential.json" \
    > "${LOG_DIR}/verify.${dataset}.parquet-default-sequential.log" 2>&1

  "${BIN}" --seed "$((SEED_BASE + 99))" verify-blob \
    --dataset "${dataset_args[${dataset}]}" \
    --lance-path "${lance_path}" \
    --parquet-path "${parquet_default_path}" \
    --column "${columns[${dataset}]}" \
    --samples "${VERIFY_SAMPLES}" \
    --parquet-read-mode row-selection \
    --result-out "${JSON_DIR}/verify.${dataset}.parquet-default-row-selection.json" \
    > "${LOG_DIR}/verify.${dataset}.parquet-default-row-selection.log" 2>&1

  "${BIN}" --seed "$((SEED_BASE + 99))" verify-blob \
    --dataset "${dataset_args[${dataset}]}" \
    --lance-path "${lance_path}" \
    --parquet-path "${parquet_random_blob_path}" \
    --column "${columns[${dataset}]}" \
    --samples "${VERIFY_SAMPLES}" \
    --parquet-read-mode row-selection \
    --result-out "${JSON_DIR}/verify.${dataset}.parquet-random-blob-row-selection.json" \
    > "${LOG_DIR}/verify.${dataset}.parquet-random-blob-row-selection.log" 2>&1

  "${BIN}" size \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${parquet_default_path}" \
    --result-out "${JSON_DIR}/size.${dataset}.parquet-default.json"
  "${BIN}" size \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${parquet_random_blob_path}" \
    --result-out "${JSON_DIR}/size.${dataset}.parquet-random-blob.json"
  "${BIN}" size \
    --engine lance \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${lance_path}" \
    --result-out "${JSON_DIR}/size.${dataset}.lance.json"
done

drop_page_cache() {
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

run_one() {
  local dataset="$1"
  local mode="$2"
  local repetition="$3"
  local implementation="$4"
  local seed="$5"
  local engine
  local path
  local read_mode="sequential"
  local writer_profile="default"

  case "${implementation}" in
    lance-v2.2)
      engine="lance"
      path="${DATASET_DIR}/${dataset}.lance.v2_2"
      ;;
    parquet-default-writer-sequential-reader)
      engine="parquet"
      path="${DATASET_DIR}/${dataset}.default.parquet"
      ;;
    parquet-default-writer-row-selection-reader)
      engine="parquet"
      path="${DATASET_DIR}/${dataset}.default.parquet"
      read_mode="row-selection"
      ;;
    parquet-random-blob-writer-row-selection-reader)
      engine="parquet"
      path="${DATASET_DIR}/${dataset}.random-blob.parquet"
      read_mode="row-selection"
      writer_profile="random-blob"
      ;;
    *)
      echo "unknown implementation: ${implementation}" >&2
      return 1
      ;;
  esac

  local result="${JSON_DIR}/blob.${dataset}.${mode}.${implementation}.r${repetition}.json"
  local log="${LOG_DIR}/blob.${dataset}.${mode}.${implementation}.r${repetition}.log"
  drop_page_cache
  "${BIN}" --seed "${seed}" blob \
    --engine "${engine}" \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${path}" \
    --column "${columns[${dataset}]}" \
    --iters "${ITERS}" \
    --open-mode "${mode}" \
    --parquet-read-mode "${read_mode}" \
    --parquet-writer-profile "${writer_profile}" \
    --result-out "${result}" \
    > "${log}" 2>&1
}

implementations=(
  lance-v2.2
  parquet-default-writer-sequential-reader
  parquet-default-writer-row-selection-reader
  parquet-random-blob-writer-row-selection-reader
)

for dataset in openvid laion10m; do
  for mode in opened reopen; do
    for repetition in $(seq 1 "${REPETITIONS}"); do
      seed=$((SEED_BASE + repetition))
      start=$(((repetition - 1) % ${#implementations[@]}))
      for step in $(seq 0 $((${#implementations[@]} - 1))); do
        index=$(((start + step) % ${#implementations[@]}))
        run_one "${dataset}" "${mode}" "${repetition}" "${implementations[${index}]}" "${seed}"
      done
    done
  done
done

python3 scripts/summarize_random_blob_parquet_variants.py \
  --run-root "${RUN_ROOT}"

{
  printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  du -sh "${DATASET_DIR}"/*
  sha256sum "${DATA_DIR}/openvid/openvid.parquet"
  find "${DATA_DIR}/laion10m" -maxdepth 1 -name '*.tar' -print0 | sort -z | xargs -0 sha256sum
} > "${RUN_ROOT}/ARTIFACTS.txt"

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
