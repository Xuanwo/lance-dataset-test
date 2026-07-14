#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-random-blob-parquet-writer-tuning-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/random-blob-parquet-writer-tuning/${RUN_ID}}"
ITERS="${ITERS:-200}"
REPETITIONS="${REPETITIONS:-2}"
SEED_BASE="${SEED_BASE:-2026071500}"
OPENVID_ROWS="${OPENVID_ROWS:-100000}"
LAION_ROWS="${LAION_ROWS:-5000}"
PARQUET_MAX_ROW_GROUP_BYTES="${PARQUET_MAX_ROW_GROUP_BYTES:-134217728}"

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
  printf 'iters=%s\n' "${ITERS}"
  printf 'repetitions=%s\n' "${REPETITIONS}"
  printf 'seed_base=%s\n' "${SEED_BASE}"
  printf 'openvid_rows=%s\n' "${OPENVID_ROWS}"
  printf 'laion_rows=%s\n' "${LAION_ROWS}"
  printf 'parquet_max_row_group_bytes=%s\n' "${PARQUET_MAX_ROW_GROUP_BYTES}"
  printf 'rustflags=%s\n' "${RUSTFLAGS}"
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  rustc --version
  lscpu
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

candidates=(p16k-b1 p64k-b1 p256k-b1 p64k-b8)
declare -A page_sizes=(
  [p16k-b1]=16384
  [p64k-b1]=65536
  [p256k-b1]=262144
  [p64k-b8]=65536
)
declare -A write_batches=(
  [p16k-b1]=1
  [p64k-b1]=1
  [p256k-b1]=1
  [p64k-b8]=8
)

for dataset in openvid laion10m; do
  for candidate in "${candidates[@]}"; do
    path="${DATASET_DIR}/${dataset}.${candidate}.parquet"
    rm -f "${path}"
    "${BIN}" ingest \
      --engine parquet \
      --dataset "${dataset_args[${dataset}]}" \
      --input "${inputs[${dataset}]}" \
      --out "${path}" \
      --batch-size 8192 \
      --limit-rows "${limits[${dataset}]}" \
      --parquet-writer-profile random-blob \
      --parquet-data-page-size-limit "${page_sizes[${candidate}]}" \
      --parquet-write-batch-size "${write_batches[${candidate}]}" \
      --parquet-max-row-group-bytes "${PARQUET_MAX_ROW_GROUP_BYTES}" \
      --result-out "${JSON_DIR}/ingest.${dataset}.${candidate}.json" \
      > "${LOG_DIR}/ingest.${dataset}.${candidate}.log" 2>&1

    "${BIN}" size \
      --engine parquet \
      --dataset "${dataset_args[${dataset}]}" \
      --path "${path}" \
      --result-out "${JSON_DIR}/size.${dataset}.${candidate}.json"
  done
done

drop_page_cache() {
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

for dataset in openvid laion10m; do
  for mode in opened reopen; do
    for repetition in $(seq 1 "${REPETITIONS}"); do
      seed=$((SEED_BASE + repetition))
      start=$(((repetition - 1) % ${#candidates[@]}))
      for step in $(seq 0 $((${#candidates[@]} - 1))); do
        index=$(((start + step) % ${#candidates[@]}))
        candidate="${candidates[${index}]}"
        result="${JSON_DIR}/tune.${dataset}.${mode}.${candidate}.r${repetition}.json"
        log="${LOG_DIR}/tune.${dataset}.${mode}.${candidate}.r${repetition}.log"
        drop_page_cache
        "${BIN}" --seed "${seed}" blob \
          --engine parquet \
          --dataset "${dataset_args[${dataset}]}" \
          --path "${DATASET_DIR}/${dataset}.${candidate}.parquet" \
          --column "${columns[${dataset}]}" \
          --iters "${ITERS}" \
          --open-mode "${mode}" \
          --parquet-read-mode row-selection \
          --parquet-writer-profile random-blob \
          --result-out "${result}" \
          > "${log}" 2>&1
      done
    done
  done
done

python3 scripts/summarize_random_blob_parquet_writer_tuning.py \
  --run-root "${RUN_ROOT}"

printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RUN_ROOT}/ENVIRONMENT.txt"
printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
