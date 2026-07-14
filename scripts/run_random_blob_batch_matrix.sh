#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_ID="${RUN_ID:-random-blob-batch-matrix-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/random-blob-batch-matrix/${RUN_ID}}"
BASE_DATASET_ROOT="${BASE_DATASET_ROOT:-${ROOT_DIR}/results/random-blob-parquet-variants/random-blob-parquet-variants-20260714T093156Z/datasets}"
REPETITIONS="${REPETITIONS:-3}"
SEED_BASE="${SEED_BASE:-2026071600}"
TARGET_BLOBS="${TARGET_BLOBS:-2048}"
MIN_REQUESTS="${MIN_REQUESTS:-64}"
ANN_REQUESTS="${ANN_REQUESTS:-128}"
VERIFY_SAMPLES="${VERIFY_SAMPLES:-64}"
LANCE_FILE_VERSION="2.2"

BIN="${ROOT_DIR}/target/release/bench"
DATA_DIR="${ROOT_DIR}/data"
DATASET_DIR="${RUN_ROOT}/datasets"
TRACE_DIR="${RUN_ROOT}/traces"
JSON_DIR="${RUN_ROOT}/json"
LOG_DIR="${RUN_ROOT}/logs"
PLOT_DIR="${RUN_ROOT}/plots"
CASE_FILE="${RUN_ROOT}/MATRIX.tsv"

mkdir -p "${DATASET_DIR}" "${TRACE_DIR}" "${JSON_DIR}" "${LOG_DIR}" "${PLOT_DIR}"

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
"${ROOT_DIR}/.venv/bin/python" -m pip install 'numpy<3' faiss-cpu matplotlib

export HUGGINGFACE_CLI="${ROOT_DIR}/.venv/bin/hf"
export BENCH_BIN="${BIN}"
export OPENVID_ROWS=1000000
bash scripts/download_openvid.sh
bash scripts/download_laion10m.sh

SIFT_DIR="${DATA_DIR}/sift1m"
mkdir -p "${SIFT_DIR}"
for file in sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs; do
  if [[ ! -s "${SIFT_DIR}/${file}" ]]; then
    "${HUGGINGFACE_CLI}" download qbo-odp/sift1m "${file}" \
      --repo-type dataset \
      --local-dir "${SIFT_DIR}"
  fi
done

declare -A rows=(
  [openvid]=1000000
  [laion10m]=200000
)
declare -A inputs=(
  [openvid]="${DATA_DIR}/openvid"
  [laion10m]="${DATA_DIR}/laion10m"
)
declare -A columns=(
  [openvid]=video_blob
  [laion10m]=image
)
declare -A dataset_args=(
  [openvid]=open-vid
  [laion10m]=laion10m
)

for dataset in openvid laion10m; do
  trace="${TRACE_DIR}/${dataset}.sift-exact-top64.json"
  "${ROOT_DIR}/.venv/bin/python" scripts/prepare_sift_topk_trace.py \
    --groundtruth "${SIFT_DIR}/sift_groundtruth.ivecs" \
    --base "${SIFT_DIR}/sift_base.fvecs" \
    --queries "${SIFT_DIR}/sift_query.fvecs" \
    --row-count "${rows[${dataset}]}" \
    --requests "${ANN_REQUESTS}" \
    --top-k 64 \
    --out "${trace}"
done

for dataset in openvid laion10m; do
  for required in \
    "${BASE_DATASET_ROOT}/${dataset}.lance.v2_2" \
    "${BASE_DATASET_ROOT}/${dataset}.default.parquet"; do
    if [[ ! -e "${required}" ]]; then
      echo "missing reusable benchmark dataset: ${required}" >&2
      exit 1
    fi
  done

  layout_path="${DATASET_DIR}/${dataset}.default-encoding-layout.parquet"
  "${BIN}" ingest \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --input "${inputs[${dataset}]}" \
    --out "${layout_path}" \
    --batch-size 8192 \
    --limit-rows "${rows[${dataset}]}" \
    --parquet-writer-profile random-blob \
    --result-out "${JSON_DIR}/ingest.${dataset}.parquet-default-encoding-layout.json" \
    > "${LOG_DIR}/ingest.${dataset}.parquet-default-encoding-layout.log" 2>&1

  "${BIN}" --seed "$((SEED_BASE + 99))" verify-blob \
    --dataset "${dataset_args[${dataset}]}" \
    --lance-path "${BASE_DATASET_ROOT}/${dataset}.lance.v2_2" \
    --parquet-path "${BASE_DATASET_ROOT}/${dataset}.default.parquet" \
    --column "${columns[${dataset}]}" \
    --samples "${VERIFY_SAMPLES}" \
    --parquet-read-mode row-selection \
    --result-out "${JSON_DIR}/verify.${dataset}.parquet-default.json" \
    > "${LOG_DIR}/verify.${dataset}.parquet-default.log" 2>&1

  "${BIN}" --seed "$((SEED_BASE + 99))" verify-blob \
    --dataset "${dataset_args[${dataset}]}" \
    --lance-path "${BASE_DATASET_ROOT}/${dataset}.lance.v2_2" \
    --parquet-path "${layout_path}" \
    --column "${columns[${dataset}]}" \
    --samples "${VERIFY_SAMPLES}" \
    --parquet-read-mode row-selection \
    --result-out "${JSON_DIR}/verify.${dataset}.parquet-default-encoding-layout.json" \
    > "${LOG_DIR}/verify.${dataset}.parquet-default-encoding-layout.log" 2>&1

  "${BIN}" size \
    --engine lance \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${BASE_DATASET_ROOT}/${dataset}.lance.v2_2" \
    --result-out "${JSON_DIR}/size.${dataset}.lance.json"
  "${BIN}" size \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${BASE_DATASET_ROOT}/${dataset}.default.parquet" \
    --result-out "${JSON_DIR}/size.${dataset}.parquet-default.json"
  "${BIN}" size \
    --engine parquet \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${layout_path}" \
    --result-out "${JSON_DIR}/size.${dataset}.parquet-default-encoding-layout.json"
done

{
  printf 'run_id=%s\n' "${RUN_ID}"
  printf 'benchmark_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'lance_commit=%s\n' '09174bc9f49e372c1e8c13b73c4e21207150faa6'
  printf 'arrow_parquet_version=%s\n' '58.3.0'
  printf 'lance_file_version=%s\n' "${LANCE_FILE_VERSION}"
  printf 'base_dataset_root=%s\n' "${BASE_DATASET_ROOT}"
  printf 'repetitions=%s\n' "${REPETITIONS}"
  printf 'seed_base=%s\n' "${SEED_BASE}"
  printf 'target_blobs=%s\n' "${TARGET_BLOBS}"
  printf 'min_requests=%s\n' "${MIN_REQUESTS}"
  printf 'ann_requests=%s\n' "${ANN_REQUESTS}"
  printf 'parquet_encoding=%s\n' 'writer-defaults'
  printf 'parquet_layout_page_size=%s\n' '1048576'
  printf 'parquet_layout_write_batch_size=%s\n' '1'
  printf 'parquet_layout_max_row_group_bytes=%s\n' '134217728'
  printf 'rustflags=%s\n' "${RUSTFLAGS}"
  printf 'started_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  rustc --version
  cargo --version
  lscpu
  lsblk -o NAME,MODEL,SIZE,TYPE,FSTYPE,MOUNTPOINTS
  findmnt -T "${ROOT_DIR}"
} > "${RUN_ROOT}/ENVIRONMENT.txt"

printf 'implementation\tapi\tselector\tordering\tdistribution\tbatch_size\trequest_concurrency\topen_mode\n' > "${CASE_FILE}"
declare -A seen_cases=()

add_case() {
  local implementation="$1"
  local api="$2"
  local selector="$3"
  local ordering="$4"
  local distribution="$5"
  local batch_size="$6"
  local concurrency="$7"
  local open_mode="$8"
  local key="${implementation}|${api}|${selector}|${ordering}|${distribution}|${batch_size}|${concurrency}|${open_mode}"
  if [[ -n "${seen_cases[${key}]:-}" ]]; then
    return
  fi
  seen_cases[${key}]=1
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${implementation}" "${api}" "${selector}" "${ordering}" \
    "${distribution}" "${batch_size}" "${concurrency}" "${open_mode}" >> "${CASE_FILE}"
}

for distribution in uniform ann-top-k; do
  for batch_size in 1 4 16 64; do
    for concurrency in 1 8 32; do
      for api in singleton-take batched-take planned-read-blobs; do
        add_case lance "${api}" indices preserve "${distribution}" "${batch_size}" "${concurrency}" opened
      done
      add_case parquet-default parquet-row-selection indices preserve "${distribution}" "${batch_size}" "${concurrency}" opened
      add_case parquet-layout parquet-row-selection indices preserve "${distribution}" "${batch_size}" "${concurrency}" opened
    done
  done

  for concurrency in 1 8 32; do
    for selector in indices addresses; do
      for ordering in preserve unordered; do
        add_case lance planned-read-blobs "${selector}" "${ordering}" "${distribution}" 16 "${concurrency}" opened
      done
    done
    for api in singleton-take batched-take; do
      add_case lance "${api}" addresses preserve "${distribution}" 16 "${concurrency}" opened
    done
    add_case parquet-default parquet-row-selection indices unordered "${distribution}" 16 "${concurrency}" opened
    add_case parquet-layout parquet-row-selection indices unordered "${distribution}" 16 "${concurrency}" opened
  done
done

add_case lance planned-read-blobs indices preserve uniform 16 1 reopen
add_case parquet-default parquet-row-selection indices preserve uniform 16 1 reopen
add_case parquet-layout parquet-row-selection indices preserve uniform 16 1 reopen

drop_page_cache() {
  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
}

run_case() {
  local dataset="$1"
  local repetition="$2"
  local implementation="$3"
  local api="$4"
  local selector="$5"
  local ordering="$6"
  local distribution="$7"
  local batch_size="$8"
  local concurrency="$9"
  local open_mode="${10}"
  local engine
  local path
  local writer_profile=default

  case "${implementation}" in
    lance)
      engine=lance
      path="${BASE_DATASET_ROOT}/${dataset}.lance.v2_2"
      ;;
    parquet-default)
      engine=parquet
      path="${BASE_DATASET_ROOT}/${dataset}.default.parquet"
      ;;
    parquet-layout)
      engine=parquet
      path="${DATASET_DIR}/${dataset}.default-encoding-layout.parquet"
      writer_profile=random-blob
      ;;
    *)
      echo "unknown implementation: ${implementation}" >&2
      return 1
      ;;
  esac

  local request_count
  if [[ "${open_mode}" == reopen ]]; then
    request_count=1
  elif [[ "${distribution}" == ann-top-k ]]; then
    request_count="${ANN_REQUESTS}"
  else
    request_count=$((TARGET_BLOBS / batch_size))
    if ((request_count < MIN_REQUESTS)); then
      request_count="${MIN_REQUESTS}"
    fi
  fi

  local seed=$((SEED_BASE + repetition))
  local stem="batch.${dataset}.${implementation}.${api}.${selector}.${ordering}.${distribution}.b${batch_size}.c${concurrency}.${open_mode}.r${repetition}"
  local result="${JSON_DIR}/${stem}.json"
  local log="${LOG_DIR}/${stem}.log"
  local trace_args=()
  if [[ "${distribution}" == ann-top-k ]]; then
    trace_args=(--trace-file "${TRACE_DIR}/${dataset}.sift-exact-top64.json")
  fi

  drop_page_cache
  "${BIN}" --seed "${seed}" blob-batch \
    --engine "${engine}" \
    --dataset "${dataset_args[${dataset}]}" \
    --path "${path}" \
    --column "${columns[${dataset}]}" \
    --requests "${request_count}" \
    --batch-size "${batch_size}" \
    --request-concurrency "${concurrency}" \
    --selector "${selector}" \
    --ordering "${ordering}" \
    --api "${api}" \
    --distribution "${distribution}" \
    "${trace_args[@]}" \
    --open-mode "${open_mode}" \
    --parquet-writer-profile "${writer_profile}" \
    --result-out "${result}" \
    > "${log}" 2>&1
}

mapfile -t cases < <(tail -n +2 "${CASE_FILE}")
for dataset in openvid laion10m; do
  for repetition in $(seq 1 "${REPETITIONS}"); do
    if ((repetition % 2 == 1)); then
      start=0
      end=$((${#cases[@]} - 1))
      step=1
    else
      start=$((${#cases[@]} - 1))
      end=0
      step=-1
    fi
    index="${start}"
    while :; do
      IFS=$'\t' read -r implementation api selector ordering distribution batch_size concurrency open_mode <<< "${cases[${index}]}"
      run_case "${dataset}" "${repetition}" "${implementation}" "${api}" "${selector}" \
        "${ordering}" "${distribution}" "${batch_size}" "${concurrency}" "${open_mode}"
      if [[ "${index}" == "${end}" ]]; then
        break
      fi
      index=$((index + step))
    done
  done
done

"${ROOT_DIR}/.venv/bin/python" scripts/summarize_random_blob_batch_matrix.py \
  --run-root "${RUN_ROOT}"

{
  printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "${SIFT_DIR}/sift_base.fvecs" "${SIFT_DIR}/sift_query.fvecs" "${SIFT_DIR}/sift_groundtruth.ivecs"
  du -sh "${BASE_DATASET_ROOT}"/* "${DATASET_DIR}"/* "${TRACE_DIR}"/*
} > "${RUN_ROOT}/ARTIFACTS.txt"

printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
