#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-run}" # setup | run | setup-and-run

SETUP_ROOT="${SETUP_ROOT:-results/aws-local-vs-s3/setup}"
SETUP_ENV_FILE="${SETUP_ENV_FILE:-${SETUP_ROOT}/env.sh}"
SETUP_NOTE_FILE="${SETUP_NOTE_FILE:-${SETUP_ROOT}/SETUP.md}"

AWS_PROFILE="${AWS_PROFILE-}"
AWS_REGION="${AWS_REGION:-}"
BENCH_S3_BUCKET="${BENCH_S3_BUCKET:-}"
S3_BASE_PREFIX="${S3_BASE_PREFIX:-bench/lance-local-vs-s3}"
BENCH_BIN="${BENCH_BIN:-./target/release/bench}"

DATASET="${DATASET:-fine-web}"
INPUT="${INPUT:-}"
LIMIT_ROWS="${LIMIT_ROWS:-1000000}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
LANCE_FILE_VERSION="${LANCE_FILE_VERSION:-2.2}"
LANCE_DISABLE_COMPRESSION="${LANCE_DISABLE_COMPRESSION:-false}"

SCAN_REPEATS="${SCAN_REPEATS:-3}"
TAKE_ITERS="${TAKE_ITERS:-1000}"
BLOB_ITERS="${BLOB_ITERS:-1000}"
BLOB_COLUMN="${BLOB_COLUMN:-}"

RUN_ID="${RUN_ID:-aws-local-vs-s3-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-results/aws-local-vs-s3/runs/${RUN_ID}}"
JSON_ROOT="${RESULT_ROOT}/json"
LOCAL_DATASET_ROOT="${RESULT_ROOT}/datasets"

SKIP_BUILD="${SKIP_BUILD:-true}"
AWS_PROFILE_ARGS=()

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: ${cmd}" >&2
    exit 1
  fi
}

resolve_profile_args() {
  AWS_PROFILE_ARGS=()
  if [[ -z "${AWS_PROFILE:-}" ]]; then
    unset AWS_PROFILE
    return
  fi
  if [[ -n "${AWS_PROFILE:-}" ]]; then
    AWS_PROFILE_ARGS=(--profile "$AWS_PROFILE")
  fi
}

resolve_region() {
  resolve_profile_args
  if [[ -n "$AWS_REGION" ]]; then
    return
  fi
  AWS_REGION="$(aws configure get region "${AWS_PROFILE_ARGS[@]}" 2>/dev/null || true)"
  if [[ -z "$AWS_REGION" ]]; then
    AWS_REGION="us-east-1"
  fi
}

create_bucket_if_missing() {
  if aws s3api head-bucket --bucket "$BENCH_S3_BUCKET" "${AWS_PROFILE_ARGS[@]}" >/dev/null 2>&1; then
    return 0
  fi

  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws s3api create-bucket \
      --bucket "$BENCH_S3_BUCKET" \
      "${AWS_PROFILE_ARGS[@]}" \
      >/dev/null
  else
    aws s3api create-bucket \
      --bucket "$BENCH_S3_BUCKET" \
      "${AWS_PROFILE_ARGS[@]}" \
      --region "$AWS_REGION" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}" \
      >/dev/null
  fi
}

write_setup_files() {
  mkdir -p "$SETUP_ROOT"

  cat > "$SETUP_ENV_FILE" <<SETUP_ENV
export AWS_REGION='${AWS_REGION}'
export AWS_DEFAULT_REGION='${AWS_REGION}'
export BENCH_S3_BUCKET='${BENCH_S3_BUCKET}'
export S3_BASE_PREFIX='${S3_BASE_PREFIX}'
export BENCH_BIN='${BENCH_BIN}'
SETUP_ENV

  cat > "$SETUP_NOTE_FILE" <<SETUP_NOTE
# AWS Benchmark Setup

- generated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- aws_region: ${AWS_REGION}
- s3_bucket: ${BENCH_S3_BUCKET}
- s3_base_prefix: ${S3_BASE_PREFIX}
- bench_bin: ${BENCH_BIN}

## Reuse

1. source ${SETUP_ENV_FILE}
2. MODE=run DATASET=fine-web ./scripts/run_lance_local_vs_s3.sh
SETUP_NOTE
}

run_setup() {
  require_cmd aws
  require_cmd cargo

  resolve_region

  local account_id
  account_id="$(aws sts get-caller-identity "${AWS_PROFILE_ARGS[@]}" --query Account --output text)"

  if [[ -z "$BENCH_S3_BUCKET" ]]; then
    BENCH_S3_BUCKET="lance-bench-${account_id}-shared"
  fi

  if [[ -n "${AWS_PROFILE:-}" ]]; then
    export AWS_PROFILE
  else
    unset AWS_PROFILE
  fi
  export AWS_REGION
  export AWS_DEFAULT_REGION="$AWS_REGION"

  create_bucket_if_missing

  if [[ "$SKIP_BUILD" != "true" ]]; then
    cargo build -p bench-cli --release
  fi

  write_setup_files

  echo "[setup] completed"
  echo "[setup] env file: ${SETUP_ENV_FILE}"
  echo "[setup] note file: ${SETUP_NOTE_FILE}"
}

load_setup() {
  if [[ ! -f "$SETUP_ENV_FILE" ]]; then
    echo "setup env file not found: ${SETUP_ENV_FILE}" >&2
    echo "run MODE=setup ./scripts/run_lance_local_vs_s3.sh first" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$SETUP_ENV_FILE"

  if [[ -n "${AWS_PROFILE:-}" ]]; then
    export AWS_PROFILE
  else
    unset AWS_PROFILE
  fi
  export AWS_REGION
  export AWS_DEFAULT_REGION
  resolve_profile_args

  if [[ ! -x "$BENCH_BIN" ]]; then
    if [[ "$SKIP_BUILD" == "true" ]]; then
      echo "bench binary not found: ${BENCH_BIN}" >&2
      echo "rerun with SKIP_BUILD=false or build manually" >&2
      exit 1
    fi
    cargo build -p bench-cli --release
  fi

  if ! aws s3api head-bucket --bucket "$BENCH_S3_BUCKET" "${AWS_PROFILE_ARGS[@]}" >/dev/null 2>&1; then
    echo "configured bucket is not accessible: ${BENCH_S3_BUCKET}" >&2
    exit 1
  fi
}

resolve_input() {
  if [[ -n "$INPUT" ]]; then
    return
  fi

  case "$DATASET" in
    fine-web) INPUT="data/fineweb" ;;
    open-vid) INPUT="data/openvid" ;;
    laion10m) INPUT="data/laion10m" ;;
    le-robot-push-t) INPUT="data/lerobot-pusht" ;;
    le-robot-push-t-image) INPUT="data/lerobot-pusht_image" ;;
    *)
      echo "unsupported dataset for default --input mapping: ${DATASET}" >&2
      exit 1
      ;;
  esac
}

resolve_blob_column() {
  if [[ -n "$BLOB_COLUMN" ]]; then
    return
  fi

  case "$DATASET" in
    laion10m) BLOB_COLUMN="image" ;;
    open-vid) BLOB_COLUMN="video_blob" ;;
    *) BLOB_COLUMN="" ;;
  esac
}

run_ingest() {
  local target="$1"
  local out_uri="$2"
  local result_out="${JSON_ROOT}/ingest.${DATASET}.${target}.json"

  local cmd=(
    "$BENCH_BIN" ingest
    --engine lance
    --dataset "$DATASET"
    --input "$INPUT"
    --out "$out_uri"
    --batch-size "$BATCH_SIZE"
    --lance-file-version "$LANCE_FILE_VERSION"
    --result-out "$result_out"
  )

  if [[ "$LIMIT_ROWS" != "" && "$LIMIT_ROWS" != "0" ]]; then
    cmd+=(--limit-rows "$LIMIT_ROWS")
  fi
  "${cmd[@]}"
}

run_workloads() {
  local target="$1"
  local path_uri="$2"

  for mode in full project filter-low filter-high; do
    "$BENCH_BIN" scan \
      --engine lance \
      --dataset "$DATASET" \
      --path "$path_uri" \
      --mode "$mode" \
      --repeats "$SCAN_REPEATS" \
      --result-out "${JSON_ROOT}/scan-${mode}.${DATASET}.${target}.json"
  done

  "$BENCH_BIN" take \
    --engine lance \
    --dataset "$DATASET" \
    --path "$path_uri" \
    --iters "$TAKE_ITERS" \
    --result-out "${JSON_ROOT}/take.${DATASET}.${target}.json"

  if [[ -n "$BLOB_COLUMN" ]]; then
    "$BENCH_BIN" blob \
      --engine lance \
      --dataset "$DATASET" \
      --path "$path_uri" \
      --column "$BLOB_COLUMN" \
      --iters "$BLOB_ITERS" \
      --result-out "${JSON_ROOT}/blob.${DATASET}.${target}.json"
  fi
}

run_benchmark() {
  load_setup
  resolve_input
  resolve_blob_column

  mkdir -p "$JSON_ROOT" "$LOCAL_DATASET_ROOT"

  local local_dataset_uri="${LOCAL_DATASET_ROOT}/${DATASET}.lance"
  local s3_dataset_uri="s3://${BENCH_S3_BUCKET}/${S3_BASE_PREFIX}/${DATASET}/${RUN_ID}/dataset.lance"

  cat > "${RESULT_ROOT}/run.env" <<RUN_ENV
AWS_REGION='${AWS_REGION}'
BENCH_S3_BUCKET='${BENCH_S3_BUCKET}'
S3_BASE_PREFIX='${S3_BASE_PREFIX}'
DATASET='${DATASET}'
INPUT='${INPUT}'
LIMIT_ROWS='${LIMIT_ROWS}'
BATCH_SIZE='${BATCH_SIZE}'
LANCE_FILE_VERSION='${LANCE_FILE_VERSION}'
LANCE_DISABLE_COMPRESSION='${LANCE_DISABLE_COMPRESSION}'
SCAN_REPEATS='${SCAN_REPEATS}'
TAKE_ITERS='${TAKE_ITERS}'
BLOB_ITERS='${BLOB_ITERS}'
BLOB_COLUMN='${BLOB_COLUMN}'
RUN_ID='${RUN_ID}'
RESULT_ROOT='${RESULT_ROOT}'
LOCAL_DATASET_URI='${local_dataset_uri}'
S3_DATASET_URI='${s3_dataset_uri}'
RUN_ENV

  echo "[run] dataset=${DATASET} input=${INPUT}"
  echo "[run] local_dataset_uri=${local_dataset_uri}"
  echo "[run] s3_dataset_uri=${s3_dataset_uri}"
  echo "[run] json_root=${JSON_ROOT}"

  run_ingest "local" "$local_dataset_uri"
  run_ingest "s3" "$s3_dataset_uri"

  run_workloads "local" "$local_dataset_uri"
  run_workloads "s3" "$s3_dataset_uri"

  echo "[done] json results: ${JSON_ROOT}"
  echo "[done] run env: ${RESULT_ROOT}/run.env"
  echo "[done] cleanup example: aws s3 rm --recursive s3://${BENCH_S3_BUCKET}/${S3_BASE_PREFIX}/${DATASET}/${RUN_ID}/"
}

case "$MODE" in
  setup)
    run_setup
    ;;
  run)
    run_benchmark
    ;;
  setup-and-run)
    run_setup
    run_benchmark
    ;;
  *)
    echo "unsupported MODE: ${MODE} (expected: setup | run | setup-and-run)" >&2
    exit 1
    ;;
esac
