#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/profile_time_profiler.sh --out <dir> --name <case_name> -- <command...>

Records an Instruments "Time Profiler" trace via xctrace, then exports and summarizes it.

Artifacts:
  <dir>/<case_name>/
    - recording.trace
    - toc.xml
    - summary.json
    - summary.txt
    - cmd.txt
    - xctrace.log
    - target.stdout.log  (stdout from the launched target)
EOF
}

OUT_DIR=""
CASE_NAME=""

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --name)
      CASE_NAME="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -z "$OUT_DIR" || -z "$CASE_NAME" || $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

CASE_DIR="$OUT_DIR/$CASE_NAME"
mkdir -p "$CASE_DIR"

TRACE="$CASE_DIR/recording.trace"
TOC_XML="$CASE_DIR/toc.xml"
SUMMARY_JSON="$CASE_DIR/summary.json"
SUMMARY_TXT="$CASE_DIR/summary.txt"
CMD_TXT="$CASE_DIR/cmd.txt"
XCTRACE_LOG="$CASE_DIR/xctrace.log"
TARGET_STDOUT="$CASE_DIR/target.stdout.log"

{
  echo "cwd=$(pwd)"
  echo -n "cmd="
  printf '%q ' "$@"
  echo
} >"$CMD_TXT"

rm -rf "$TRACE"

{
  echo "xctrace record started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  xcrun xctrace record \
    --template 'Time Profiler' \
    --no-prompt \
    --output "$TRACE" \
    --target-stdout "$TARGET_STDOUT" \
    --launch -- "$@"
  echo "xctrace record finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$XCTRACE_LOG" 2>&1

xcrun xctrace export --input "$TRACE" --toc --output "$TOC_XML" >/dev/null

python3 scripts/xctrace_time_profile_summary.py \
  --trace "$TRACE" \
  --top 50 \
  --format json \
  --output "$SUMMARY_JSON"

python3 scripts/xctrace_time_profile_summary.py \
  --trace "$TRACE" \
  --top 50 \
  --format text \
  --output "$SUMMARY_TXT"

echo "ok: $CASE_DIR"

