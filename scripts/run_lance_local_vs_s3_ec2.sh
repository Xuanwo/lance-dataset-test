#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-setup}" # setup | run | setup-and-run | status

CONTROL_AWS_PROFILE="${CONTROL_AWS_PROFILE:-}"
AWS_REGION="${AWS_REGION:-}"

SETUP_ROOT="${SETUP_ROOT:-results/aws-local-vs-s3/ec2-setup}"
SETUP_ENV_FILE="${SETUP_ENV_FILE:-${SETUP_ROOT}/env.sh}"
SETUP_NOTE_FILE="${SETUP_NOTE_FILE:-${SETUP_ROOT}/SETUP.md}"

INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.4xlarge}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-500}"
ROLE_NAME="${ROLE_NAME:-LanceBenchEc2Role}"
INSTANCE_PROFILE_NAME="${INSTANCE_PROFILE_NAME:-LanceBenchEc2Profile}"
INSTANCE_TAG_NAME="${INSTANCE_TAG_NAME:-lance-bench-runner}"
SUBNET_ID="${SUBNET_ID:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
AMI_ID="${AMI_ID:-}"
INSTANCE_ID="${INSTANCE_ID:-}"

BENCH_S3_BUCKET="${BENCH_S3_BUCKET:-}"
S3_BASE_PREFIX="${S3_BASE_PREFIX:-bench/lance-local-vs-s3}"

DATASET="${DATASET:-fine-web}"
LIMIT_ROWS="${LIMIT_ROWS:-200000}"
SCAN_REPEATS="${SCAN_REPEATS:-2}"
TAKE_ITERS="${TAKE_ITERS:-200}"
BLOB_ITERS="${BLOB_ITERS:-200}"
LANCE_FILE_VERSION="${LANCE_FILE_VERSION:-2.2}"
RUN_ID="${RUN_ID:-ec2-aws-local-vs-s3-$(date -u +%Y%m%dT%H%M%SZ)}"

AWS_PROFILE_ARGS=()
AWS_REGION_ARGS=()
ACCOUNT_ID=""

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing required command: ${cmd}" >&2
    exit 1
  fi
}

resolve_aws_args() {
  AWS_PROFILE_ARGS=()
  if [[ -n "$CONTROL_AWS_PROFILE" ]]; then
    AWS_PROFILE_ARGS=(--profile "$CONTROL_AWS_PROFILE")
  fi

  if [[ -z "$AWS_REGION" ]]; then
    AWS_REGION="$(aws configure get region "${AWS_PROFILE_ARGS[@]}" 2>/dev/null || true)"
  fi
  if [[ -z "$AWS_REGION" ]]; then
    AWS_REGION="us-east-2"
  fi

  AWS_REGION_ARGS=(--region "$AWS_REGION")
}

aws_cli() {
  aws "${AWS_PROFILE_ARGS[@]}" "${AWS_REGION_ARGS[@]}" "$@"
}

ensure_bucket() {
  if [[ -z "$BENCH_S3_BUCKET" ]]; then
    BENCH_S3_BUCKET="lance-bench-${ACCOUNT_ID}-shared"
  fi

  if aws_cli s3api head-bucket --bucket "$BENCH_S3_BUCKET" >/dev/null 2>&1; then
    return
  fi

  if [[ "$AWS_REGION" == "us-east-1" ]]; then
    aws "${AWS_PROFILE_ARGS[@]}" s3api create-bucket --bucket "$BENCH_S3_BUCKET" >/dev/null
  else
    aws_cli s3api create-bucket \
      --bucket "$BENCH_S3_BUCKET" \
      --create-bucket-configuration "LocationConstraint=${AWS_REGION}" \
      >/dev/null
  fi
}

ensure_iam_role_and_profile() {
  local trust_file
  trust_file="$(mktemp)"
  cat > "$trust_file" <<'TRUST'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
TRUST

  if ! aws "${AWS_PROFILE_ARGS[@]}" iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    aws "${AWS_PROFILE_ARGS[@]}" iam create-role \
      --role-name "$ROLE_NAME" \
      --assume-role-policy-document "file://${trust_file}" \
      >/dev/null
  fi

  aws "${AWS_PROFILE_ARGS[@]}" iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore \
    >/dev/null

  local s3_policy_file
  s3_policy_file="$(mktemp)"
  cat > "$s3_policy_file" <<POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": ["arn:aws:s3:::${BENCH_S3_BUCKET}"]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
        "s3:ListBucketMultipartUploads",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": ["arn:aws:s3:::${BENCH_S3_BUCKET}/*"]
    }
  ]
}
POLICY

  aws "${AWS_PROFILE_ARGS[@]}" iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "LanceBenchS3Access" \
    --policy-document "file://${s3_policy_file}" \
    >/dev/null

  if ! aws "${AWS_PROFILE_ARGS[@]}" iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE_NAME" >/dev/null 2>&1; then
    aws "${AWS_PROFILE_ARGS[@]}" iam create-instance-profile \
      --instance-profile-name "$INSTANCE_PROFILE_NAME" \
      >/dev/null
    sleep 5
  fi

  local attached_count
  attached_count="$(aws "${AWS_PROFILE_ARGS[@]}" iam get-instance-profile \
    --instance-profile-name "$INSTANCE_PROFILE_NAME" \
    --query "length(InstanceProfile.Roles[?RoleName=='${ROLE_NAME}'])" \
    --output text)"
  if [[ "$attached_count" == "0" ]]; then
    aws "${AWS_PROFILE_ARGS[@]}" iam add-role-to-instance-profile \
      --instance-profile-name "$INSTANCE_PROFILE_NAME" \
      --role-name "$ROLE_NAME" \
      >/dev/null
    sleep 10
  fi

  rm -f "$trust_file" "$s3_policy_file"
}

resolve_network_defaults() {
  local vpc_id
  vpc_id="$(aws_cli ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  if [[ -z "$vpc_id" || "$vpc_id" == "None" ]]; then
    echo "default VPC not found in region ${AWS_REGION}" >&2
    exit 1
  fi

  if [[ -z "$SUBNET_ID" ]]; then
    SUBNET_ID="$(aws_cli ec2 describe-subnets --filters Name=vpc-id,Values="$vpc_id" --query 'Subnets[0].SubnetId' --output text)"
  fi
  if [[ -z "$SUBNET_ID" || "$SUBNET_ID" == "None" ]]; then
    echo "failed to resolve subnet in VPC ${vpc_id}" >&2
    exit 1
  fi

  if [[ -z "$SECURITY_GROUP_ID" ]]; then
    SECURITY_GROUP_ID="$(aws_cli ec2 describe-security-groups --filters Name=vpc-id,Values="$vpc_id" Name=group-name,Values=default --query 'SecurityGroups[0].GroupId' --output text)"
  fi
  if [[ -z "$SECURITY_GROUP_ID" || "$SECURITY_GROUP_ID" == "None" ]]; then
    echo "failed to resolve default security group in VPC ${vpc_id}" >&2
    exit 1
  fi
}

resolve_ami() {
  if [[ -n "$AMI_ID" ]]; then
    return
  fi
  AMI_ID="$(aws_cli ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64 --query 'Parameter.Value' --output text)"
  if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
    echo "failed to resolve Amazon Linux AMI" >&2
    exit 1
  fi
}

instance_state() {
  local id="$1"
  aws_cli ec2 describe-instances \
    --instance-ids "$id" \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null || true
}

ensure_instance() {
  if [[ -n "$INSTANCE_ID" ]]; then
    local state
    state="$(instance_state "$INSTANCE_ID")"
    if [[ "$state" == "running" ]]; then
      return
    fi
  fi

  local existing
  existing="$(aws_cli ec2 describe-instances \
    --filters "Name=tag:Name,Values=${INSTANCE_TAG_NAME}" "Name=instance-state-name,Values=running,pending,stopping,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' \
    --output text)"

  if [[ -n "$existing" && "$existing" != "None" ]]; then
    INSTANCE_ID="$existing"
    local state
    state="$(instance_state "$INSTANCE_ID")"
    if [[ "$state" == "stopped" || "$state" == "stopping" ]]; then
      aws_cli ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
    fi
    aws_cli ec2 wait instance-running --instance-ids "$INSTANCE_ID"
    return
  fi

  INSTANCE_ID="$(aws_cli ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --iam-instance-profile "Name=${INSTANCE_PROFILE_NAME}" \
    --subnet-id "$SUBNET_ID" \
    --security-group-ids "$SECURITY_GROUP_ID" \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLUME_SIZE_GB},VolumeType=gp3,DeleteOnTermination=true}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_TAG_NAME}},{Key=Project,Value=lance-bench}]" \
    --query 'Instances[0].InstanceId' \
    --output text)"

  aws_cli ec2 wait instance-running --instance-ids "$INSTANCE_ID"
}

wait_ssm_online() {
  local attempts=60
  local i
  for i in $(seq 1 "$attempts"); do
    local status
    status="$(aws_cli ssm describe-instance-information \
      --filters "Key=InstanceIds,Values=${INSTANCE_ID}" \
      --query 'InstanceInformationList[0].PingStatus' \
      --output text 2>/dev/null || true)"
    if [[ "$status" == "Online" ]]; then
      return
    fi
    sleep 10
  done

  echo "instance ${INSTANCE_ID} did not become SSM online in time" >&2
  exit 1
}

run_ssm_script() {
  local script_path="$1"
  local label="$2"
  local remote_script_key="${S3_BASE_PREFIX}/control/${label}-$(date -u +%Y%m%dT%H%M%SZ).sh"
  local remote_script_uri="s3://${BENCH_S3_BUCKET}/${remote_script_key}"

  aws_cli s3 cp "$script_path" "$remote_script_uri" >/dev/null

  local params_file
  params_file="$(mktemp)"
  cat > "$params_file" <<PARAMS
{
  "commands": [
    "aws s3 cp ${remote_script_uri} /tmp/${label}.sh --region ${AWS_REGION}",
    "chmod +x /tmp/${label}.sh",
    "bash /tmp/${label}.sh"
  ]
}
PARAMS

  local command_id
  command_id="$(aws_cli ssm send-command \
    --instance-ids "$INSTANCE_ID" \
    --document-name AWS-RunShellScript \
    --comment "$label" \
    --parameters "file://${params_file}" \
    --timeout-seconds 7200 \
    --query 'Command.CommandId' \
    --output text)"

  local status=""
  local deadline=$((SECONDS + 28800))
  while true; do
    status="$(aws_cli ssm get-command-invocation \
      --command-id "$command_id" \
      --instance-id "$INSTANCE_ID" \
      --query 'Status' \
      --output text 2>/dev/null || true)"
    case "$status" in
      Success|Failed|Cancelled|TimedOut|DeliveryTimedOut|ExecutionTimedOut|Undeliverable|Terminated)
        break
        ;;
      "")
        ;;
      *)
        ;;
    esac
    if (( SECONDS >= deadline )); then
      status="TimedOut"
      break
    fi
    sleep 20
  done

  local out_dir="results/aws-local-vs-s3/ec2-runs/${RUN_ID}"
  mkdir -p "$out_dir"

  aws_cli ssm get-command-invocation \
    --command-id "$command_id" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardOutputContent' \
    --output text > "${out_dir}/${label}.stdout.log"
  aws_cli ssm get-command-invocation \
    --command-id "$command_id" \
    --instance-id "$INSTANCE_ID" \
    --query 'StandardErrorContent' \
    --output text > "${out_dir}/${label}.stderr.log"

  status="$(aws_cli ssm get-command-invocation \
    --command-id "$command_id" \
    --instance-id "$INSTANCE_ID" \
    --query 'Status' \
    --output text)"

  rm -f "$params_file"

  if [[ "$status" != "Success" ]]; then
    echo "SSM command failed: ${label} (${status})" >&2
    echo "stdout: ${out_dir}/${label}.stdout.log" >&2
    echo "stderr: ${out_dir}/${label}.stderr.log" >&2
    exit 1
  fi
}

write_setup_files() {
  mkdir -p "$SETUP_ROOT"
  cat > "$SETUP_ENV_FILE" <<SETUP_ENV
export AWS_REGION='${AWS_REGION}'
export BENCH_S3_BUCKET='${BENCH_S3_BUCKET}'
export S3_BASE_PREFIX='${S3_BASE_PREFIX}'
export INSTANCE_ID='${INSTANCE_ID}'
export INSTANCE_TYPE='${INSTANCE_TYPE}'
SETUP_ENV

  cat > "$SETUP_NOTE_FILE" <<SETUP_NOTE
# EC2 Benchmark Setup

- generated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
- aws_region: ${AWS_REGION}
- instance_id: ${INSTANCE_ID}
- instance_type: ${INSTANCE_TYPE}
- s3_bucket: ${BENCH_S3_BUCKET}
- s3_base_prefix: ${S3_BASE_PREFIX}

## Reuse

1. source ${SETUP_ENV_FILE}
2. MODE=run DATASET=fine-web ./scripts/run_lance_local_vs_s3_ec2.sh
SETUP_NOTE
}

load_setup() {
  if [[ ! -f "$SETUP_ENV_FILE" ]]; then
    echo "missing setup env file: ${SETUP_ENV_FILE}" >&2
    echo "run MODE=setup ./scripts/run_lance_local_vs_s3_ec2.sh first" >&2
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$SETUP_ENV_FILE"

  resolve_aws_args
}

create_bootstrap_script() {
  local script_path
  script_path="$(mktemp)"
  cat > "$script_path" <<'BOOTSTRAP'
#!/usr/bin/env bash
set -euo pipefail

if [[ -f /var/lib/lance_bench/bootstrap_done ]]; then
  if command -v protoc >/dev/null 2>&1 && [[ -f /usr/include/google/protobuf/empty.proto ]]; then
    exit 0
  fi
fi

sudo dnf -y install git tar gzip gcc gcc-c++ make openssl-devel pkgconfig python3-pip protobuf protobuf-compiler protobuf-devel

runuser -u ec2-user -- bash -lc '
set -euo pipefail
if [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
python3 -m pip install --user -U "huggingface_hub[cli]"
'

sudo mkdir -p /var/lib/lance_bench
sudo touch /var/lib/lance_bench/bootstrap_done
BOOTSTRAP
  echo "$script_path"
}

create_remote_run_script() {
  local src_s3_uri="$1"
  local script_path
  script_path="$(mktemp)"

  cat > "$script_path" <<RUNSCRIPT
#!/usr/bin/env bash
set -euo pipefail

export AWS_REGION='${AWS_REGION}'
export AWS_DEFAULT_REGION='${AWS_REGION}'

RUN_ID='${RUN_ID}'
DATASET='${DATASET}'
LIMIT_ROWS='${LIMIT_ROWS}'
SCAN_REPEATS='${SCAN_REPEATS}'
TAKE_ITERS='${TAKE_ITERS}'
BLOB_ITERS='${BLOB_ITERS}'
LANCE_FILE_VERSION='${LANCE_FILE_VERSION}'
BENCH_S3_BUCKET='${BENCH_S3_BUCKET}'
S3_BASE_PREFIX='${S3_BASE_PREFIX}'
SRC_S3_URI='${src_s3_uri}'

runuser -u ec2-user -- env \\
RUN_ID="\$RUN_ID" \\
DATASET="\$DATASET" \\
LIMIT_ROWS="\$LIMIT_ROWS" \\
SCAN_REPEATS="\$SCAN_REPEATS" \\
TAKE_ITERS="\$TAKE_ITERS" \\
BLOB_ITERS="\$BLOB_ITERS" \\
LANCE_FILE_VERSION="\$LANCE_FILE_VERSION" \\
BENCH_S3_BUCKET="\$BENCH_S3_BUCKET" \\
S3_BASE_PREFIX="\$S3_BASE_PREFIX" \\
SRC_S3_URI="\$SRC_S3_URI" \\
bash <<'INNER_SCRIPT'
set -euo pipefail
export PATH="\$HOME/.cargo/bin:\$HOME/.local/bin:\$PATH"
WORK_ROOT="\$HOME/lance-bench"
RUN_DIR="\$WORK_ROOT/\$RUN_ID"
mkdir -p "\$RUN_DIR"
cd "\$RUN_DIR"

aws s3 cp "\$SRC_S3_URI" ./source.tgz --region '${AWS_REGION}'
tar -xzf source.tgz

if [[ ! -x ./target/release/bench ]]; then
  if ! cargo build -p bench-cli --release > build.log 2>&1; then
    tail -n 200 build.log >&2
    exit 1
  fi
fi

case "\$DATASET" in
  fine-web)
    if ! find data/fineweb -name '*.parquet' -print -quit | grep -q .; then
      ./target/release/bench download --dataset fine-web --out data/fineweb
    fi
    ;;
  open-vid)
    if [[ ! -f data/openvid/openvid.parquet ]]; then
      ./target/release/bench download --dataset open-vid --out data/openvid
    fi
    ;;
  laion10m)
    if ! find data/laion10m -name '*.tar' -print -quit | grep -q .; then
      ./target/release/bench download --dataset laion10m --out data/laion10m
    fi
    ;;
  le-robot-push-t)
    if [[ ! -d data/lerobot-pusht/data ]]; then
      ./target/release/bench download --dataset le-robot-push-t --out data/lerobot-pusht
    fi
    ;;
  le-robot-push-t-image)
    if [[ ! -d data/lerobot-pusht_image/data ]]; then
      ./target/release/bench download --dataset le-robot-push-t-image --out data/lerobot-pusht_image
    fi
    ;;
  *)
    echo "unsupported dataset: \$DATASET" >&2
    exit 1
    ;;
esac

AWS_REGION='${AWS_REGION}' BENCH_S3_BUCKET='${BENCH_S3_BUCKET}' S3_BASE_PREFIX='${S3_BASE_PREFIX}' MODE=setup SKIP_BUILD=true ./scripts/run_lance_local_vs_s3.sh
AWS_REGION='${AWS_REGION}' DATASET="\$DATASET" LIMIT_ROWS="\$LIMIT_ROWS" SCAN_REPEATS="\$SCAN_REPEATS" TAKE_ITERS="\$TAKE_ITERS" BLOB_ITERS="\$BLOB_ITERS" LANCE_FILE_VERSION="\$LANCE_FILE_VERSION" RUN_ID="\$RUN_ID" MODE=run SKIP_BUILD=true ./scripts/run_lance_local_vs_s3.sh

aws s3 cp --recursive "results/aws-local-vs-s3/runs/\$RUN_ID" "s3://${BENCH_S3_BUCKET}/${S3_BASE_PREFIX}/results/\$RUN_ID" --region '${AWS_REGION}'
INNER_SCRIPT
RUNSCRIPT

  echo "$script_path"
}

setup_remote() {
  require_cmd aws
  require_cmd tar

  resolve_aws_args
  ACCOUNT_ID="$(aws_cli sts get-caller-identity --query Account --output text)"

  ensure_bucket
  ensure_iam_role_and_profile
  resolve_network_defaults
  resolve_ami
  ensure_instance
  wait_ssm_online

  local bootstrap_script
  bootstrap_script="$(create_bootstrap_script)"
  run_ssm_script "$bootstrap_script" "bootstrap"
  rm -f "$bootstrap_script"

  write_setup_files

  echo "[setup] completed"
  echo "[setup] env file: ${SETUP_ENV_FILE}"
  echo "[setup] note file: ${SETUP_NOTE_FILE}"
}

run_remote() {
  load_setup

  local src_tgz
  src_tgz="$(mktemp /tmp/lance-src-XXXXXX.tgz)"
  tar \
    --exclude='.git' \
    --exclude='target' \
    --exclude='data' \
    --exclude='results' \
    --exclude='results.bak-*' \
    --exclude='scripts/__pycache__' \
    -czf "$src_tgz" .

  local src_s3_uri="s3://${BENCH_S3_BUCKET}/${S3_BASE_PREFIX}/control/source-${RUN_ID}.tgz"
  aws_cli s3 cp "$src_tgz" "$src_s3_uri" >/dev/null
  rm -f "$src_tgz"

  local remote_run_script
  remote_run_script="$(create_remote_run_script "$src_s3_uri")"
  run_ssm_script "$remote_run_script" "run-${RUN_ID}"
  rm -f "$remote_run_script"

  echo "[run] completed"
  echo "[run] instance_id: ${INSTANCE_ID}"
  echo "[run] result_s3_prefix: s3://${BENCH_S3_BUCKET}/${S3_BASE_PREFIX}/results/${RUN_ID}/"
  echo "[run] local logs: results/aws-local-vs-s3/ec2-runs/${RUN_ID}/"
}

status_remote() {
  load_setup
  local state
  state="$(instance_state "$INSTANCE_ID")"
  local ping
  ping="$(aws_cli ssm describe-instance-information --filters "Key=InstanceIds,Values=${INSTANCE_ID}" --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null || true)"

  echo "instance_id=${INSTANCE_ID}"
  echo "instance_state=${state}"
  echo "ssm_ping_status=${ping}"
  echo "bucket=${BENCH_S3_BUCKET}"
  echo "prefix=${S3_BASE_PREFIX}"
}

case "$MODE" in
  setup)
    setup_remote
    ;;
  run)
    run_remote
    ;;
  setup-and-run)
    setup_remote
    run_remote
    ;;
  status)
    status_remote
    ;;
  *)
    echo "unsupported MODE: ${MODE} (expected: setup | run | setup-and-run | status)" >&2
    exit 1
    ;;
esac
