#!/usr/bin/env bash
set -euo pipefail

MOUNT_POINT="${1:-/mnt/benchmark}"
RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.97.0}"

sudo dnf install -y \
  clang \
  cmake \
  gcc \
  gcc-c++ \
  git \
  jq \
  make \
  nvme-cli \
  openssl-devel \
  perl \
  pkgconf-pkg-config \
  protobuf-compiler \
  protobuf-devel \
  python3 \
  python3-pip \
  xfsprogs

if [[ ! -x "${HOME}/.cargo/bin/rustup" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi

source "${HOME}/.cargo/env"
rustup toolchain install "${RUST_TOOLCHAIN}" --profile minimal
rustup default "${RUST_TOOLCHAIN}"

mapfile -t instance_store_devices < <(
  while IFS= read -r link; do
    readlink -f "${link}"
  done < <(
    compgen -G '/dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_*' || true
  ) | sort -u
)
if [[ ${#instance_store_devices[@]} -ne 1 ]]; then
  printf 'expected exactly one EC2 NVMe instance-store device, found %s\n' \
    "${#instance_store_devices[@]}" >&2
  exit 1
fi

device="${instance_store_devices[0]}"
if [[ -z "${device}" || ! -b "${device}" ]]; then
  printf 'invalid instance-store device: %s\n' "${device}" >&2
  exit 1
fi

if ! sudo blkid "${device}" >/dev/null 2>&1; then
  sudo mkfs.xfs -f "${device}"
fi

sudo mkdir -p "${MOUNT_POINT}"
if ! findmnt -rn "${MOUNT_POINT}" >/dev/null 2>&1; then
  sudo mount -o noatime "${device}" "${MOUNT_POINT}"
fi
sudo chown "$(id -u):$(id -g)" "${MOUNT_POINT}"

printf 'benchmark_mount=%s\n' "${MOUNT_POINT}"
printf 'benchmark_device=%s\n' "${device}"
findmnt "${MOUNT_POINT}"
rustc --version
cargo --version
