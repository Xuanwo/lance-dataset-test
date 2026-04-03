#!/usr/bin/env bash
set -euo pipefail

cargo build -p bench-cli --release

./target/release/bench suite "$@"
./target/release/bench report --out REPORT.md
./target/release/bench plot

