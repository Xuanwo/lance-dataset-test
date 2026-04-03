# lance-dataset-test

`lance-dataset-test` is a Rust benchmark harness for comparing columnar storage engines on multi-modal datasets.

The repository currently focuses on:

- `lance`
- `lance-fragment` for fragment-level scan comparisons
- `parquet`

## What It Covers

### Datasets

- `fine-web`
- `open-vid`
- `laion10m`
- `le-robot-push-t`
- `le-robot-push-t-image`

### Workloads

- ingest
- full scan
- projection scan
- filtered scan
- random row access
- random blob access
- schema evolution / backfill
- dataset size measurement

## Requirements

- Rust toolchain
- `hf` CLI for the dataset download scripts or `bench download`
- enough local disk space for raw inputs and generated datasets

Optional:

- AWS CLI for the S3 / EC2 helper scripts
- Xcode Instruments / `xctrace` on macOS for Time Profiler workflows

## Quick Start

### 1. Download input datasets

Use the helper scripts from the repository root:

```bash
./scripts/download_fineweb.sh
./scripts/download_openvid.sh
./scripts/download_laion10m.sh
./scripts/download_lerobot.sh
```

Or download everything:

```bash
./scripts/download_all.sh
```

Notes:

- `download_openvid.sh` converts the source CSV into `data/openvid/openvid.parquet`.
- The benchmark expects local input data under `data/`.

### 2. Build the CLI

```bash
cargo build -p bench-cli --release
```

The benchmark binary will be available at:

```bash
./target/release/bench
```

### 3. Ingest datasets

Ingest FineWeb into Lance:

```bash
./target/release/bench ingest \
  --engine lance \
  --dataset fine-web \
  --input data/fineweb \
  --out results/fineweb.lance \
  --limit-rows 10000000
```

Generate multiple Lance storage versions from the same input:

```bash
./target/release/bench ingest-lance-versions \
  --dataset fine-web \
  --input data/fineweb \
  --out-root results \
  --lance-file-versions 2.0,2.1,2.2 \
  --limit-rows 10000000
```

### 4. Run workloads

Full scan:

```bash
./target/release/bench scan \
  --engine lance \
  --dataset fine-web \
  --path results/fineweb.lance \
  --mode full
```

Filtered scan:

```bash
./target/release/bench scan \
  --engine parquet \
  --dataset fine-web \
  --path results/fineweb.parquet \
  --mode filter-low
```

Random take:

```bash
./target/release/bench take \
  --engine lance \
  --dataset fine-web \
  --path results/fineweb.lance \
  --iters 1000
```

Random blob:

```bash
./target/release/bench blob \
  --engine lance \
  --dataset laion10m \
  --path results/laion10m.lance \
  --column image \
  --iters 1000
```

Evolution / backfill:

```bash
./target/release/bench evolve \
  --engine lance \
  --dataset fine-web \
  --path results/fineweb.lance \
  --new-column bench_derived_u64
```

Dataset size:

```bash
./target/release/bench size \
  --engine lance \
  --dataset fine-web \
  --path results/fineweb.lance
```

## Full Benchmark Suite

Run the built-in suite:

```bash
./target/release/bench suite
```

Or use the helper script:

```bash
./scripts/run_suite.sh
```

The suite runs:

- ingest
- size
- full / project / filtered scans
- random take
- random blob on blob-capable datasets
- evolve

By default it compares:

- `parquet`
- `lance` file versions `2.0,2.1,2.2`

Outputs are written under `results/`, including:

- per-run JSON artifacts
- generated `REPORT.md`
- generated plots

## Useful Commands

Prepare OpenVid manually from a CSV input:

```bash
./target/release/bench prepare-openvid \
  --input path/to/OpenVid-1M.csv \
  --out data/openvid/openvid.parquet
```

Run one workload through the generic dispatcher:

```bash
./target/release/bench run \
  --engine lance \
  --dataset fine-web \
  --workload scan-full \
  --path results/fineweb.lance \
  --out results/scan-full.lance.json
```

Generate a report from existing JSON results:

```bash
./target/release/bench report --out REPORT.md
```

Generate plots from existing JSON results:

```bash
./target/release/bench plot
```

Inspect a dataset schema:

```bash
./target/release/bench schema \
  --engine lance \
  --path results/fineweb.lance
```

## Notes

- `lance-fragment` supports scan workloads only.
- `open-vid` can materialize a synthetic `video_blob` column during ingest; the size is controlled by `OPENVID_FAKE_BLOB_BYTES`.
- Generated benchmark data, local datasets, and result artifacts are intentionally ignored by Git.
- Some helper scripts under `scripts/` are operational utilities for larger benchmark runs; they are optional and not required for local benchmarking.

## License

This project is licensed under Apache-2.0. See `LICENSE`.
