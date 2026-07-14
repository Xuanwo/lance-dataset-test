#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(r"\.r(\d+)\.json$")


def median(values):
    return float(statistics.median(values))


def load_records(run_root: Path):
    records = []
    required_params = [
        "implementation",
        "api",
        "selector",
        "ordering",
        "distribution",
        "batch_size",
        "request_concurrency",
        "open_mode",
        "requests",
        "trace_fingerprint",
        "selected_blobs",
        "materialized_blobs",
        "content_consumption",
    ]
    missing = []
    for path in sorted((run_root / "json").glob("batch.*.json")):
        payload = json.loads(path.read_text())
        params = payload["meta"]["params"]
        for name in required_params:
            if name not in params:
                missing.append(f"{path.name}:{name}")
        match = RUN_RE.search(path.name)
        if not match:
            raise ValueError(f"cannot parse repetition from {path.name}")
        selected_blobs = int(params["selected_blobs"])
        requests = int(params["requests"])
        batch_size = int(params["batch_size"])
        if selected_blobs != requests * batch_size:
            raise ValueError(
                f"selected blob count mismatch in {path.name}: "
                f"{selected_blobs} != {requests} * {batch_size}"
            )
        if params["content_consumption"] != "full-payload-length":
            raise ValueError(f"unexpected content consumption in {path.name}")
        wall_us = int(payload["timing"]["wall_time_us"])
        materialized_blobs = int(params["materialized_blobs"])
        if materialized_blobs == 0:
            raise ValueError(f"no non-null blob payloads materialized in {path.name}")
        records.append(
            {
                "file": path.name,
                "dataset": payload["meta"]["dataset"],
                "implementation": params["implementation"],
                "api": params["api"],
                "selector": params["selector"],
                "ordering": params["ordering"],
                "distribution": params["distribution"],
                "batch_size": batch_size,
                "request_concurrency": int(params["request_concurrency"]),
                "open_mode": params["open_mode"],
                "repetition": int(match.group(1)),
                "seed": int(payload["meta"]["seed"]),
                "requests": requests,
                "selected_blobs": selected_blobs,
                "materialized_blobs": materialized_blobs,
                "raw_us_per_blob_uses_selected_rows": "us_per_selected_row"
                not in params,
                "trace_fingerprint": params["trace_fingerprint"],
                "wall_time_us": wall_us,
                "us_per_selected_row": wall_us / selected_blobs,
                "us_per_blob": wall_us / materialized_blobs,
                "batch_p50_us": int(payload["latency"]["p50_us"]),
                "batch_p95_us": int(payload["latency"]["p95_us"]),
                "batch_p99_us": int(payload["latency"]["p99_us"]),
                "bytes": int(payload["bytes"]),
            }
        )
    if missing:
        raise ValueError("missing required result fields: " + ", ".join(missing[:20]))
    if not records:
        raise ValueError(f"no batch results found under {run_root / 'json'}")
    return records


GROUP_FIELDS = [
    "dataset",
    "implementation",
    "api",
    "selector",
    "ordering",
    "distribution",
    "batch_size",
    "request_concurrency",
    "open_mode",
]


def validate_records(records, expected_repetitions):
    run_keys = set()
    duplicates = []
    for record in records:
        key = tuple(record[field] for field in GROUP_FIELDS) + (record["repetition"],)
        if key in run_keys:
            duplicates.append(key)
        run_keys.add(key)
    if duplicates:
        raise ValueError(f"duplicate result keys: {duplicates[:3]}")

    trace_groups = defaultdict(set)
    materialized_groups = defaultdict(set)
    byte_groups = defaultdict(set)
    for record in records:
        workload_key = (
            record["dataset"],
            record["distribution"],
            record["batch_size"],
            record["requests"],
            record["open_mode"],
            record["repetition"],
        )
        trace_groups[workload_key].add(record["trace_fingerprint"])
        materialized_groups[workload_key].add(record["materialized_blobs"])
        byte_groups[workload_key].add(record["bytes"])
    bad_traces = {key: value for key, value in trace_groups.items() if len(value) != 1}
    bad_materialized = {
        key: value for key, value in materialized_groups.items() if len(value) != 1
    }
    bad_bytes = {key: value for key, value in byte_groups.items() if len(value) != 1}
    if bad_traces:
        raise ValueError(f"row trace mismatch across engines/APIs: {list(bad_traces.items())[:3]}")
    if bad_materialized:
        raise ValueError(
            "materialized blob count mismatch across engines/APIs: "
            f"{list(bad_materialized.items())[:3]}"
        )
    if bad_bytes:
        raise ValueError(f"materialized byte mismatch across engines/APIs: {list(bad_bytes.items())[:3]}")

    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in GROUP_FIELDS)].append(record)
    bad_repetitions = {
        key: len(value)
        for key, value in grouped.items()
        if len(value) != expected_repetitions
    }
    if bad_repetitions:
        raise ValueError(
            f"incomplete repetitions: expected={expected_repetitions} "
            f"examples={list(bad_repetitions.items())[:5]}"
        )
    return grouped, len(trace_groups), len(materialized_groups), len(byte_groups)


def aggregate(grouped):
    summary = []
    for key, values in sorted(grouped.items()):
        row = dict(zip(GROUP_FIELDS, key))
        row.update(
            {
                "repetitions": len(values),
                "requests_per_run": values[0]["requests"],
                "selected_blobs_per_run": values[0]["selected_blobs"],
                "median_materialized_blobs_per_run": median(
                    [value["materialized_blobs"] for value in values]
                ),
                "min_materialized_blobs_per_run": min(
                    value["materialized_blobs"] for value in values
                ),
                "max_materialized_blobs_per_run": max(
                    value["materialized_blobs"] for value in values
                ),
                "median_bytes_per_run": median([value["bytes"] for value in values]),
                "min_bytes_per_run": min(value["bytes"] for value in values),
                "max_bytes_per_run": max(value["bytes"] for value in values),
                "median_wall_time_us": median([value["wall_time_us"] for value in values]),
                "median_us_per_selected_row": median(
                    [value["us_per_selected_row"] for value in values]
                ),
                "min_us_per_selected_row": min(
                    value["us_per_selected_row"] for value in values
                ),
                "max_us_per_selected_row": max(
                    value["us_per_selected_row"] for value in values
                ),
                "median_us_per_blob": median([value["us_per_blob"] for value in values]),
                "min_us_per_blob": min(value["us_per_blob"] for value in values),
                "max_us_per_blob": max(value["us_per_blob"] for value in values),
                "median_batch_p50_us": median([value["batch_p50_us"] for value in values]),
                "median_batch_p95_us": median([value["batch_p95_us"] for value in values]),
                "median_batch_p99_us": median([value["batch_p99_us"] for value in values]),
            }
        )
        summary.append(row)
    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary_index(summary):
    return {
        tuple(row[field] for field in GROUP_FIELDS): row
        for row in summary
    }


def lookup(index, dataset, implementation, api, selector, ordering, distribution, batch, concurrency, open_mode="opened"):
    return index[
        (
            dataset,
            implementation,
            api,
            selector,
            ordering,
            distribution,
            batch,
            concurrency,
            open_mode,
        )
    ]


def format_number(value, digits=1):
    return f"{value:,.{digits}f}"


def load_sizes(run_root: Path):
    rows = []
    for path in sorted((run_root / "json").glob("size.*.json")):
        payload = json.loads(path.read_text())
        parts = path.stem.split(".")
        dataset = "open_vid" if parts[1] == "openvid" else parts[1]
        implementation = {
            "lance": "lance-v2.2",
            "parquet-default": "parquet-default",
            "parquet-default-encoding-layout": "parquet-default-encoding-layout",
        }[parts[2]]
        rows.append(
            {
                "dataset": dataset,
                "implementation": implementation,
                "bytes": int(payload["bytes"]),
            }
        )
    return rows


def validate_verification(run_root: Path):
    files = sorted((run_root / "json").glob("verify.*.json"))
    if not files:
        raise ValueError("no full-content verification files found")
    for path in files:
        payload = json.loads(path.read_text())
        if not payload.get("matched") or not payload.get("full_content_compared"):
            raise ValueError(f"blob verification failed: {path}")
    return len(files)


def plot_curves(run_root: Path, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = run_root / "plots"
    plot_dir.mkdir(exist_ok=True)
    colors = {
        "Lance singleton take": "#4E79A7",
        "Lance batched take": "#F28E2B",
        "Lance planned read_blobs": "#59A14F",
        "Parquet default": "#B07AA1",
        "Parquet default-encoding layout": "#E15759",
    }
    markers = ["o", "s", "^", "D", "P"]
    outputs = []

    for dataset in ["open_vid", "laion10m"]:
        for distribution in ["uniform", "ann-top-k"]:
            series = [
                ("Lance singleton take", "lance-v2.2", "singleton-take"),
                ("Lance batched take", "lance-v2.2", "batched-take"),
                ("Lance planned read_blobs", "lance-v2.2", "planned-read-blobs"),
                ("Parquet default", "parquet-default", "parquet-row-selection"),
                (
                    "Parquet default-encoding layout",
                    "parquet-default-encoding-layout",
                    "parquet-row-selection",
                ),
            ]
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            for marker, (label, implementation, api) in zip(markers, series):
                rows = sorted(
                    [
                        row
                        for row in summary
                        if row["dataset"] == dataset
                        and row["distribution"] == distribution
                        and row["implementation"] == implementation
                        and row["api"] == api
                        and row["selector"] == "indices"
                        and row["ordering"] == "preserve"
                        and row["request_concurrency"] == 1
                        and row["open_mode"] == "opened"
                    ],
                    key=lambda row: row["batch_size"],
                )
                x = [row["batch_size"] for row in rows]
                y = [row["median_us_per_blob"] for row in rows]
                low = [row["min_us_per_blob"] for row in rows]
                high = [row["max_us_per_blob"] for row in rows]
                ax.plot(x, y, marker=marker, label=label, color=colors[label], linewidth=2)
                ax.fill_between(x, low, high, color=colors[label], alpha=0.10)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            ax.set_xlabel("Business batch size (row selectors)")
            ax.set_ylabel("Median wall time per materialized blob (µs/blob, log scale)")
            ax.set_title(f"{dataset}: batch amortization, C=1, {distribution}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            stem = plot_dir / f"us_per_blob_batch_size.{dataset}.{distribution}"
            png = Path(f"{stem}.png")
            svg = Path(f"{stem}.svg")
            fig.savefig(png, dpi=180)
            fig.savefig(svg)
            plt.close(fig)
            outputs.extend([png, svg])

            concurrency_series = [
                ("Lance planned read_blobs", "lance-v2.2", "planned-read-blobs"),
                ("Parquet default", "parquet-default", "parquet-row-selection"),
                (
                    "Parquet default-encoding layout",
                    "parquet-default-encoding-layout",
                    "parquet-row-selection",
                ),
            ]
            fig, ax = plt.subplots(figsize=(8.2, 5.0))
            for marker, (label, implementation, api) in zip(markers, concurrency_series):
                rows = sorted(
                    [
                        row
                        for row in summary
                        if row["dataset"] == dataset
                        and row["distribution"] == distribution
                        and row["implementation"] == implementation
                        and row["api"] == api
                        and row["selector"] == "indices"
                        and row["ordering"] == "preserve"
                        and row["batch_size"] == 16
                        and row["open_mode"] == "opened"
                    ],
                    key=lambda row: row["request_concurrency"],
                )
                ax.plot(
                    [row["request_concurrency"] for row in rows],
                    [row["median_batch_p50_us"] / 1000 for row in rows],
                    marker=marker,
                    label=label,
                    color=colors[label],
                    linewidth=2,
                )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([1, 8, 32], ["1", "8", "32"])
            ax.set_xlabel("Concurrent business requests")
            ax.set_ylabel("Median batch latency (ms, log scale)")
            ax.set_title(f"{dataset}: request concurrency, B=16, {distribution}")
            ax.grid(True, axis="y", alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            stem = plot_dir / f"batch_latency_concurrency.{dataset}.{distribution}"
            png = Path(f"{stem}.png")
            svg = Path(f"{stem}.svg")
            fig.savefig(png, dpi=180)
            fig.savefig(svg)
            plt.close(fig)
            outputs.extend([png, svg])
    return outputs


def write_report(
    run_root: Path,
    summary,
    trace_group_count,
    materialized_group_count,
    byte_group_count,
    verification_count,
    legacy_metric_count,
    sizes,
    plot_files,
):
    index = summary_index(summary)
    lines = [
        "# Lance v2.2 vs Parquet batched blob benchmark",
        "",
        "## Contract and validation",
        "",
        "- Headline requests reuse an opened `Arc<Dataset>` / Parquet file; `reopen` appears only in the cold-start diagnostic.",
        "- Lance `planned-read-blobs` performs one `read_blobs(...).execute()` per business batch.",
        "- Parquet performs one reader builder plus one whole-batch `RowSelection` per business batch.",
        "- All implementations materialize full payloads and consume payload lengths inside the timed request.",
        "- `B` is the number of row selectors in a business request. `µs/blob` divides by non-null payloads actually materialized; `µs/selected row` is retained in the CSV files.",
        "- Parquet default and layout files use Arrow Parquet writer default encodings. The layout file only changes write batching, row-group bytes, and offset-index/page layout.",
        f"- Verified {trace_group_count} workload/repetition groups had identical trace fingerprints across engines and APIs.",
        f"- Verified {materialized_group_count} workload/repetition groups returned identical non-null blob counts across engines and APIs.",
        f"- Verified {byte_group_count} workload/repetition groups materialized identical byte totals across engines and APIs.",
        f"- {verification_count} independent sampled full-content comparisons passed.",
        f"- {legacy_metric_count} raw run JSON files use the legacy selected-row denominator in `params.us_per_blob`; this report and both CSV files recompute `µs/blob` from raw wall time and non-null materialized count.",
        "",
        "The complete, unrounded result set is in `per_run.csv`; repetition medians and min/max ranges are in `summary.csv`.",
        "",
        "## File size",
        "",
        "| Dataset | Implementation | Size (GiB) |",
        "|---|---|---:|",
    ]
    for row in sizes:
        lines.append(
            f"| {row['dataset']} | {row['implementation']} | {row['bytes'] / 2**30:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Payload coverage at B=16, C=1",
            "",
            "LAION WebDataset rows can have a null `image`. Those rows still participate in the same selector trace and batch latency, but are excluded from the `µs/blob` denominator because no payload exists.",
            "",
            "| Dataset | Distribution | Selected rows/run | Median materialized blobs/run | Coverage |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset in ["open_vid", "laion10m"]:
        for distribution in ["uniform", "ann-top-k"]:
            row = lookup(
                index,
                dataset,
                "lance-v2.2",
                "planned-read-blobs",
                "indices",
                "preserve",
                distribution,
                16,
                1,
            )
            selected = row["selected_blobs_per_run"]
            materialized = row["median_materialized_blobs_per_run"]
            lines.append(
                f"| {dataset} | {distribution} | {selected} | {materialized:.0f} | {materialized / selected:.1%} |"
            )

    lines.extend(
        [
            "",
            "## ANN trace provenance",
            "",
            "- OpenVid uses 128 precomputed exact L2 top-64 result lists from the SIFT1M ground truth over one million base vectors.",
            "- LAION uses 128 exact L2 top-64 result lists recomputed with `faiss.IndexFlatL2` over the first 200,000 SIFT1M base vectors.",
            "- These are data-dependent nearest-neighbor result traces, preserving ranked top-k access locality. Their row IDs are mapped onto the blob datasets; they are not searches over OpenVid or LAION embeddings.",
        ]
    )

    for dataset in ["open_vid", "laion10m"]:
        for distribution in ["uniform", "ann-top-k"]:
            lines.extend(
                [
                    "",
                    f"## Headline: {dataset}, {distribution}, preserve order",
                    "",
                    "| B | C | Lance batch p50 (ms) | Lance µs/blob | Parquet default µs/blob | Parquet layout µs/blob | Default/Lance | Layout/Lance |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for batch in [1, 4, 16, 64]:
                for concurrency in [1, 8, 32]:
                    lance = lookup(
                        index,
                        dataset,
                        "lance-v2.2",
                        "planned-read-blobs",
                        "indices",
                        "preserve",
                        distribution,
                        batch,
                        concurrency,
                    )
                    parquet = lookup(
                        index,
                        dataset,
                        "parquet-default",
                        "parquet-row-selection",
                        "indices",
                        "preserve",
                        distribution,
                        batch,
                        concurrency,
                    )
                    layout = lookup(
                        index,
                        dataset,
                        "parquet-default-encoding-layout",
                        "parquet-row-selection",
                        "indices",
                        "preserve",
                        distribution,
                        batch,
                        concurrency,
                    )
                    lines.append(
                        "| {} | {} | {} | {} | {} | {} | {}x | {}x |".format(
                            batch,
                            concurrency,
                            format_number(lance["median_batch_p50_us"] / 1000, 2),
                            format_number(lance["median_us_per_blob"]),
                            format_number(parquet["median_us_per_blob"]),
                            format_number(layout["median_us_per_blob"]),
                            format_number(
                                parquet["median_us_per_blob"] / lance["median_us_per_blob"], 2
                            ),
                            format_number(
                                layout["median_us_per_blob"] / lance["median_us_per_blob"], 2
                            ),
                        )
                    )

            lines.extend(
                [
                    "",
                    f"### API amortization at C=1: {dataset}, {distribution}",
                    "",
                    "| B | Singleton µs/blob | Batched take µs/blob | Planned µs/blob | Planned / singleton | Parquet default µs/blob |",
                    "|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for batch in [1, 4, 16, 64]:
                values = {}
                for api in ["singleton-take", "batched-take", "planned-read-blobs"]:
                    values[api] = lookup(
                        index,
                        dataset,
                        "lance-v2.2",
                        api,
                        "indices",
                        "preserve",
                        distribution,
                        batch,
                        1,
                    )["median_us_per_blob"]
                parquet = lookup(
                    index,
                    dataset,
                    "parquet-default",
                    "parquet-row-selection",
                    "indices",
                    "preserve",
                    distribution,
                    batch,
                    1,
                )["median_us_per_blob"]
                lines.append(
                    "| {} | {} | {} | {} | {}x | {} |".format(
                        batch,
                        format_number(values["singleton-take"]),
                        format_number(values["batched-take"]),
                        format_number(values["planned-read-blobs"]),
                        format_number(
                            values["planned-read-blobs"] / values["singleton-take"], 2
                        ),
                        format_number(parquet),
                    )
                )

            lines.extend(
                [
                    "",
                    f"### Selector and ordering at B=16, C=8: {dataset}, {distribution}",
                    "",
                    "| Selector | Ordering | Lance batch p50 (ms) | Lance µs/blob |",
                    "|---|---|---:|---:|",
                ]
            )
            for selector in ["indices", "addresses"]:
                for ordering in ["preserve", "unordered"]:
                    row = lookup(
                        index,
                        dataset,
                        "lance-v2.2",
                        "planned-read-blobs",
                        selector,
                        ordering,
                        distribution,
                        16,
                        8,
                    )
                    lines.append(
                        f"| {selector} | {ordering} | {row['median_batch_p50_us'] / 1000:.2f} | {row['median_us_per_blob']:.1f} |"
                    )

        lines.extend(
            [
                "",
                f"## Cold-start diagnostic: {dataset}, B=16, C=1",
                "",
                "| Implementation | Median batch p50 (ms) | Median µs/blob |",
                "|---|---:|---:|",
            ]
        )
        for implementation, api in [
            ("lance-v2.2", "planned-read-blobs"),
            ("parquet-default", "parquet-row-selection"),
            ("parquet-default-encoding-layout", "parquet-row-selection"),
        ]:
            row = lookup(
                index,
                dataset,
                implementation,
                api,
                "indices",
                "preserve",
                "uniform",
                16,
                1,
                "reopen",
            )
            lines.append(
                f"| {implementation} | {row['median_batch_p50_us'] / 1000:.2f} | {row['median_us_per_blob']:.1f} |"
            )

    lines.extend(["", "## Charts", ""])
    for path in plot_files:
        if path.suffix == ".png":
            lines.append(f"- `{path.relative_to(run_root)}`")
    (run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()

    environment = {}
    for line in (run_root / "ENVIRONMENT.txt").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            environment.setdefault(key, value)
    expected_repetitions = int(environment["repetitions"])

    records = load_records(run_root)
    grouped, trace_group_count, materialized_group_count, byte_group_count = validate_records(
        records, expected_repetitions
    )
    summary = aggregate(grouped)
    sizes = load_sizes(run_root)
    verification_count = validate_verification(run_root)
    legacy_metric_count = sum(
        record["raw_us_per_blob_uses_selected_rows"] for record in records
    )

    write_csv(run_root / "per_run.csv", records)
    write_csv(run_root / "summary.csv", summary)
    quality = {
        "run_records": len(records),
        "aggregate_groups": len(summary),
        "expected_repetitions": expected_repetitions,
        "trace_groups_validated": trace_group_count,
        "materialized_blob_groups_validated": materialized_group_count,
        "byte_groups_validated": byte_group_count,
        "verification_files": verification_count,
        "legacy_raw_us_per_blob_records": legacy_metric_count,
        "missing_required_fields": 0,
        "duplicate_run_keys": 0,
        "batch_sizes": sorted({record["batch_size"] for record in records}),
        "request_concurrency": sorted(
            {record["request_concurrency"] for record in records}
        ),
        "distributions": sorted({record["distribution"] for record in records}),
        "us_per_blob_range": [
            min(record["us_per_blob"] for record in records),
            max(record["us_per_blob"] for record in records),
        ],
        "us_per_selected_row_range": [
            min(record["us_per_selected_row"] for record in records),
            max(record["us_per_selected_row"] for record in records),
        ],
    }
    (run_root / "DATA_QUALITY.json").write_text(json.dumps(quality, indent=2) + "\n")
    plot_files = plot_curves(run_root, summary)
    write_report(
        run_root,
        summary,
        trace_group_count,
        materialized_group_count,
        byte_group_count,
        verification_count,
        legacy_metric_count,
        sizes,
        plot_files,
    )


if __name__ == "__main__":
    main()
