#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


LANCE = "lance-v2.2"
PARQUET_BASELINE = "parquet-default-writer-sequential-reader"
PARQUET_READER = "parquet-default-writer-row-selection-reader"
PARQUET_WRITER = "parquet-random-blob-writer-row-selection-reader"
EXPECTED_IMPLEMENTATIONS = {
    LANCE,
    PARQUET_BASELINE,
    PARQUET_READER,
    PARQUET_WRITER,
}

DISPLAY = {
    LANCE: "Lance v2.2",
    PARQUET_BASELINE: "Parquet default writer + sequential reader",
    PARQUET_READER: "Parquet default writer + RowSelection reader",
    PARQUET_WRITER: "Parquet random-blob writer + RowSelection reader",
    "parquet-default-writer": "Parquet default writer",
    "parquet-random-blob-writer": "Parquet random-blob writer",
}


def load_json(path: Path):
    with path.open() as file:
        return json.load(file)


def load_blob_records(json_dir: Path):
    records = []
    for path in sorted(json_dir.glob("blob.*.json")):
        record = load_json(path)
        if record.get("notes"):
            raise RuntimeError(f"benchmark record has notes: {path}: {record['notes']}")
        latency = record.get("latency")
        if latency is None:
            raise RuntimeError(f"benchmark record has no latency: {path}")
        match = re.search(r"\.r(\d+)\.json$", path.name)
        if not match:
            raise RuntimeError(f"cannot parse repetition from {path}")
        params = record["meta"]["params"]
        implementation = params.get("implementation")
        if implementation not in EXPECTED_IMPLEMENTATIONS:
            raise RuntimeError(f"unexpected implementation in {path}: {implementation}")
        records.append(
            {
                "path": str(path),
                "dataset": record["meta"]["dataset"],
                "open_mode": params["open_mode"],
                "implementation": implementation,
                "repetition": int(match.group(1)),
                "seed": int(record["meta"]["seed"]),
                "iters": int(params["iters"]),
                "row_count": int(params["row_count"]),
                "bytes": int(record["bytes"]),
                "wall_ms": float(record["timing"]["wall_time_us"]) / 1000.0,
                "p50_ms": float(latency["p50_us"]) / 1000.0,
                "p95_ms": float(latency["p95_us"]) / 1000.0,
                "p99_ms": float(latency["p99_us"]) / 1000.0,
                "offset_index_loaded": params.get("parquet_offset_index_loaded", ""),
            }
        )
    return records


def validate_groups(records):
    groups = defaultdict(dict)
    for record in records:
        key = (record["dataset"], record["open_mode"], record["repetition"])
        implementation = record["implementation"]
        if implementation in groups[key]:
            raise RuntimeError(f"duplicate implementation for {key}: {implementation}")
        groups[key][implementation] = record

    for key, implementations in groups.items():
        if set(implementations) != EXPECTED_IMPLEMENTATIONS:
            raise RuntimeError(
                f"incomplete implementation group for {key}: "
                f"{sorted(implementations)}"
            )
        lance = implementations[LANCE]
        for implementation, record in implementations.items():
            for field in ("seed", "iters", "row_count", "bytes"):
                if record[field] != lance[field]:
                    raise RuntimeError(
                        f"paired mismatch for {key}, {implementation}, {field}: "
                        f"lance={lance[field]} candidate={record[field]}"
                    )
    return groups


def geometric_mean(values):
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"geometric mean requires positive values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize_engines(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[
            (record["dataset"], record["open_mode"], record["implementation"])
        ].append(record)

    rows = []
    for (dataset, open_mode, implementation), group in sorted(grouped.items()):
        rows.append(
            {
                "dataset": dataset,
                "open_mode": open_mode,
                "implementation": implementation,
                "repetitions": len(group),
                "wall_median_ms": statistics.median(r["wall_ms"] for r in group),
                "p50_median_ms": statistics.median(r["p50_ms"] for r in group),
                "p95_median_ms": statistics.median(r["p95_ms"] for r in group),
                "p99_median_ms": statistics.median(r["p99_ms"] for r in group),
            }
        )
    return rows


def summarize_vs_lance(groups):
    paired = defaultdict(list)
    for (dataset, open_mode, _repetition), implementations in groups.items():
        lance = implementations[LANCE]
        for implementation in (PARQUET_BASELINE, PARQUET_READER, PARQUET_WRITER):
            parquet = implementations[implementation]
            paired[(dataset, open_mode, implementation)].append(
                {
                    metric: parquet[metric] / lance[metric]
                    for metric in ("wall_ms", "p50_ms", "p95_ms", "p99_ms")
                }
            )

    rows = []
    for (dataset, open_mode, implementation), ratios in sorted(paired.items()):
        rows.append(
            {
                "dataset": dataset,
                "open_mode": open_mode,
                "implementation": implementation,
                "wall_parquet_over_lance_median": statistics.median(
                    r["wall_ms"] for r in ratios
                ),
                "p50_parquet_over_lance_median": statistics.median(
                    r["p50_ms"] for r in ratios
                ),
                "p95_parquet_over_lance_median": statistics.median(
                    r["p95_ms"] for r in ratios
                ),
                "p99_parquet_over_lance_median": statistics.median(
                    r["p99_ms"] for r in ratios
                ),
            }
        )
    return rows


def summarize_improvements(groups):
    comparisons = {
        "reader_only": (PARQUET_BASELINE, PARQUET_READER),
        "writer_after_reader": (PARQUET_READER, PARQUET_WRITER),
        "end_to_end": (PARQUET_BASELINE, PARQUET_WRITER),
    }
    paired = defaultdict(list)
    for (dataset, open_mode, _repetition), implementations in groups.items():
        for comparison, (before_name, after_name) in comparisons.items():
            before = implementations[before_name]
            after = implementations[after_name]
            paired[(dataset, open_mode, comparison)].append(
                {
                    metric: before[metric] / after[metric]
                    for metric in ("wall_ms", "p50_ms", "p95_ms", "p99_ms")
                }
            )

    rows = []
    for (dataset, open_mode, comparison), ratios in sorted(paired.items()):
        rows.append(
            {
                "dataset": dataset,
                "open_mode": open_mode,
                "comparison": comparison,
                "wall_improvement_median": statistics.median(
                    r["wall_ms"] for r in ratios
                ),
                "p50_improvement_median": statistics.median(
                    r["p50_ms"] for r in ratios
                ),
                "p95_improvement_median": statistics.median(
                    r["p95_ms"] for r in ratios
                ),
                "p99_improvement_median": statistics.median(
                    r["p99_ms"] for r in ratios
                ),
            }
        )
    return rows


def summarize_overall(vs_lance_rows, improvement_rows):
    rows = []
    for open_mode in sorted({row["open_mode"] for row in vs_lance_rows}):
        for implementation in (PARQUET_BASELINE, PARQUET_READER, PARQUET_WRITER):
            selected = [
                row
                for row in vs_lance_rows
                if row["open_mode"] == open_mode
                and row["implementation"] == implementation
            ]
            rows.append(
                {
                    "kind": "parquet_over_lance",
                    "open_mode": open_mode,
                    "name": implementation,
                    "datasets": len(selected),
                    "wall_geomean": geometric_mean(
                        [row["wall_parquet_over_lance_median"] for row in selected]
                    ),
                    "p50_geomean": geometric_mean(
                        [row["p50_parquet_over_lance_median"] for row in selected]
                    ),
                    "p95_geomean": geometric_mean(
                        [row["p95_parquet_over_lance_median"] for row in selected]
                    ),
                    "p99_geomean": geometric_mean(
                        [row["p99_parquet_over_lance_median"] for row in selected]
                    ),
                }
            )
        for comparison in ("reader_only", "writer_after_reader", "end_to_end"):
            selected = [
                row
                for row in improvement_rows
                if row["open_mode"] == open_mode
                and row["comparison"] == comparison
            ]
            rows.append(
                {
                    "kind": "parquet_improvement",
                    "open_mode": open_mode,
                    "name": comparison,
                    "datasets": len(selected),
                    "wall_geomean": geometric_mean(
                        [row["wall_improvement_median"] for row in selected]
                    ),
                    "p50_geomean": geometric_mean(
                        [row["p50_improvement_median"] for row in selected]
                    ),
                    "p95_geomean": geometric_mean(
                        [row["p95_improvement_median"] for row in selected]
                    ),
                    "p99_geomean": geometric_mean(
                        [row["p99_improvement_median"] for row in selected]
                    ),
                }
            )
    return rows


def classify_artifact(path: Path):
    if ".parquet-default.json" in path.name:
        return "parquet-default-writer"
    if ".parquet-random-blob.json" in path.name:
        return "parquet-random-blob-writer"
    if ".lance.json" in path.name:
        return LANCE
    raise RuntimeError(f"cannot classify artifact: {path}")


def load_ingest_rows(json_dir: Path):
    sizes = {}
    for path in sorted(json_dir.glob("size.*.json")):
        record = load_json(path)
        sizes[(record["meta"]["dataset"], classify_artifact(path))] = int(
            record["bytes"]
        )

    rows = []
    for path in sorted(json_dir.glob("ingest.*.json")):
        record = load_json(path)
        params = record["meta"]["params"]
        dataset = record["meta"]["dataset"]
        implementation = classify_artifact(path)
        rows.append(
            {
                "dataset": dataset,
                "implementation": implementation,
                "rows": int(record["rows"]),
                "input_arrow_bytes": int(record["bytes"]),
                "ingest_wall_ms": float(record["timing"]["wall_time_us"]) / 1000.0,
                "artifact_bytes": sizes[(dataset, implementation)],
                "row_group_count": params.get("parquet_row_group_count", ""),
                "offset_index_loaded": params.get(
                    "parquet_offset_index_loaded", ""
                ),
                "data_page_size_limit": params.get(
                    "parquet_data_page_size_limit", ""
                ),
                "write_batch_size": params.get("parquet_write_batch_size", ""),
                "max_row_group_bytes": params.get(
                    "parquet_max_row_group_bytes", ""
                ),
                "blob_dictionary_enabled": params.get(
                    "parquet_blob_dictionary_enabled", ""
                ),
                "blob_statistics": params.get("parquet_blob_statistics", ""),
            }
        )
    return rows


def load_verification_rows(json_dir: Path):
    rows = []
    for path in sorted(json_dir.glob("verify.*.json")):
        record = load_json(path)
        if not record.get("matched") or not record.get("full_content_compared"):
            raise RuntimeError(f"blob verification did not pass: {path}")
        rows.append(
            {
                "dataset": record["dataset"],
                "variant": path.name.split(".")[2],
                "samples": int(record["samples"]),
                "total_bytes": int(record["total_bytes"]),
                "fnv1a64": record["fnv1a64"],
                "parquet_read_mode": record["parquet_read_mode"],
                "parquet_writer_profile": record["parquet_writer_profile"],
                "offset_index_loaded": record["parquet_offset_index_loaded"],
                "full_content_compared": record["full_content_compared"],
                "matched": record["matched"],
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:,.3f}"


def ratio(value):
    return f"{value:,.3f}x"


def gib(value):
    return f"{value / (1024 ** 3):,.3f}"


def write_markdown(
    path,
    records,
    engine_rows,
    vs_lance_rows,
    improvement_rows,
    overall_rows,
    ingest_rows,
    verification_rows,
):
    repetitions = len({record["repetition"] for record in records})
    iters = sorted({record["iters"] for record in records})
    seeds = sorted({record["seed"] for record in records})
    optimized = next(
        row
        for row in ingest_rows
        if row["implementation"] == "parquet-random-blob-writer"
    )
    lines = [
        "# Random Blob: Lance v2.2 and Native Parquet Variants",
        "",
        "## Method",
        "",
        f"- Repetitions: `{repetitions}`",
        f"- Iterations per repetition: `{iters}`",
        f"- Seeds: `{seeds}` (paired across all implementations)",
        "- Page cache is dropped before every implementation repetition.",
        "- `opened`: the dataset/file and required Parquet offset index are opened outside the timed loop.",
        "- `reopen`: the dataset/file and required Parquet offset index are reopened inside every timed iteration.",
        "- The timed operation materializes the selected blob bytes and observes their length; it does not decode pixels or serialize to an external sink.",
        "- Separate untimed validation compares the complete byte content returned by Lance and every Parquet path.",
        "- All Parquet variants store complete blob values inside Parquet. No external blob descriptors or sidecar files are used.",
        "- `Parquet / Lance` greater than 1 means Lance is faster. `Improvement` greater than 1 means the newer Parquet path is faster.",
        "",
        "## Random-Blob Writer Configuration",
        "",
        f"- Data page size limit: `{optimized['data_page_size_limit']}` bytes",
        f"- Write batch size: `{optimized['write_batch_size']}` row",
        f"- Maximum row-group bytes: `{optimized['max_row_group_bytes']}`",
        "- Blob column dictionary: disabled",
        "- Blob column page statistics: disabled",
        "- Blob column compression: uncompressed",
        "- Offset index: enabled and required by the RowSelection reader",
        "",
        "## Full-Content Validation",
        "",
        "| Dataset | Variant | Samples | Bytes compared | FNV-1a 64 | Writer profile | Read mode | Offset index loaded | Matched |",
        "|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in verification_rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['samples']} | "
            f"{row['total_bytes']:,} | `{row['fnv1a64']}` | {row['parquet_writer_profile']} | "
            f"{row['parquet_read_mode']} | "
            f"{row['offset_index_loaded']} | {row['matched']} |"
        )

    lines.extend(
        [
            "",
            "## Ingest and Artifact Layout",
            "",
            "| Dataset | Writer | Ingest wall (s) | Artifact (GiB) | Rows | Row groups | Offset index loaded |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in ingest_rows:
        lines.append(
            f"| {row['dataset']} | {DISPLAY.get(row['implementation'], row['implementation'])} | "
            f"{fmt(row['ingest_wall_ms'] / 1000.0)} | {gib(row['artifact_bytes'])} | "
            f"{row['rows']:,} | {row['row_group_count']} | {row['offset_index_loaded']} |"
        )

    lines.extend(
        [
            "",
            "## Median Results",
            "",
            "| Dataset | Open mode | Implementation | Wall for all iterations (ms) | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in engine_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {DISPLAY[row['implementation']]} | "
            f"{fmt(row['wall_median_ms'])} | {fmt(row['p50_median_ms'])} | "
            f"{fmt(row['p95_median_ms'])} | {fmt(row['p99_median_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Parquet Improvements",
            "",
            "| Dataset | Open mode | Comparison | Wall improvement | p50 | p95 | p99 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in improvement_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {row['comparison']} | "
            f"{ratio(row['wall_improvement_median'])} | "
            f"{ratio(row['p50_improvement_median'])} | "
            f"{ratio(row['p95_improvement_median'])} | "
            f"{ratio(row['p99_improvement_median'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired Parquet / Lance Ratios",
            "",
            "| Dataset | Open mode | Parquet implementation | Wall Parquet / Lance | p50 | p95 | p99 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in vs_lance_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {DISPLAY[row['implementation']]} | "
            f"{ratio(row['wall_parquet_over_lance_median'])} | "
            f"{ratio(row['p50_parquet_over_lance_median'])} | "
            f"{ratio(row['p95_parquet_over_lance_median'])} | "
            f"{ratio(row['p99_parquet_over_lance_median'])} |"
        )

    lines.extend(
        [
            "",
            "## Cross-Dataset Geomeans",
            "",
            "| Kind | Open mode | Name | Wall | p50 | p95 | p99 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"| {row['kind']} | {row['open_mode']} | {DISPLAY.get(row['name'], row['name'])} | "
            f"{ratio(row['wall_geomean'])} | {ratio(row['p50_geomean'])} | "
            f"{ratio(row['p95_geomean'])} | {ratio(row['p99_geomean'])} |"
        )

    lines.extend(
        [
            "",
            "## Complete Repetition Results",
            "",
            "| Dataset | Open mode | Rep | Seed | Implementation | Wall (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Bytes |",
            "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        records,
        key=lambda item: (
            item["dataset"],
            item["open_mode"],
            item["repetition"],
            item["implementation"],
        ),
    ):
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {row['repetition']} | "
            f"{row['seed']} | {DISPLAY[row['implementation']]} | {fmt(row['wall_ms'])} | "
            f"{fmt(row['p50_ms'])} | {fmt(row['p95_ms'])} | {fmt(row['p99_ms'])} | "
            f"{row['bytes']:,} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    json_dir = args.run_root / "json"

    records = load_blob_records(json_dir)
    if not records:
        raise RuntimeError("no random blob records found")
    groups = validate_groups(records)
    engine_rows = summarize_engines(records)
    vs_lance_rows = summarize_vs_lance(groups)
    improvement_rows = summarize_improvements(groups)
    overall_rows = summarize_overall(vs_lance_rows, improvement_rows)
    ingest_rows = load_ingest_rows(json_dir)
    verification_rows = load_verification_rows(json_dir)

    write_csv(args.run_root / "raw_repetitions.csv", records)
    write_csv(args.run_root / "summary_engine.csv", engine_rows)
    write_csv(args.run_root / "summary_vs_lance.csv", vs_lance_rows)
    write_csv(args.run_root / "summary_parquet_improvement.csv", improvement_rows)
    write_csv(args.run_root / "summary_overall.csv", overall_rows)
    write_csv(args.run_root / "ingest_and_size.csv", ingest_rows)
    write_csv(args.run_root / "full_content_validation.csv", verification_rows)
    (args.run_root / "summary.json").write_text(
        json.dumps(
            {
                "engine": engine_rows,
                "vs_lance": vs_lance_rows,
                "parquet_improvement": improvement_rows,
                "overall": overall_rows,
                "ingest": ingest_rows,
                "validation": verification_rows,
                "raw_repetitions": records,
            },
            indent=2,
        )
    )
    write_markdown(
        args.run_root / "SUMMARY.md",
        records,
        engine_rows,
        vs_lance_rows,
        improvement_rows,
        overall_rows,
        ingest_rows,
        verification_rows,
    )


if __name__ == "__main__":
    main()
