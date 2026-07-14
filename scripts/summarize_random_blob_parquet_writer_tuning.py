#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def load(path):
    with path.open() as file:
        return json.load(file)


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    json_dir = args.run_root / "json"

    sizes = {}
    for path in sorted(json_dir.glob("size.*.json")):
        parts = path.name.split(".")
        sizes[(parts[1], parts[2])] = int(load(path)["bytes"])

    ingest_rows = []
    for path in sorted(json_dir.glob("ingest.*.json")):
        parts = path.name.split(".")
        dataset, candidate = parts[1], parts[2]
        record = load(path)
        params = record["meta"]["params"]
        ingest_rows.append(
            {
                "dataset": dataset,
                "candidate": candidate,
                "rows": int(record["rows"]),
                "ingest_wall_ms": float(record["timing"]["wall_time_us"]) / 1000.0,
                "artifact_bytes": sizes[(dataset, candidate)],
                "row_group_count": int(params["parquet_row_group_count"]),
                "data_page_size_limit": int(params["parquet_data_page_size_limit"]),
                "write_batch_size": int(params["parquet_write_batch_size"]),
                "max_row_group_bytes": int(params["parquet_max_row_group_bytes"]),
            }
        )

    raw_rows = []
    for path in sorted(json_dir.glob("tune.*.json")):
        parts = path.name.split(".")
        match = re.search(r"\.r(\d+)\.json$", path.name)
        if not match:
            raise RuntimeError(f"cannot parse repetition: {path}")
        record = load(path)
        latency = record["latency"]
        raw_rows.append(
            {
                "dataset": parts[1],
                "open_mode": parts[2],
                "candidate": parts[3],
                "repetition": int(match.group(1)),
                "seed": int(record["meta"]["seed"]),
                "iters": int(record["meta"]["params"]["iters"]),
                "bytes": int(record["bytes"]),
                "wall_ms": float(record["timing"]["wall_time_us"]) / 1000.0,
                "p50_ms": float(latency["p50_us"]) / 1000.0,
                "p95_ms": float(latency["p95_us"]) / 1000.0,
                "p99_ms": float(latency["p99_us"]) / 1000.0,
            }
        )

    paired = defaultdict(dict)
    for row in raw_rows:
        key = (row["dataset"], row["open_mode"], row["repetition"])
        paired[key][row["candidate"]] = row
    candidates = {row["candidate"] for row in ingest_rows}
    for key, group in paired.items():
        if set(group) != candidates:
            raise RuntimeError(f"incomplete candidate group for {key}: {sorted(group)}")
        first = next(iter(group.values()))
        for row in group.values():
            for field in ("seed", "iters", "bytes"):
                if row[field] != first[field]:
                    raise RuntimeError(f"paired mismatch for {key}, {field}")

    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["dataset"], row["open_mode"], row["candidate"])].append(row)
    summary_rows = []
    for (dataset, open_mode, candidate), group in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "open_mode": open_mode,
                "candidate": candidate,
                "repetitions": len(group),
                "wall_median_ms": statistics.median(r["wall_ms"] for r in group),
                "p50_median_ms": statistics.median(r["p50_ms"] for r in group),
                "p95_median_ms": statistics.median(r["p95_ms"] for r in group),
                "p99_median_ms": statistics.median(r["p99_ms"] for r in group),
            }
        )

    write_csv(args.run_root / "ingest_and_size.csv", ingest_rows)
    write_csv(args.run_root / "raw_repetitions.csv", raw_rows)
    write_csv(args.run_root / "summary.csv", summary_rows)
    (args.run_root / "summary.json").write_text(
        json.dumps(
            {"ingest": ingest_rows, "raw_repetitions": raw_rows, "summary": summary_rows},
            indent=2,
        )
    )

    lines = [
        "# Native Parquet Random-Blob Writer Tuning",
        "",
        "All candidates store complete blob values inside Parquet and use the same RowSelection reader.",
        "",
        "## Ingest and Size",
        "",
        "| Dataset | Candidate | Page bytes | Write batch | Ingest (s) | Artifact (GiB) | Row groups |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ingest_rows:
        lines.append(
            f"| {row['dataset']} | {row['candidate']} | {row['data_page_size_limit']} | "
            f"{row['write_batch_size']} | {row['ingest_wall_ms'] / 1000.0:.3f} | "
            f"{row['artifact_bytes'] / (1024 ** 3):.3f} | {row['row_group_count']} |"
        )
    lines.extend(
        [
            "",
            "## Median Point-Read Results",
            "",
            "| Dataset | Open mode | Candidate | Wall (ms) | p50 (ms) | p95 (ms) | p99 (ms) |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {row['candidate']} | "
            f"{row['wall_median_ms']:.3f} | {row['p50_median_ms']:.3f} | "
            f"{row['p95_median_ms']:.3f} | {row['p99_median_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Complete Repetition Results",
            "",
            "| Dataset | Open mode | Candidate | Rep | Seed | Wall (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Bytes |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in raw_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {row['candidate']} | "
            f"{row['repetition']} | {row['seed']} | {row['wall_ms']:.3f} | "
            f"{row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} | "
            f"{row['bytes']} |"
        )
    lines.append("")
    (args.run_root / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
