#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load_records(json_dir: Path):
    records = []
    for path in sorted(json_dir.glob("blob.*.json")):
        with path.open() as f:
            record = json.load(f)
        if record.get("notes"):
            raise RuntimeError(f"benchmark record has notes: {path}: {record['notes']}")
        latency = record.get("latency")
        if latency is None:
            raise RuntimeError(f"benchmark record has no latency: {path}")
        params = record["meta"]["params"]
        parts = path.name.split(".")
        records.append(
            {
                "path": path,
                "dataset": record["meta"]["dataset"],
                "engine": record["meta"]["engine"],
                "open_mode": params["open_mode"],
                "seed": record["meta"]["seed"],
                "iters": int(params["iters"]),
                "row_count": int(params["row_count"]),
                "wall_ms": float(record["timing"]["wall_time_ms"]),
                "p50_ms": float(latency["p50_us"]) / 1000.0,
                "p95_ms": float(latency["p95_us"]) / 1000.0,
                "repetition": int(parts[-2][1:]),
            }
        )
    return records


def geometric_mean(values):
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"geometric mean requires positive values: {values}")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_pairs(records):
    pairs = defaultdict(dict)
    for record in records:
        key = (record["dataset"], record["open_mode"], record["repetition"])
        if record["engine"] in pairs[key]:
            raise RuntimeError(f"duplicate engine record for {key}: {record['engine']}")
        pairs[key][record["engine"]] = record

    for key, engines in pairs.items():
        if set(engines) != {"lance", "parquet"}:
            raise RuntimeError(f"incomplete engine pair for {key}: {sorted(engines)}")
        lance = engines["lance"]
        parquet = engines["parquet"]
        for field in ("seed", "iters", "row_count"):
            if lance[field] != parquet[field]:
                raise RuntimeError(
                    f"pair mismatch for {key}, {field}: "
                    f"lance={lance[field]} parquet={parquet[field]}"
                )
    return pairs


def summarize(records, pairs):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["open_mode"], record["engine"])].append(
            record
        )

    engine_rows = []
    for key, group in sorted(grouped.items()):
        dataset, open_mode, engine = key
        engine_rows.append(
            {
                "dataset": dataset,
                "open_mode": open_mode,
                "engine": engine,
                "repetitions": len(group),
                "wall_median_ms": statistics.median(r["wall_ms"] for r in group),
                "p50_median_ms": statistics.median(r["p50_ms"] for r in group),
                "p95_median_ms": statistics.median(r["p95_ms"] for r in group),
            }
        )

    ratio_groups = defaultdict(list)
    for (dataset, open_mode, _repetition), engines in pairs.items():
        lance = engines["lance"]
        parquet = engines["parquet"]
        ratio_groups[(dataset, open_mode)].append(
            {
                "wall": parquet["wall_ms"] / lance["wall_ms"],
                "p50": parquet["p50_ms"] / lance["p50_ms"],
                "p95": parquet["p95_ms"] / lance["p95_ms"],
            }
        )

    ratio_rows = []
    for (dataset, open_mode), ratios in sorted(ratio_groups.items()):
        ratio_rows.append(
            {
                "dataset": dataset,
                "open_mode": open_mode,
                "wall_speedup_median": statistics.median(r["wall"] for r in ratios),
                "p50_speedup_median": statistics.median(r["p50"] for r in ratios),
                "p95_speedup_median": statistics.median(r["p95"] for r in ratios),
            }
        )

    overall_rows = []
    for open_mode in sorted({row["open_mode"] for row in ratio_rows}):
        mode_rows = [row for row in ratio_rows if row["open_mode"] == open_mode]
        overall_rows.append(
            {
                "open_mode": open_mode,
                "datasets": len(mode_rows),
                "wall_speedup_geomean": geometric_mean(
                    [row["wall_speedup_median"] for row in mode_rows]
                ),
                "p50_speedup_geomean": geometric_mean(
                    [row["p50_speedup_median"] for row in mode_rows]
                ),
                "p95_speedup_geomean": geometric_mean(
                    [row["p95_speedup_median"] for row in mode_rows]
                ),
            }
        )
    return engine_rows, ratio_rows, overall_rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.3f}"


def write_markdown(path, records, engine_rows, ratio_rows, overall_rows):
    repetitions = len({record["repetition"] for record in records})
    iters = sorted({record["iters"] for record in records})
    seeds = sorted({record["seed"] for record in records})
    lines = [
        "# Random Blob: Lance v2.2 vs Parquet",
        "",
        "## Method",
        "",
        f"- Repetitions: `{repetitions}`",
        f"- Iterations per repetition: `{iters}`",
        f"- Seeds: `{seeds}` (paired across engines)",
        "- Page cache is dropped before every engine repetition.",
        "- `opened`: both engines open once outside the timed loop.",
        "- `reopen`: both engines reopen inside every timed iteration.",
        "- Speedup is `Parquet / Lance`; values greater than 1 mean Lance is faster.",
        "",
        "## Median Engine Results",
        "",
        "| Dataset | Open mode | Engine | Wall (ms) | p50 (ms) | p95 (ms) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in engine_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | {row['engine']} | "
            f"{fmt(row['wall_median_ms'])} | {fmt(row['p50_median_ms'])} | "
            f"{fmt(row['p95_median_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Dataset Speedup",
            "",
            "| Dataset | Open mode | Wall | p50 | p95 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in ratio_rows:
        lines.append(
            f"| {row['dataset']} | {row['open_mode']} | "
            f"{fmt(row['wall_speedup_median'])}x | "
            f"{fmt(row['p50_speedup_median'])}x | "
            f"{fmt(row['p95_speedup_median'])}x |"
        )

    lines.extend(
        [
            "",
            "## Cross-Dataset Geomean Speedup",
            "",
            "| Open mode | Datasets | Wall | p50 | p95 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"| {row['open_mode']} | {row['datasets']} | "
            f"{fmt(row['wall_speedup_geomean'])}x | "
            f"{fmt(row['p50_speedup_geomean'])}x | "
            f"{fmt(row['p95_speedup_geomean'])}x |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.run_root / "json")
    if not records:
        raise RuntimeError("no random blob records found")
    pairs = validate_pairs(records)
    engine_rows, ratio_rows, overall_rows = summarize(records, pairs)

    write_csv(args.run_root / "summary_engine.csv", engine_rows)
    write_csv(args.run_root / "summary_speedup.csv", ratio_rows)
    write_csv(args.run_root / "summary_overall.csv", overall_rows)
    write_markdown(
        args.run_root / "SUMMARY.md",
        records,
        engine_rows,
        ratio_rows,
        overall_rows,
    )


if __name__ == "__main__":
    main()
