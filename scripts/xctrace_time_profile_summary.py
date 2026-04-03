#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from typing import IO, Any


@dataclass(frozen=True)
class FrameKey:
    name: str
    binary: str

    def display(self) -> str:
        if self.binary:
            return f"{self.name} ({self.binary})"
        return self.name


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize an Instruments Time Profiler trace (.trace) by aggregating xctrace-exported samples."
    )
    p.add_argument("--trace", required=True, help="Input .trace path")
    p.add_argument("--run-number", type=int, default=1, help="Run number inside the trace (default: 1)")
    p.add_argument("--top", type=int, default=30, help="Top N entries to report (default: 30)")
    p.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )
    p.add_argument("--output", help="Write output to this path (default: stdout)")
    return p.parse_args()


def _open_out(path: str | None) -> IO[str]:
    if path is None or path == "-":
        return sys.stdout
    return open(path, "w", encoding="utf-8")


def _crate_of(symbol: str) -> str:
    if not symbol:
        return "<unknown>"
    if "::" in symbol:
        return symbol.split("::", 1)[0]
    if symbol.startswith("_$LT$"):
        return "<rust_mangled>"
    if symbol.startswith("_ZN"):
        return "<rust_mangled>"
    return "<no_namespace>"


def _xctrace_time_profile_proc(trace: str, run_number: int) -> subprocess.Popen[bytes]:
    xpath = f'/trace-toc/run[@number="{run_number}"]/data/table[@schema="time-profile"]'
    proc = subprocess.Popen(
        ["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc


def _summarize_time_profile_xml(stream: IO[bytes]) -> dict[str, Any]:
    binary_by_id: dict[str, str] = {}
    frame_by_id: dict[str, FrameKey] = {}
    weight_by_id: dict[str, int] = {}

    total_weight = 0
    leaf_weight: dict[FrameKey, int] = defaultdict(int)
    inclusive_weight: dict[FrameKey, int] = defaultdict(int)
    inclusive_weight_by_crate: dict[str, int] = defaultdict(int)
    leaf_weight_by_crate: dict[str, int] = defaultdict(int)

    context = ET.iterparse(stream, events=("end",))
    for event, elem in context:
        tag = elem.tag

        if tag == "weight":
            if "id" in elem.attrib and elem.text is not None:
                try:
                    weight_by_id[elem.attrib["id"]] = int(elem.text)
                except ValueError:
                    pass

        if tag == "binary":
            if "id" in elem.attrib and "name" in elem.attrib:
                binary_by_id[elem.attrib["id"]] = elem.attrib["name"]

        if tag == "frame":
            if "id" in elem.attrib and "name" in elem.attrib:
                frame_id = elem.attrib["id"]
                name = elem.attrib["name"]
                binary = ""
                for child in list(elem):
                    if child.tag != "binary":
                        continue
                    if "name" in child.attrib:
                        binary = child.attrib["name"]
                        if "id" in child.attrib:
                            binary_by_id[child.attrib["id"]] = binary
                    elif "ref" in child.attrib:
                        binary = binary_by_id.get(child.attrib["ref"], "")
                frame_by_id[frame_id] = FrameKey(name=name, binary=binary)

        if tag == "row":
            weight_elem = elem.find("weight")
            if weight_elem is None:
                elem.clear()
                continue

            weight: int | None = None
            if "ref" in weight_elem.attrib:
                weight = weight_by_id.get(weight_elem.attrib["ref"])
            elif weight_elem.text is not None:
                try:
                    weight = int(weight_elem.text)
                except ValueError:
                    weight = None

            if weight is None:
                elem.clear()
                continue

            backtrace = elem.find("backtrace")
            if backtrace is None:
                elem.clear()
                continue

            frames: list[FrameKey] = []
            for f in list(backtrace):
                if f.tag != "frame":
                    continue
                if "ref" in f.attrib:
                    key = frame_by_id.get(f.attrib["ref"])
                    if key is not None:
                        frames.append(key)
                    continue
                if "id" in f.attrib:
                    key = frame_by_id.get(f.attrib["id"])
                    if key is not None:
                        frames.append(key)
                    continue
                if "name" in f.attrib:
                    frames.append(FrameKey(name=f.attrib["name"], binary=""))

            if frames:
                total_weight += weight
                leaf_weight[frames[0]] += weight
                leaf_weight_by_crate[_crate_of(frames[0].name)] += weight
                for key in frames:
                    inclusive_weight[key] += weight
                    inclusive_weight_by_crate[_crate_of(key.name)] += weight

            elem.clear()

    def top_n(m: dict[FrameKey, int], n: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, w in sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:n]:
            out.append(
                {
                    "name": key.name,
                    "binary": key.binary,
                    "weight": w,
                    "pct": (w / total_weight * 100.0) if total_weight else 0.0,
                }
            )
        return out

    def top_n_crate(m: dict[str, int], n: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for crate, w in sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:n]:
            out.append(
                {
                    "crate": crate,
                    "weight": w,
                    "pct": (w / total_weight * 100.0) if total_weight else 0.0,
                }
            )
        return out

    return {
        "total_weight": total_weight,
        "top_inclusive": top_n(inclusive_weight, 10_000),
        "top_leaf": top_n(leaf_weight, 10_000),
        "top_crates_inclusive": top_n_crate(inclusive_weight_by_crate, 10_000),
        "top_crates_leaf": top_n_crate(leaf_weight_by_crate, 10_000),
    }


def _render_text(summary: dict[str, Any], top: int) -> str:
    lines: list[str] = []
    total = summary.get("total_weight", 0)
    lines.append(f"total_weight={total}")
    lines.append("")

    def render_table(title: str, rows: list[dict[str, Any]], key: str) -> None:
        lines.append(title)
        for row in rows[:top]:
            pct = row.get("pct", 0.0)
            w = row.get("weight", 0)
            lines.append(f"- {pct:6.2f}%  {w:>12}  {row.get(key)}")
        lines.append("")

    render_table(
        "top_crates_inclusive:",
        summary.get("top_crates_inclusive", []),
        "crate",
    )
    render_table(
        "top_crates_leaf:",
        summary.get("top_crates_leaf", []),
        "crate",
    )
    render_table(
        "top_inclusive_frames:",
        [
            {"pct": r["pct"], "weight": r["weight"], "frame": f'{r["name"]} ({r["binary"]})'.rstrip(" ()")}
            for r in summary.get("top_inclusive", [])
        ],
        "frame",
    )
    render_table(
        "top_leaf_frames:",
        [
            {"pct": r["pct"], "weight": r["weight"], "frame": f'{r["name"]} ({r["binary"]})'.rstrip(" ()")}
            for r in summary.get("top_leaf", [])
        ],
        "frame",
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parse_args()

    proc = _xctrace_time_profile_proc(args.trace, args.run_number)
    assert proc.stdout is not None
    summary = _summarize_time_profile_xml(proc.stdout)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"xctrace export failed with exit code {rc}")

    # Trim to requested top-N to keep output readable and diff-friendly.
    summary["top_inclusive"] = summary["top_inclusive"][: args.top]
    summary["top_leaf"] = summary["top_leaf"][: args.top]
    summary["top_crates_inclusive"] = summary["top_crates_inclusive"][: args.top]

    with _open_out(args.output) as f:
        if args.format == "json":
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
        else:
            f.write(_render_text(summary, top=args.top))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
