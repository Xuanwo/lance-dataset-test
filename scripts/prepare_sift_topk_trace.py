#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path


def vector_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as file:
        raw = file.read(4)
    if len(raw) != 4:
        raise ValueError(f"empty vector file: {path}")
    dimension = struct.unpack("<i", raw)[0]
    if dimension <= 0:
        raise ValueError(f"invalid vector dimension {dimension} in {path}")
    record_bytes = 4 * (dimension + 1)
    size = path.stat().st_size
    if size % record_bytes:
        raise ValueError(f"invalid vector file size {size} for dimension {dimension}: {path}")
    return size // record_bytes, dimension


def read_ivecs(path: Path, limit: int):
    import numpy as np

    rows, dimension = vector_shape(path)
    if rows < limit:
        raise ValueError(f"{path} has {rows} rows but {limit} are required")
    raw = np.memmap(path, dtype="<i4", mode="r", shape=(rows, dimension + 1))
    if not np.all(raw[:limit, 0] == dimension):
        raise ValueError(f"inconsistent dimensions in {path}")
    return np.asarray(raw[:limit, 1:], dtype=np.int64)


def read_fvecs(path: Path, limit: int):
    import numpy as np

    rows, dimension = vector_shape(path)
    if rows < limit:
        raise ValueError(f"{path} has {rows} rows but {limit} are required")
    raw_i32 = np.memmap(path, dtype="<i4", mode="r", shape=(rows, dimension + 1))
    if not np.all(raw_i32[:limit, 0] == dimension):
        raise ValueError(f"inconsistent dimensions in {path}")
    raw_f32 = np.memmap(path, dtype="<f4", mode="r", shape=(rows, dimension + 1))
    return np.ascontiguousarray(raw_f32[:limit, 1:])


def exact_topk(base: Path, queries: Path, row_count: int, requests: int, top_k: int):
    import faiss

    base_vectors = read_fvecs(base, row_count)
    query_vectors = read_fvecs(queries, requests)
    if base_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            f"dimension mismatch: base={base_vectors.shape[1]} query={query_vectors.shape[1]}"
        )
    index = faiss.IndexFlatL2(base_vectors.shape[1])
    index.add(base_vectors)
    _, indices = index.search(query_vectors, top_k)
    return indices, "faiss-index-flat-l2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a real SIFT exact top-k row trace for blob benchmarks."
    )
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--row-count", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.row_count <= 0 or args.requests <= 0 or args.top_k <= 0:
        raise ValueError("row-count, requests, and top-k must be greater than zero")
    if args.top_k > args.row_count:
        raise ValueError("top-k cannot exceed row-count")

    groundtruth_rows = None
    if args.groundtruth:
        groundtruth_rows, _ = vector_shape(args.groundtruth)

    if args.row_count == 1_000_000 and groundtruth_rows is not None:
        indices = read_ivecs(args.groundtruth, args.requests)[:, : args.top_k]
        generator = "texmex-precomputed-exact-groundtruth"
    else:
        if not args.base or not args.queries:
            raise ValueError(
                "base and queries are required when precomputed ground truth does not match row-count"
            )
        indices, generator = exact_topk(
            args.base, args.queries, args.row_count, args.requests, args.top_k
        )

    requests = []
    for query_index, row in enumerate(indices.tolist()):
        if len(row) != args.top_k:
            raise ValueError(f"query {query_index} returned {len(row)} rows")
        if len(set(row)) != len(row):
            raise ValueError(f"query {query_index} contains duplicate rows")
        if min(row) < 0 or max(row) >= args.row_count:
            raise ValueError(f"query {query_index} contains an out-of-range row")
        requests.append({"query_index": query_index, "indices": row})

    output = {
        "format": "lance-dataset-test-ann-top-k-v1",
        "source": "SIFT1M exact L2 top-k",
        "generator": generator,
        "row_count": args.row_count,
        "top_k": args.top_k,
        "request_count": args.requests,
        "requests": requests,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
