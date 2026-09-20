#!/usr/bin/env python3
"""Summarize GraphKV JSONL benchmark outputs without external dependencies."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank empirical percentile."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95_nearest_rank": percentile(values, 0.95),
        "sum": sum(values),
    }


def summarize(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"no samples in {path}")
    decisions = sorted({str(row["decision"]) for row in rows})
    result: dict[str, object] = {
        "path": str(path),
        "samples": len(rows),
        "decisions": {
            decision: sum(row["decision"] == decision for row in rows)
            for decision in decisions
        },
        "all_verified": all(bool(row.get("verified")) for row in rows),
        "payload_bytes": int(rows[0]["payload_bytes"]),
    }
    if all("observed_state_ready_ms" in row for row in rows):
        observed = [float(row["observed_state_ready_ms"]) for row in rows]
        result["observed_state_ready_ms"] = timing_summary(observed)

    modeled_key = (
        "modeled_state_ready_ms"
        if all("modeled_state_ready_ms" in row for row in rows)
        else "critical_path_ms"
    )
    if all(modeled_key in row for row in rows):
        modeled = [float(row[modeled_key]) for row in rows]
        result["modeled_state_ready_ms"] = timing_summary(modeled)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
