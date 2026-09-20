#!/usr/bin/env python3
"""Summarize GraphKV JSONL benchmark outputs without external dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise ValueError(f"no samples in {path}")
    critical = [float(row["critical_path_ms"]) for row in rows]
    decisions = sorted({str(row["decision"]) for row in rows})
    return {
        "path": str(path),
        "samples": len(rows),
        "decisions": {
            decision: sum(row["decision"] == decision for row in rows)
            for decision in decisions
        },
        "critical_path_ms": {
            "mean": statistics.mean(critical),
            "median": statistics.median(critical),
            "p95": percentile(critical, 0.95),
            "sum": sum(critical),
        },
        "all_verified": all(bool(row.get("verified")) for row in rows),
        "payload_bytes": int(rows[0]["payload_bytes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
