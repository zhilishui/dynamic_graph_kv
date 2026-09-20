"""Metric-system core kept outside the runtime decision path."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, ClassVar, Self, TextIO


class JsonlMetricWriter:
    """Append validated agent-invocation records to one JSONL result file."""

    REQUIRED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "request_id",
            "role",
            "instance",
            "layout",
            "decision",
            "observed_state_ready_ms",
            "verified",
        }
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._output: TextIO | None = None

    def __enter__(self) -> Self:
        self._output = self.path.open("w", encoding="utf-8")
        return self

    def write(self, record: dict[str, Any]) -> None:
        if self._output is None:
            raise RuntimeError("metric writer is not open")
        missing = self.REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"metric record is missing fields: {sorted(missing)}")
        self._output.write(json.dumps(record) + "\n")
        self._output.flush()

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self._output is not None:
            self._output.close()
            self._output = None


def percentile(values: list[float], fraction: float) -> float:
    """Return the nearest-rank empirical percentile."""

    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[index]


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95_nearest_rank": percentile(values, 0.95),
        "sum": sum(values),
    }


def summarize_result_file(path: Path) -> dict[str, object]:
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
        "payload_bytes": max(int(row["payload_bytes"]) for row in rows),
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
