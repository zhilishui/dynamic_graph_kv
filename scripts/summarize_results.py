#!/usr/bin/env python3
"""Summarize GraphKV JSONL benchmark outputs without external dependencies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphkv.metrics import summarize_result_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize_result_file(path) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
