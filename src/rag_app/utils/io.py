"""File I/O helpers used across the pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
