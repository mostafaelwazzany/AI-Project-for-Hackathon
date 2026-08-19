"""Retrieval evaluation metrics: relevance matching, AP@k, MRR."""

from __future__ import annotations

import re


def expected_recommendations(source: str) -> list[str]:
    """Read recommendation numbers and expand simple ranges such as 1.5.14-1.5.17."""
    # Regex101: \d+\.\d+\.\d+
    numbers = re.findall(r"\d+\.\d+\.\d+", source)
    if len(numbers) != 2:
        return numbers

    start = [int(part) for part in numbers[0].split(".")]
    end = [int(part) for part in numbers[1].split(".")]
    if start[:2] != end[:2] or start[2] > end[2]:
        return numbers
    return [f"{start[0]}.{start[1]}.{number}" for number in range(start[2], end[2] + 1)]


def contains_recommendation(text: str, recommendation: str) -> bool:
    # Regex101 template (replace 1.6.1 with the expected recommendation):
    # (?<![\d.])1\.6\.1(?![\d.])
    pattern = rf"(?<![\d.]){re.escape(recommendation)}(?![\d.])"
    return re.search(pattern, text) is not None


def page_numbers(page_label: str) -> set[int]:
    """Expand stored page labels such as '10-12' into their page numbers."""
    # Regex101: \d+
    numbers = [int(value) for value in re.findall(r"\d+", str(page_label))]
    if len(numbers) == 2:
        return set(range(numbers[0], numbers[1] + 1))
    return set(numbers)


def is_relevant(document: str, page_label: str, source: str, expected: list[str]) -> bool:
    """Match expected recommendations; Table 1 spans PDF pages 10-12."""
    if any(contains_recommendation(document, item) for item in expected):
        return True
    return "Table 1" in source and bool(page_numbers(page_label) & {10, 11, 12})


def average_precision_at_k(relevant_ranks: list[int], expected_count: int, top_k: int) -> float:
    """Calculate AP@k from the ranks that contain expected evidence."""
    if not relevant_ranks:
        return 0.0
    precision_sum = sum(position / rank for position, rank in enumerate(relevant_ranks, start=1))
    return precision_sum / min(expected_count, top_k)
