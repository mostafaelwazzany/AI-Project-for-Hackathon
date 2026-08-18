"""Evaluate retrieval against the Day 2 test questions.

Run:
    python evaluate.py
    python evaluate.py --top-k 3
"""

from __future__ import annotations

import argparse
import csv
import re

import chromadb
from sentence_transformers import SentenceTransformer

import config


def load_questions() -> list[dict]:
    """Load the provided bilingual test questions."""
    with config.TEST_QUESTIONS_PATH.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def expected_recommendations(source: str) -> list[str]:
    """Read recommendation numbers and expand simple ranges such as 1.5.14-1.5.17."""
    numbers = re.findall(r"\d+\.\d+\.\d+", source)
    if len(numbers) != 2:
        return numbers

    start = [int(part) for part in numbers[0].split(".")]
    end = [int(part) for part in numbers[1].split(".")]
    if start[:2] != end[:2] or start[2] > end[2]:
        return numbers
    return [f"{start[0]}.{start[1]}.{number}" for number in range(start[2], end[2] + 1)]


def contains_recommendation(text: str, recommendation: str) -> bool:
    pattern = rf"(?<![\d.]){re.escape(recommendation)}(?![\d.])"
    return re.search(pattern, text) is not None


def page_numbers(page_label: str) -> set[int]:
    """Expand stored page labels such as '10-12' into their page numbers."""
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


def evaluate(top_k: int) -> list[dict]:
    """Embed all questions once, search recommendations, and score the results."""
    questions = load_questions()
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode_query(
        [f"query: {row['text']}" for row in questions],
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = client.get_collection(config.COLLECTION_NAME, embedding_function=None)
    results = collection.query(
        query_embeddings=embeddings.tolist(),
        n_results=min(top_k, collection.count()),
        where={"content_type": "recommendation"},
        include=["documents", "metadatas", "distances"],
    )

    report = []
    for index, question in enumerate(questions):
        source = question["expected_source"]
        out_of_scope = source.startswith("NOT COVERED")
        expected = expected_recommendations(source)
        retrieved = list(
            zip(
                results["ids"][index],
                results["documents"][index],
                results["metadatas"][index],
                results["distances"][index],
            )
        )

        relevant_ranks = []
        for rank, (_, document, metadata, _) in enumerate(retrieved, start=1):
            if not out_of_scope and is_relevant(
                document, metadata["page_number"], source, expected
            ):
                relevant_ranks.append(rank)

        top_id, _, top_metadata, top_distance = retrieved[0]
        if out_of_scope:
            status = "REVIEW_REFUSAL"
            found = ""
            best_rank = ""
            precision = ""
            average_precision = ""
            reciprocal_rank = ""
        else:
            status = "PASS" if relevant_ranks else "FAIL"
            found = "yes" if relevant_ranks else "no"
            best_rank = min(relevant_ranks) if relevant_ranks else ""
            precision = round(len(relevant_ranks) / top_k, 4)
            average_precision = round(
                average_precision_at_k(relevant_ranks, len(expected), top_k), 4
            )
            reciprocal_rank = round(1 / relevant_ranks[0], 4) if relevant_ranks else 0

        report.append(
            {
                "id": question["id"],
                "variant": question["variant"],
                "language": question["language"],
                "question": question["text"],
                "expected_source": source,
                "expected_recommendations": ";".join(expected),
                "top_k": top_k,
                "status": status,
                "found": found,
                "best_rank": best_rank,
                "relevant_in_top_k": len(relevant_ranks) if not out_of_scope else "",
                "precision_at_k": precision,
                "average_precision_at_k": average_precision,
                "reciprocal_rank": reciprocal_rank,
                "top_score": round(1 - float(top_distance), 4),
                "top_chunk_id": top_id,
                "top_page": top_metadata["page_number"],
            }
        )
    return report


def save_report(rows: list[dict]) -> None:
    config.EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.EVALUATION_RESULTS_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> dict:
    """Calculate one summary row that can be opened directly in Excel."""
    scored = [row for row in rows if row["status"] != "REVIEW_REFUSAL"]
    passed = [row for row in scored if row["status"] == "PASS"]
    mean_precision = sum(float(row["precision_at_k"]) for row in scored) / len(scored)
    map_at_k = sum(float(row["average_precision_at_k"]) for row in scored) / len(scored)
    mean_reciprocal_rank = sum(float(row["reciprocal_rank"]) for row in scored) / len(scored)

    summary = {
        "top_k": rows[0]["top_k"],
        "total_questions": len(rows),
        "scored_questions": len(scored),
        "out_of_scope_questions": len(rows) - len(scored),
        "found_expected_evidence": len(passed),
        "found_rate": round(len(passed) / len(scored), 4),
        "mean_precision_at_k": round(mean_precision, 4),
        "map_at_k": round(map_at_k, 4),
        "mrr": round(mean_reciprocal_rank, 4),
    }
    for language in sorted({row["language"] for row in scored}):
        language_rows = [row for row in scored if row["language"] == language]
        language_passed = sum(row["status"] == "PASS" for row in language_rows)
        summary[f"{language}_found_rate"] = round(language_passed / len(language_rows), 4)
        summary[f"{language}_found_count"] = f"{language_passed}/{len(language_rows)}"
    return summary


def save_summary(summary: dict) -> None:
    """Save overall results separately from per-question details."""
    with config.EVALUATION_SUMMARY_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)


def print_summary(rows: list[dict], summary: dict) -> None:
    print(f"\nFound expected evidence: {summary['found_expected_evidence']}/{summary['scored_questions']}")
    print(f"Found rate: {summary['found_rate']:.1%}")
    print(f"Mean Precision@k: {summary['mean_precision_at_k']:.4f}")
    print(f"MAP@k (average ranking quality): {summary['map_at_k']:.4f}")
    print(f"MRR (how early the first correct result appears): {summary['mrr']:.4f}")
    for language in sorted({row["language"] for row in rows if row["status"] != "REVIEW_REFUSAL"}):
        print(f"{language.upper()} found rate: {summary[f'{language}_found_count']}")

    for row in rows:
        if row["status"] == "REVIEW_REFUSAL":
            print(
                f"Out-of-scope ({row['language']}): top score={row['top_score']} "
                f"-> choose a refusal threshold after comparing score distributions"
            )
    print(f"Saved report: {config.EVALUATION_RESULTS_PATH}")
    print(f"Saved summary: {config.EVALUATION_SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    rows = evaluate(args.top_k)
    save_report(rows)
    summary = build_summary(rows)
    save_summary(summary)
    print_summary(rows, summary)


if __name__ == "__main__":
    main()
