"""Compare two embedding models using the same chunks, questions, and top-k.

Run:
    python experiment_models.py

The first run downloads multilingual-e5-base from Hugging Face. Each model gets
its own vector store, so the project's main Chroma index is not changed.
"""

from __future__ import annotations

import csv
import gc
import json
from time import perf_counter

import config
import evaluate
import ingest


MODELS = [
    ("multilingual_e5_small", "intfloat/multilingual-e5-small"),
    ("multilingual_e5_base", "intfloat/multilingual-e5-base"),
]


def load_chunks() -> list[dict]:
    """Load the fixed 500/80 chunks so the model is the only variable."""
    with config.CHUNKS_PATH.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize(
    rows: list[dict], model_name: str, model_id: str, index_seconds: float,
    evaluation_seconds: float,
) -> dict:
    scored = [row for row in rows if row["status"] != "REVIEW_REFUSAL"]
    passed = [row for row in scored if row["status"] == "PASS"]
    arabic = [row for row in scored if row["language"] == "ar"]
    english = [row for row in scored if row["language"] == "en"]
    out_of_scope = {
        row["language"]: row["top_score"]
        for row in rows
        if row["status"] == "REVIEW_REFUSAL"
    }

    return {
        "model_name": model_name,
        "model_id": model_id,
        "chunks": len(load_chunks()),
        "top_k": config.TOP_K,
        "found": len(passed),
        "total": len(scored),
        "found_rate": round(len(passed) / len(scored), 4),
        "arabic_found": sum(row["status"] == "PASS" for row in arabic),
        "arabic_total": len(arabic),
        "english_found": sum(row["status"] == "PASS" for row in english),
        "english_total": len(english),
        "average_strict_precision": round(
            sum(float(row["precision_at_k"]) for row in scored) / len(scored), 4
        ),
        "out_of_scope_ar_score": out_of_scope.get("ar", ""),
        "out_of_scope_en_score": out_of_scope.get("en", ""),
        "index_seconds": round(index_seconds, 2),
        "evaluation_seconds": round(evaluation_seconds, 2),
    }


def write_csv(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    chunks = load_chunks()
    experiment_root = config.ROOT / "data" / "experiments" / "models"
    comparison_path = experiment_root / "comparison.csv"
    original = {
        "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
        "CHROMA_PATH": config.CHROMA_PATH,
        "COLLECTION_NAME": config.COLLECTION_NAME,
        "EVALUATION_RESULTS_PATH": config.EVALUATION_RESULTS_PATH,
    }

    comparison = []
    try:
        for model_name, model_id in MODELS:
            folder = experiment_root / model_name
            config.EMBEDDING_MODEL = model_id
            config.CHROMA_PATH = folder / "vector_store"
            config.COLLECTION_NAME = model_name
            config.EVALUATION_RESULTS_PATH = folder / "evaluation_results.csv"

            print(f"\n=== {model_id} ===")
            index_started = perf_counter()
            ingest.build_index(chunks)
            index_seconds = perf_counter() - index_started

            evaluation_started = perf_counter()
            rows = evaluate.evaluate(config.TOP_K)
            evaluation_seconds = perf_counter() - evaluation_started
            evaluate.save_report(rows)

            summary = summarize(
                rows, model_name, model_id, index_seconds, evaluation_seconds
            )
            comparison.append(summary)
            print(
                f"Found: {summary['found']}/{summary['total']} "
                f"({summary['found_rate']:.1%}) | "
                f"Arabic: {summary['arabic_found']}/{summary['arabic_total']} | "
                f"English: {summary['english_found']}/{summary['english_total']} | "
                f"Index: {summary['index_seconds']:.2f}s | "
                f"Evaluation: {summary['evaluation_seconds']:.2f}s"
            )
            gc.collect()
    finally:
        for setting, value in original.items():
            setattr(config, setting, value)

    best = max(
        comparison,
        key=lambda row: (
            row["found_rate"],
            row["average_strict_precision"],
            -row["evaluation_seconds"],
        ),
    )
    for row in comparison:
        row["selected"] = "yes" if row is best else "no"
        row["selection_reason"] = (
            "highest Found@5, then Precision@5, then faster evaluation"
            if row is best
            else ""
        )
    write_csv(comparison, comparison_path)

    print("\n=== Comparison ===")
    for row in comparison:
        print(
            f"{row['model_name']}: found={row['found_rate']:.1%}, "
            f"precision={row['average_strict_precision']:.4f}, "
            f"Arabic={row['arabic_found']}/{row['arabic_total']}, "
            f"English={row['english_found']}/{row['english_total']}"
        )
    print(f"Selected model: {best['model_id']}")
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
