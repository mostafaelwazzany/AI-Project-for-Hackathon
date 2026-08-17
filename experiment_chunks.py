"""Compare three chunk-size/overlap configurations without touching the main index.

Run:
    python experiment_chunks.py
"""

from __future__ import annotations

import csv

import config
import evaluate
import ingest


EXPERIMENTS = [
    (300, 50),
    (500, 80),
    (700, 100),
]


def summarize(rows: list[dict], chunk_size: int, overlap: int, chunks: list[dict]) -> dict:
    scored = [row for row in rows if row["status"] != "REVIEW_REFUSAL"]
    passed = [row for row in scored if row["status"] == "PASS"]
    arabic = [row for row in scored if row["language"] == "ar"]
    english = [row for row in scored if row["language"] == "en"]
    recommendation_chunks = [
        chunk for chunk in chunks if chunk["content_type"] == "recommendation"
    ]

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": len(chunks),
        "recommendation_chunks": len(recommendation_chunks),
        "average_recommendation_chars": round(
            sum(len(chunk["content"]) for chunk in recommendation_chunks)
            / len(recommendation_chunks),
            1,
        ),
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
    }


def write_comparison(rows: list[dict], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    experiment_root = config.ROOT / "data" / "experiments" / "chunking"
    comparison_path = experiment_root / "comparison.csv"
    original = {
        "CHUNK_SIZE": config.CHUNK_SIZE,
        "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
        "CHUNKS_PATH": config.CHUNKS_PATH,
        "CHROMA_PATH": config.CHROMA_PATH,
        "COLLECTION_NAME": config.COLLECTION_NAME,
        "EVALUATION_RESULTS_PATH": config.EVALUATION_RESULTS_PATH,
    }

    pages = ingest.load_pdf()
    comparison = []
    try:
        for chunk_size, overlap in EXPERIMENTS:
            name = f"size_{chunk_size}_overlap_{overlap}"
            folder = experiment_root / name

            config.CHUNK_SIZE = chunk_size
            config.CHUNK_OVERLAP = overlap
            config.CHUNKS_PATH = folder / "chunks.jsonl"
            config.CHROMA_PATH = folder / "vector_store"
            config.COLLECTION_NAME = name
            config.EVALUATION_RESULTS_PATH = folder / "evaluation_results.csv"

            print(f"\n=== {chunk_size} tokens / {overlap} overlap ===")
            chunks = ingest.chunk_pages(pages)
            print(f"Created {len(chunks)} chunks")
            ingest.build_index(chunks)
            rows = evaluate.evaluate(config.TOP_K)
            evaluate.save_report(rows)
            summary = summarize(rows, chunk_size, overlap, chunks)
            comparison.append(summary)
            print(
                f"Found: {summary['found']}/{summary['total']} "
                f"({summary['found_rate']:.1%}) | "
                f"Arabic: {summary['arabic_found']}/{summary['arabic_total']} | "
                f"English: {summary['english_found']}/{summary['english_total']}"
            )
    finally:
        for setting, value in original.items():
            setattr(config, setting, value)

    best_score = max(
        (row["found_rate"], row["average_strict_precision"]) for row in comparison
    )
    tied = [
        row
        for row in comparison
        if (row["found_rate"], row["average_strict_precision"]) == best_score
    ]
    baseline = next(
        (
            row
            for row in tied
            if row["chunk_size"] == original["CHUNK_SIZE"]
            and row["overlap"] == original["CHUNK_OVERLAP"]
        ),
        None,
    )
    best = baseline or min(tied, key=lambda row: row["chunks"])
    reason = (
        "accuracy tie; kept the existing baseline"
        if baseline
        else "accuracy tie; selected the smaller index"
    )
    for row in comparison:
        row["selected"] = "yes" if row is best else "no"
        row["selection_reason"] = reason if row is best else ""
    write_comparison(comparison, comparison_path)
    print("\n=== Comparison ===")
    for row in comparison:
        print(
            f"{row['chunk_size']}/{row['overlap']}: "
            f"found={row['found_rate']:.1%}, "
            f"precision={row['average_strict_precision']:.4f}, "
            f"chunks={row['chunks']}"
        )
    print(
        f"Selected configuration: {best['chunk_size']} tokens / "
        f"{best['overlap']} overlap ({reason})"
    )
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
