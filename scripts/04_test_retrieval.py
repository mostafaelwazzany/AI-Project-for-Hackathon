"""Run manual queries or the small bilingual retrieval check against Chroma."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "vector_store" / "chroma"
DEFAULT_INDEX_REPORT = PROJECT_ROOT / "data" / "vector_store" / "index_report.json"
DEFAULT_EVALUATION = PROJECT_ROOT / "data" / "evaluation" / "day1_test_queries.json"
DEFAULT_EVALUATION_REPORT = PROJECT_ROOT / "data" / "evaluation" / "retrieval_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query or evaluate the local vector index.")
    parser.add_argument("--query", help="A single Arabic or English question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--index-report", type=Path, default=DEFAULT_INDEX_REPORT)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVALUATION_REPORT)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def search(
    collection: Any,
    model: SentenceTransformer,
    question: str,
    top_k: int,
) -> list[dict[str, Any]]:
    model_name = str(getattr(model, "_retrieval_model_name", ""))
    # The selected E5 model requires these literal retrieval prefixes in every language.
    query_input = f"query: {question}" if "e5" in model_name.lower() else question
    query_vector = model.encode_query(
        query_input,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "cosine_similarity": round(1.0 - float(distance), 6),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "section_code": metadata.get("section_code"),
                "recommendation_ids": metadata.get("recommendation_ids", ""),
                "content_type": metadata.get("content_type"),
                "source_url": metadata.get("source_url"),
                "document": document,
            }
        )
    return rows


def print_results(question: str, rows: list[dict[str, Any]]) -> None:
    print(f"\nQuery: {question}")
    for row in rows:
        preview = " ".join(row["document"].split())[:220]
        print(
            f"{row['rank']}. {row['chunk_id']} | score={row['cosine_similarity']:.4f} "
            f"| pages={row['page_start']}-{row['page_end']} | {preview}"
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    index_report = json.loads(args.index_report.resolve().read_text(encoding="utf-8"))
    model_name = index_report["embedding_model"]
    collection_name = index_report["collection"]

    model = SentenceTransformer(model_name, device=args.device)
    # Store the exact loaded ID explicitly; model-card string formatting is not a stable API.
    model._retrieval_model_name = model_name
    client = chromadb.PersistentClient(path=str(args.db.resolve()))
    collection = client.get_collection(name=collection_name, embedding_function=None)

    if args.query:
        rows = search(collection, model, args.query, args.top_k)
        print_results(args.query, rows)
        return

    tests = json.loads(args.evaluation.resolve().read_text(encoding="utf-8"))
    results = []
    for test in tests:
        top_k = int(test.get("top_k", args.top_k))
        rows = search(collection, model, test["query"], top_k)
        returned_ids = {row["chunk_id"] for row in rows}
        expected_ids = set(test["expected_any_chunk_ids"])
        passed = bool(returned_ids & expected_ids)
        print_results(test["query"], rows)
        print(f"PASS={passed} expected_any={sorted(expected_ids)}")
        results.append(
            {
                **test,
                "passed": passed,
                "matched_chunk_ids": sorted(returned_ids & expected_ids),
                "results": rows,
            }
        )

    passed_count = sum(item["passed"] for item in results)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model_name,
        "collection": collection_name,
        "test_count": len(results),
        "passed_count": passed_count,
        "all_passed": passed_count == len(results),
        "tests": results,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nEvaluation: {passed_count}/{len(results)} passed")
    print(f"Report: {args.output.resolve()}")
    if passed_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
