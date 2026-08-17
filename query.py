"""Search the local Chroma index.

Run:
    python query.py "What follow-up is recommended after surgery?"
    python query.py --test
"""

from __future__ import annotations

import argparse
import json
import sys

import chromadb
from sentence_transformers import SentenceTransformer

import config


def search(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Embed a question and return the closest chunks from Chroma."""
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    query_vector = model.encode_query(
        f"query: {question}", normalize_embeddings=True
    )

    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    collection = client.get_collection(
        name=config.COLLECTION_NAME, embedding_function=None
    )
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    rows = []
    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": 1 - float(distance),
                "page_number": metadata["page_number"],
                "section_title": metadata["section_title"],
                "content_type": metadata.get("content_type", "unknown"),
                "source_url": metadata["source_url"],
                "text": document,
            }
        )
    return rows


def print_results(question: str, rows: list[dict]) -> None:
    print(f"\nQuestion: {question}")
    for row in rows:
        preview = " ".join(row["text"].split())[:220]
        print(
            f"{row['rank']}. score={row['score']:.4f} | page={row['page_number']} "
            f"| type={row['content_type']} | {row['chunk_id']}\n   {preview}"
        )


def run_tests() -> None:
    """Run known questions and check whether an expected page is returned."""
    tests = json.loads(config.TEST_QUERIES_PATH.read_text(encoding="utf-8"))
    passed = 0
    for test in tests:
        rows = search(test["question"], test.get("top_k", config.TOP_K))
        returned_pages = {row["page_number"] for row in rows}
        success = bool(returned_pages & set(test["expected_pages"]))
        passed += int(success)
        print_results(test["question"], rows)
        print(f"PASS={success} | expected pages={test['expected_pages']}")
    print(f"\nTests passed: {passed}/{len(tests)}")
    if passed != len(tests):
        raise SystemExit(1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.question:
        print_results(args.question, search(args.question, args.top_k))
    else:
        parser.error("Write a question or use --test")


if __name__ == "__main__":
    main()
