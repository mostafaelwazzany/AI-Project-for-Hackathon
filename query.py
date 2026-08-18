"""Search the local Chroma index.

Run:
    python query.py "What follow-up is recommended after surgery?"
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per running application."""
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Open the persistent Chroma collection once per running application."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return client.get_collection(
        name=config.COLLECTION_NAME, embedding_function=None
    )


def search(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Embed a question and return the closest chunks from Chroma."""
    model = get_embedding_model()
    query_vector = model.encode_query(
        f"query: {question}", normalize_embeddings=True
    )

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, collection.count()),
        where={"content_type": "recommendation"},
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
                "document_name": metadata["document_name"],
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
        print("\n" + "=" * 80)
        print(f"Rank: {row['rank']}")
        print(f"Similarity score: {row['score']:.4f}")
        print(f"Document: {row['document_name']}")
        print(f"Section: {row['section_title']}")
        print(f"Page: {row['page_number']}")
        print(f"Content type: {row['content_type']}")
        print(f"Chunk ID: {row['chunk_id']}")
        print(f"Source: {row['source_url']}")
        print("-" * 80)
        print(row["text"].strip())


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    if args.question:
        print_results(args.question, search(args.question, args.top_k))
    else:
        parser.error("Write a question")


if __name__ == "__main__":
    main()
