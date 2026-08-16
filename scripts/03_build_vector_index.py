"""Embed the selected NICE chunks and persist them in a local Chroma index."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "nice_ng151_colorectal_cancer_chunks.jsonl"
)
DEFAULT_DB = PROJECT_ROOT / "data" / "vector_store" / "chroma"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "vector_store" / "index_report.json"
DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_COLLECTION = "nice_ng151_colorectal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create local sentence-transformer embeddings and store them in Chroma."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--scope",
        choices=("initial", "recommendations", "all"),
        default="initial",
        help="initial=the 33 approved MVP chunks; recommendations=all recommendation/table chunks; all=everything",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="For example: cpu, cuda, or leave empty for auto")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_chunks(chunks: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "initial":
        return [chunk for chunk in chunks if chunk.get("in_initial_scope") is True]
    if scope == "recommendations":
        return [
            chunk
            for chunk in chunks
            if chunk.get("content_type") in {"recommendation", "table"}
        ]
    return chunks


def joined(values: list[Any] | None) -> str:
    return "|".join(str(value) for value in (values or []))


def chroma_metadata(chunk: dict[str, Any]) -> dict[str, str | int | bool]:
    """Keep metadata scalar so it works across Chroma versions and filters."""
    heading_path = chunk.get("heading_path") or []
    return {
        # The four fields explicitly required by the Day 1 deck:
        "document_name": str(chunk.get("document_title") or ""),
        "section_title": str(chunk.get("section") or (heading_path[0] if heading_path else "")),
        "page_number": int(chunk.get("page_start") or 0),
        "chunk_id": str(chunk["chunk_id"]),
        # Extra fields needed for citations, filtering, and linked tables:
        "page_start": int(chunk.get("page_start") or 0),
        "page_end": int(chunk.get("page_end") or 0),
        "section_code": str(chunk.get("section_code") or ""),
        "subsection": str(chunk.get("subsection") or ""),
        "content_type": str(chunk.get("content_type") or ""),
        "recommendation_ids": joined(chunk.get("recommendation_ids")),
        "table_id": str(chunk.get("table_id") or ""),
        "linked_chunk_ids": joined(chunk.get("linked_chunk_ids")),
        "source_url": str(chunk.get("source_url") or ""),
        "in_initial_scope": bool(chunk.get("in_initial_scope", False)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_inputs(texts: list[str], model_name: str) -> tuple[list[str], str]:
    """Apply the prompt required by E5 retrieval models."""
    if "e5" in model_name.lower():
        return [f"passage: {text}" for text in texts], "passage: "
    return texts, ""


def main() -> None:
    args = parse_args()
    chunks_path = args.chunks.resolve()
    db_path = args.db.resolve()
    report_path = args.report.resolve()

    chunks = select_chunks(read_jsonl(chunks_path), args.scope)
    if not chunks:
        raise SystemExit(f"No chunks found for scope={args.scope!r}")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)
    texts = [str(chunk["text"]) for chunk in chunks]
    embedding_inputs, document_prefix = document_inputs(texts, args.model)
    embeddings = model.encode_document(
        embedding_inputs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    existing_names = {collection.name for collection in client.list_collections()}
    if args.collection in existing_names:
        # This is a generated collection. Recreate it so stale records cannot survive a rebuild.
        client.delete_collection(name=args.collection)

    collection = client.create_collection(
        name=args.collection,
        embedding_function=None,
        metadata={
            "description": "NICE NG151 colorectal cancer guideline chunks",
            "embedding_model": args.model,
            "scope": args.scope,
        },
        configuration={"hnsw": {"space": "cosine"}},
    )

    for start in range(0, len(chunks), args.batch_size):
        end = start + args.batch_size
        batch = chunks[start:end]
        collection.upsert(
            ids=[str(chunk["chunk_id"]) for chunk in batch],
            documents=[str(chunk["text"]) for chunk in batch],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[chroma_metadata(chunk) for chunk in batch],
        )

    stored_count = collection.count()
    if stored_count != len(chunks):
        raise RuntimeError(f"Expected {len(chunks)} vectors, but Chroma stored {stored_count}")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "collection": args.collection,
        "database_path": str(db_path),
        "scope": args.scope,
        "chunk_count": len(chunks),
        "stored_vector_count": stored_count,
        "embedding_model": args.model,
        "embedding_dimension": int(embeddings.shape[1]),
        "embeddings_normalized": True,
        "distance_metric": "cosine",
        "model_license": "MIT" if args.model == "intfloat/multilingual-e5-small" else "See model card",
        "document_embedding_prefix": document_prefix,
        "query_embedding_prefix": "query: " if document_prefix else "",
        "chunks_path": str(chunks_path),
        "chunks_sha256": sha256(chunks_path),
        "packages": {
            "sentence-transformers": version("sentence-transformers"),
            "chromadb": version("chromadb"),
        },
        "required_metadata_fields": [
            "document_name",
            "section_title",
            "page_number",
            "chunk_id",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Stored {stored_count} vectors in collection '{args.collection}'.")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Chroma database: {db_path}")
    print(f"Build report: {report_path}")


if __name__ == "__main__":
    main()
