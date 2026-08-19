"""Embedding and Chroma vector store indexing."""

from __future__ import annotations

import chromadb
from sentence_transformers import SentenceTransformer

from .. import config


def build_index(chunks: list[dict], model: SentenceTransformer) -> None:
    """Embed all chunks with Sentence Transformers and save them in Chroma."""
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    embeddings = model.encode_document(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    if config.COLLECTION_NAME in {item.name for item in client.list_collections()}:
        client.delete_collection(config.COLLECTION_NAME)

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "section_title": chunk["section_title"],
                "content_type": chunk["content_type"],
                "chunk_id": chunk["chunk_id"],
                "source_url": chunk["source_url"],
                "table_id": chunk["table_id"],
            }
            for chunk in chunks
        ],
    )


def main() -> None:
    """Full ingestion pipeline: PDF -> chunks -> embeddings -> Chroma."""
    from .pdf_loader import load_pdf, load_ng12_colorectal_chunks
    from .chunker import chunk_pages
    from ..utils.io import write_jsonl

    print("1. Reading PDF...")
    pages = load_pdf()
    print(f"   Loaded {len(pages)} pages")

    print(f"2. Loading tokenizer from {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        local_files_only=config.EMBEDDING_LOCAL_FILES_ONLY,
    )

    print("3. Creating cleaned, token-aware chunks...")
    chunks = chunk_pages(pages, model.tokenizer)
    supplementary = load_ng12_colorectal_chunks(model.tokenizer)
    chunks.extend(supplementary)
    write_jsonl(config.CHUNKS_PATH, chunks)
    print(f"   Created {len(chunks)} chunks")
    print(f"   Included {len(supplementary)} colorectal recognition/referral chunks from NICE NG12")

    print(f"4. Creating embeddings with {config.EMBEDDING_MODEL}...")
    build_index(chunks, model)
    print(f"   Saved {len(chunks)} vectors to Chroma")

    print('\nDone. Try: python run_query.py "your question"')
