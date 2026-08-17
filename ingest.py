"""Simple Day 1 ingestion pipeline: PDF -> chunks -> embeddings -> Chroma.

Run:
    python ingest.py
"""

from __future__ import annotations

import json
import re

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer

import config


def load_pdf() -> list[dict]:
    """Read the PDF and return one text record per page."""
    if not config.PDF_PATH.is_file():
        raise FileNotFoundError(f"PDF not found: {config.PDF_PATH}")

    raw_pages = pymupdf4llm.to_markdown(
        str(config.PDF_PATH), page_chunks=True, header=False, footer=False
    )

    pages = []
    previous_heading = ""
    previous_heading_level = 0
    for page_number, raw_page in enumerate(raw_pages, start=1):
        text = raw_page.get("text", "").strip()
        headings = []
        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.+)", line.strip())
            if match:
                clean_heading = re.sub(r"[*_]", "", match.group(2)).strip()
                headings.append((len(match.group(1)), clean_heading))

        if headings:
            heading_level, heading = headings[0]
            if (
                previous_heading
                and heading_level == previous_heading_level
                and heading[:1].islower()
            ):
                heading = f"{previous_heading} {heading}"
            previous_heading_level, previous_heading = headings[-1]
        else:
            heading = previous_heading or f"Page {page_number}"
        pages.append(
            {
                "page_number": page_number,
                "section_title": heading,
                "text": text,
            }
        )

    config.MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MARKDOWN_PATH.write_text(
        "\n\n---\n\n".join(
            f"<!-- PAGE: {page['page_number']} -->\n\n{page['text']}" for page in pages
        ),
        encoding="utf-8",
    )
    write_jsonl(config.PAGES_PATH, pages)
    return pages


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text near paragraph or sentence boundaries with small overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            search_from = start + int(size * 0.6)
            possible_ends = [
                text.rfind(separator, search_from, end)
                for separator in ("\n\n", ". ", "\n", " ")
            ]
            natural_end = max(possible_ends)
            if natural_end > start:
                end = natural_end + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def split_page(text: str, size: int, overlap: int) -> list[str]:
    """Start a new chunk at each NICE recommendation number when possible."""
    starts = [
        match.start()
        for match in re.finditer(r"(?m)^(?:-\s*)?\d+\.\d+\.\d+\b", text)
    ]
    if not starts:
        return split_text(text, size, overlap)

    parts = []
    if text[: starts[0]].strip():
        parts.extend(split_text(text[: starts[0]], size, overlap))
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        parts.extend(split_text(text[start:end], size, overlap))
    return parts


def content_type_for_page(page_number: int) -> str:
    """Label each useful part of the guideline for filtered retrieval later."""
    if page_number == 5:
        return "overview"
    if page_number == 6:
        return "guidance_context"
    if 7 <= page_number <= 26:
        return "recommendation"
    if 27 <= page_number <= 28:
        return "glossary"
    if 30 <= page_number <= 48:
        return "rationale"
    raise ValueError(f"Page {page_number} has no configured content type")


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Create citation-ready chunks while keeping every chunk on one PDF page."""
    max_chars = config.CHUNK_SIZE * 4
    overlap_chars = config.CHUNK_OVERLAP * 4
    chunks = []

    for page in pages:
        if page["page_number"] not in config.PAGES_TO_INDEX:
            continue
        content_type = content_type_for_page(page["page_number"])
        page_chunks = split_page(page["text"], max_chars, overlap_chars)
        for chunk_number, content in enumerate(page_chunks, start=1):
            chunk_id = f"ng151-p{page['page_number']}-c{chunk_number}"
            text_for_embedding = (
                f"Document: {config.DOCUMENT_NAME} ({config.GUIDELINE_CODE})\n"
                f"Page: {page['page_number']}\n"
                f"Section: {page['section_title']}\n\n{content}"
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": config.DOCUMENT_NAME,
                    "page_number": page["page_number"],
                    "section_title": page["section_title"],
                    "content_type": content_type,
                    "source_url": config.SOURCE_URL,
                    "content": content,
                    "text": text_for_embedding,
                }
            )

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.CHUNKS_PATH, chunks)
    return chunks


def build_index(chunks: list[dict]) -> None:
    """Embed all chunks with Sentence Transformers and save them in Chroma."""
    model = SentenceTransformer(config.EMBEDDING_MODEL)
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
            }
            for chunk in chunks
        ],
    )


def write_jsonl(path, records: list[dict]) -> None:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    print("1. Reading PDF...")
    pages = load_pdf()
    print(f"   Loaded {len(pages)} pages")

    print("2. Creating chunks...")
    chunks = chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks")

    print(f"3. Creating embeddings with {config.EMBEDDING_MODEL}...")
    build_index(chunks)
    print(f"   Saved {len(chunks)} vectors to Chroma")

    print('\nDone. Try: python query.py "your question"')


if __name__ == "__main__":
    main()
