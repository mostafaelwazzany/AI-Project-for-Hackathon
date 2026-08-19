"""Simple Day 1 ingestion pipeline: PDF -> chunks -> embeddings -> Chroma.

Run:
    python ingest.py
"""

from __future__ import annotations

import json
import re

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm
from sentence_transformers import SentenceTransformer

import config


def clean_markdown(text: str) -> str:
    """Remove PDF-conversion noise while preserving text, bullets and tables."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufffd", "-")
    # Keep the words inside formatting tags; <br> is common inside table cells.
    # A space is cleaner than a punctuation mark when a table cell was wrapped.
    # Regex101: <br\s*/?>
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    # Regex101: </?(?:u|strong|em|b|i)>
    text = re.sub(r"</?(?:u|strong|em|b|i)>", "", text, flags=re.IGNORECASE)
    # Regex101: \*{1,3}
    text = re.sub(r"\*{1,3}", "", text)
    # Regex101: (?m)^\s{0,3}#{1,6}\s+
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    # The converter uses semicolons as visual line breaks inside table cells.
    # Regex101: \s*;\s*
    text = re.sub(r"\s*;\s*", " ", text)

    cleaned_lines = []
    for line in text.splitlines():
        # Regex101: [ \t]+
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    # Regex101: \n{3,}
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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
        raw_text = raw_page.get("text", "").strip()
        headings = []
        for line in raw_text.splitlines():
            # Regex101: ^(#{1,6})\s+(.+)
            match = re.match(r"^(#{1,6})\s+(.+)", line.strip())
            if match:
                # Regex101: [*_]
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
                "text": clean_markdown(raw_text),
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


def split_text(text: str, tokenizer) -> list[str]:
    """Split oversized content using real E5 tokenizer token counts."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def split_page(text: str, tokenizer) -> list[str]:
    """Start a new chunk at each NICE recommendation number when possible."""
    # Regex101: (?m)^(?:-\s*)?\d+\.\d+\.\d+\b
    starts = [
        match.start()
        for match in re.finditer(r"(?m)^(?:-\s*)?\d+\.\d+\.\d+\b", text)
    ]
    if not starts:
        return split_text(text, tokenizer)

    parts = []
    if text[: starts[0]].strip():
        parts.extend(split_text(text[: starts[0]], tokenizer))
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        parts.extend(split_text(text[start:end], tokenizer))
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


def table_id_for(content: str, active_table_id: str) -> str:
    """Give multi-page Markdown tables one stable identifier."""
    # Regex101: \bTable\s+(\d+)\b
    match = re.search(r"\bTable\s+(\d+)\b", content, flags=re.IGNORECASE)
    if match:
        return f"nice-ng151-table-{match.group(1)}"
    return active_table_id if "|" in content else ""


def is_heading_only(content: str) -> bool:
    """Do not index page-fragment headings with no searchable evidence."""
    # Regex101: [^A-Za-z0-9]
    compact = re.sub(r"[^A-Za-z0-9]", "", content)
    # Regex101: ^\s*-?\s*\d+\.\d+\.\d+\b
    has_recommendation = bool(re.match(r"^\s*-?\s*\d+\.\d+\.\d+\b", content))
    return len(compact) < 80 and not has_recommendation and "|" not in content


def is_cross_page_continuation(previous: dict, current: dict) -> bool:
    """Detect a recommendation or table that continues on the next PDF page."""
    if current["page_start"] != previous["page_end"] + 1:
        return False
    same_table = (
        previous["table_id"]
        and previous["table_id"] == current["table_id"]
        and "|" in previous["content"]
        and "|" in current["content"]
    )
    if same_table:
        return True

    previous_text = previous["content"].rstrip()
    # Regex101: ^\s*-?\s*\d+\.\d+\.\d+\b
    current_starts_new_recommendation = bool(
        re.match(r"^\s*-?\s*\d+\.\d+\.\d+\b", current["content"])
    )
    return (
        not current_starts_new_recommendation
        and len(previous_text) < 180
        and not previous_text.endswith((".", "?", "!", ":", ";", "|"))
    )


def merge_cross_page_continuations(records: list[dict]) -> list[dict]:
    """Join split recommendation/table fragments without losing page provenance."""
    merged = []
    for record in records:
        if merged and is_cross_page_continuation(merged[-1], record):
            previous = merged[-1]
            previous["content"] = f"{previous['content']}\n\n{record['content']}"
            previous["page_end"] = record["page_end"]
            continue
        merged.append(record)
    return merged


def page_label(record: dict) -> str:
    """Return one page or an inclusive page range for citations."""
    if record["page_start"] == record["page_end"]:
        return str(record["page_start"])
    return f"{record['page_start']}-{record['page_end']}"


def token_count(text: str, tokenizer) -> int:
    """Count content tokens with the exact tokenizer used by E5."""
    return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))


def split_large_record(record: dict, tokenizer) -> list[dict]:
    """Keep tables readable while ensuring each embedding fits the E5 limit.

    Markdown table rows are separated into blocks by blank lines in this PDF.
    We split only between those blocks, so a row is never cut in half and its
    repeated table header remains with the rows that follow it.
    """
    if token_count(record["content"], tokenizer) <= config.CHUNK_SIZE:
        return [record]

    blocks = [block.strip() for block in record["content"].split("\n\n") if block.strip()]
    parts: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block
        if current and token_count(candidate, tokenizer) > config.CHUNK_SIZE:
            parts.append(current)
            current = block
        else:
            current = candidate
    if current:
        parts.append(current)

    # A single table row can theoretically exceed the limit. Only then use the
    # token-aware splitter as a safe fallback.
    final_parts = []
    for part in parts:
        if token_count(part, tokenizer) <= config.CHUNK_SIZE:
            final_parts.append(part)
        else:
            final_parts.extend(split_text(part, tokenizer))

    return [{**record, "content": part} for part in final_parts]


def chunk_pages(pages: list[dict], tokenizer) -> list[dict]:
    """Create cleaned, table-preserving, citation-ready chunks."""
    raw_chunks = []
    active_table_id = ""

    for page in pages:
        if page["page_number"] not in config.PAGES_TO_INDEX:
            continue
        content_type = content_type_for_page(page["page_number"])
        page_chunks = split_page(page["text"], tokenizer)
        for chunk_number, content in enumerate(page_chunks, start=1):
            content = clean_markdown(content)
            table_id = table_id_for(content, active_table_id)
            if table_id:
                active_table_id = table_id
            if is_heading_only(content):
                continue
            raw_chunks.append(
                {
                    "page_start": page["page_number"],
                    "page_end": page["page_number"],
                    "chunk_number": chunk_number,
                    "document_name": config.DOCUMENT_NAME,
                    "section_title": page["section_title"],
                    "content_type": content_type,
                    "source_url": config.SOURCE_URL,
                    "table_id": table_id,
                    "content": content,
                }
            )

    chunks = []
    for record in merge_cross_page_continuations(raw_chunks):
        page = page_label(record)
        records_to_save = split_large_record(record, tokenizer)
        for part_number, part in enumerate(records_to_save, start=1):
            base_id = f"ng151-p{page}-c{record['chunk_number']}"
            chunk_id = (
                base_id
                if len(records_to_save) == 1
                else f"{base_id}-part{part_number}"
            )
            text_for_embedding = (
                f"Document: {part['document_name']} ({config.GUIDELINE_CODE})\n"
                f"Pages: {page}\n"
                f"Section: {part['section_title']}\n"
                f"Content type: {part['content_type']}\n\n{part['content']}"
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": part["document_name"],
                    "page_number": page,
                    "section_title": part["section_title"],
                    "content_type": part["content_type"],
                    "source_url": part["source_url"],
                    "table_id": part["table_id"],
                    "content": part["content"],
                    "text": text_for_embedding,
                }
            )

    config.CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.CHUNKS_PATH, chunks)
    return chunks


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

    print(f"2. Loading tokenizer from {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        local_files_only=config.EMBEDDING_LOCAL_FILES_ONLY,
    )

    print("3. Creating cleaned, token-aware chunks...")
    chunks = chunk_pages(pages, model.tokenizer)
    from supplementary_sources import load_ng12_colorectal_chunks
    supplementary = load_ng12_colorectal_chunks(model.tokenizer)
    chunks.extend(supplementary)
    write_jsonl(config.CHUNKS_PATH, chunks)
    print(f"   Created {len(chunks)} chunks")
    print(f"   Included {len(supplementary)} colorectal recognition/referral chunks from NICE NG12")

    print(f"4. Creating embeddings with {config.EMBEDDING_MODEL}...")
    build_index(chunks, model)
    print(f"   Saved {len(chunks)} vectors to Chroma")

    print('\nDone. Try: python query.py "your question"')


if __name__ == "__main__":
    main()
