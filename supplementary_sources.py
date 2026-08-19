"""Extract only the colorectal section from the approved NICE NG12 PDF."""

from __future__ import annotations

import re

import pymupdf4llm

import config
from ingest import clean_markdown, split_text


def recommendation(text: str, start: str, end: str | None = None) -> str:
    """Return one recommendation from a page fragment."""
    start_index = text.find(start)
    if start_index < 0:
        raise ValueError(f"Could not find NG12 recommendation {start}")
    end_index = text.find(end, start_index) if end else len(text)
    return text[start_index : end_index if end_index >= 0 else len(text)]


def make_chunk(number: str, pages: str, content: str, tokenizer) -> list[dict]:
    """Create citation-ready chunks using the same token-aware splitter."""
    records = []
    for part_number, part in enumerate(split_text(clean_markdown(content), tokenizer), 1):
        chunk_id = f"ng12-rec-{number.replace('.', '-')}-part{part_number}"
        text = (
            f"Document: {config.NG12_DOCUMENT_NAME} ({config.NG12_GUIDELINE_CODE})\n"
            f"Pages: {pages}\nSection: 1.3 Colorectal cancer\n"
            f"Content type: recommendation\n\n{part}"
        )
        records.append(
            {
                "chunk_id": chunk_id,
                "document_name": config.NG12_DOCUMENT_NAME,
                "page_number": pages,
                "section_title": "1.3 Colorectal cancer recognition and referral",
                "content_type": "recommendation",
                "source_url": config.NG12_SOURCE_URL,
                "table_id": "",
                "content": part,
                "text": text,
            }
        )
    return records


def load_ng12_colorectal_chunks(tokenizer) -> list[dict]:
    """Extract recommendations 1.3.1–1.3.5 and exclude other cancer sites."""
    if not config.NG12_PDF_PATH.is_file():
        raise FileNotFoundError(
            f"NG12 PDF not found: {config.NG12_PDF_PATH}. Download it from NICE first."
        )
    pages = pymupdf4llm.to_markdown(
        str(config.NG12_PDF_PATH), pages=[13, 14, 15],
        page_chunks=True, header=False, footer=False,
    )
    page14, page15, page16 = [page["text"] for page in pages]
    contents = {
        "1.3.1": recommendation(page14, "- 1.3.1") + "\n" + recommendation(page15, "- with a change", "- 1.3.2"),
        "1.3.2": recommendation(page15, "- 1.3.2", "1.3.3"),
        "1.3.3": recommendation(page15, "1.3.3"),
        "1.3.4": recommendation(page16, "- 1.3.4", "- 1.3.5"),
        "1.3.5": recommendation(page16, "- 1.3.5", "#### **Anal cancer**"),
    }
    page_labels = {"1.3.1": "14-15", "1.3.2": "15", "1.3.3": "15", "1.3.4": "16", "1.3.5": "16"}
    chunks = []
    for number, content in contents.items():
        chunks.extend(make_chunk(number, page_labels[number], content, tokenizer))
    return chunks
