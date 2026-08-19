"""PDF loading, markdown cleaning and supplementary source extraction."""

from __future__ import annotations

import re

import pymupdf4llm

from .. import config
from ..utils.io import write_jsonl
from .chunker import split_text


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
    """Extract recommendations 1.3.1-1.3.5 and exclude other cancer sites."""
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
