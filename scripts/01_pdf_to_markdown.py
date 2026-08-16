from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import pymupdf4llm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "raw" / "nice_ng151_colorectal_cancer.pdf"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def page_number(chunk: dict, fallback: int) -> int:
    metadata = chunk.get("metadata", {})
    value = metadata.get("page_number", metadata.get("page", fallback))
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the NICE NG151 PDF to page-aware Markdown and JSONL."
    )
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = args.pdf.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256(pdf_path)
    expected_hash = manifest["sha256"].upper()
    if actual_hash != expected_hash:
        raise ValueError(
            "PDF SHA256 does not match source_manifest.json. "
            f"Expected {expected_hash}, got {actual_hash}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        use_ocr=False,
        header=False,
        footer=False,
        show_progress=True,
    )

    expected_pages = int(manifest["page_count"])
    if len(raw_pages) != expected_pages:
        raise ValueError(f"Expected {expected_pages} pages, extracted {len(raw_pages)}.")

    pages: list[dict] = []
    markdown_parts: list[str] = []
    for index, chunk in enumerate(raw_pages, start=1):
        number = page_number(chunk, index)
        text = chunk.get("text", "").strip()
        page_record = {
            "source_id": manifest["source_id"],
            "guideline_code": manifest["guideline_code"],
            "document_title": manifest["title"],
            "source_url": manifest["source_url"],
            "page_number": number,
            "text": text,
        }
        pages.append(page_record)
        markdown_parts.append(f"<!-- PAGE: {number} -->\n\n{text}")

    stem = manifest["source_id"]
    markdown_path = output_dir / f"{stem}.md"
    pages_path = output_dir / f"{stem}_pages.jsonl"
    report_path = output_dir / f"{stem}_conversion_report.json"

    markdown_path.write_text(
        "\n\n---\n\n".join(markdown_parts) + "\n", encoding="utf-8"
    )
    with pages_path.open("w", encoding="utf-8", newline="\n") as file_handle:
        for page in pages:
            file_handle.write(json.dumps(page, ensure_ascii=False) + "\n")

    empty_pages = [page["page_number"] for page in pages if not page["text"]]
    report = {
        "source_id": manifest["source_id"],
        "input_pdf": str(pdf_path),
        "input_sha256": actual_hash,
        "converter": "pymupdf4llm",
        "converter_version": importlib.metadata.version("pymupdf4llm"),
        "ocr_used": False,
        "page_count": len(pages),
        "empty_pages": empty_pages,
        "total_characters": sum(len(page["text"]) for page in pages),
        "markdown_heading_lines": sum(
            1
            for page in pages
            for line in page["text"].splitlines()
            if line.startswith("#")
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "markdown": str(markdown_path),
            "pages_jsonl": str(pages_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
