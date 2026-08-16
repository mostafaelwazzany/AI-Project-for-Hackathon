from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nice_ng151_colorectal_cancer_pages.jsonl"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "source_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"

SKIP_PAGES = {1, 2, 3, 4, 51}
INITIAL_SCOPE_SECTIONS = {"1.1", "1.2", "1.3", "1.6"}
TARGET_TOKENS = 500
MAX_TOKENS = 800
OVERLAP_TOKENS = 80

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
RECOMMENDATION_RE = re.compile(r"(?m)^(?:-\s*)?(\d+\.\d+\.\d+)\b")
SECTION_RE = re.compile(r"^(1\.\d+)\s+")
TABLE_RE = re.compile(r"^Table\s+(\d+)\b", re.IGNORECASE)


@dataclass
class Segment:
    content: str
    page_start: int
    page_end: int
    heading_path: list[str]
    mode: str
    section_code: str | None
    content_type: str
    recommendation_ids: list[str] = field(default_factory=list)
    table_id: str | None = None
    part_index: int = 1
    part_count: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic, page-aware chunks from the converted NICE guideline."
    )
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def clean_heading(text: str) -> str:
    text = re.sub(r"</?u>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^\*\*(.*?)\*\*$", r"\1", text).strip()
    return re.sub(r"\s+", " ", text)


def clean_content(text: str) -> str:
    text = re.sub(r"</?u>", "", text, flags=re.IGNORECASE)
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def page_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    blocks: list[tuple[str, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if HEADING_RE.match(line):
            blocks.append(("heading", line))
            index += 1
            continue
        if line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            blocks.append(("table", "\n".join(table_lines)))
            continue

        paragraph: list[str] = []
        while index < len(lines):
            current = lines[index].strip()
            if not current or HEADING_RE.match(current) or current.startswith("|"):
                break
            paragraph.append(current)
            index += 1
        blocks.append(("text", "\n".join(paragraph)))
    return blocks


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL on line {line_number}: {error}") from error
    return sorted(records, key=lambda record: int(record["page_number"]))


def current_path(headings: dict[int, str]) -> list[str]:
    return [headings[level] for level in sorted(headings)]


def table_id_from_path(path: list[str], guideline_code: str) -> str | None:
    for title in reversed(path):
        match = TABLE_RE.match(title)
        if match:
            return f"{guideline_code.lower()}-table-{match.group(1)}"
    return None


def is_reference_note(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith(
        (
            "for a short explanation",
            "full details of the evidence",
            "see also nice",
            "for full details",
        )
    )


def segment_document(pages: list[dict], guideline_code: str) -> list[Segment]:
    headings: dict[int, str] = {}
    mode = "other"
    section_code: str | None = None
    segments: list[Segment] = []
    active_recommendation: Segment | None = None

    for page in pages:
        page_number = int(page["page_number"])
        if page_number in SKIP_PAGES:
            continue

        for block_type, raw_block in page_blocks(page["text"]):
            if block_type == "heading":
                match = HEADING_RE.match(raw_block)
                if not match:
                    continue
                level = len(match.group(1))
                title = clean_heading(match.group(2))
                if title.lower().startswith("return to recommendation"):
                    active_recommendation = None
                    continue

                if (
                    level == 4
                    and title.lower() == "metastases"
                    and headings.get(level, "").lower().endswith("distant")
                ):
                    headings[level] = f"{headings[level]} metastases"
                    active_recommendation = None
                    continue

                headings[level] = title
                for deeper_level in [key for key in headings if key > level]:
                    del headings[deeper_level]

                if level == 2:
                    mode = title.lower()
                    section_code = None
                if level == 3 and mode == "recommendations":
                    section_match = SECTION_RE.match(title)
                    section_code = section_match.group(1) if section_match else None
                active_recommendation = None
                continue

            content = clean_content(raw_block)
            if not content or content.lower().startswith("isbn:"):
                continue

            path = current_path(headings)
            table_id = table_id_from_path(path, guideline_code)

            if block_type == "table":
                segment = Segment(
                    content=content,
                    page_start=page_number,
                    page_end=page_number,
                    heading_path=path,
                    mode=mode,
                    section_code=section_code,
                    content_type="table",
                    table_id=table_id,
                )
                segments.append(segment)
                active_recommendation = None
                continue

            recommendation_ids = RECOMMENDATION_RE.findall(content)
            if recommendation_ids and mode == "recommendations":
                segment = Segment(
                    content=content,
                    page_start=page_number,
                    page_end=page_number,
                    heading_path=path,
                    mode=mode,
                    section_code=section_code,
                    content_type="recommendation",
                    recommendation_ids=list(dict.fromkeys(recommendation_ids)),
                )
                segments.append(segment)
                active_recommendation = segment
                continue

            if active_recommendation is not None and not is_reference_note(content):
                active_recommendation.content = (
                    f"{active_recommendation.content}\n\n{content}"
                )
                active_recommendation.page_end = page_number
                continue

            active_recommendation = None
            if is_reference_note(content):
                content_type = "reference_note"
            elif mode == "rationale and impact":
                content_type = "rationale"
            elif mode == "recommendations for research":
                content_type = "research_recommendation"
            elif mode == "update information":
                content_type = "update_information"
            elif mode == "overview":
                content_type = "overview"
            elif any(title.lower() == "terms used in this guideline" for title in path):
                content_type = "glossary"
            else:
                content_type = "supporting_text"

            segments.append(
                Segment(
                    content=content,
                    page_start=page_number,
                    page_end=page_number,
                    heading_path=path,
                    mode=mode,
                    section_code=section_code,
                    content_type=content_type,
                )
            )
    return segments


def token_count(encoding: tiktoken.Encoding, text: str) -> int:
    return len(encoding.encode(text))


def merge_small_segments(
    segments: list[Segment], encoding: tiktoken.Encoding
) -> list[Segment]:
    mergeable_types = {"overview", "supporting_text", "rationale", "update_information"}
    merged: list[Segment] = []
    for segment in segments:
        if not merged or segment.content_type not in mergeable_types:
            merged.append(segment)
            continue
        previous = merged[-1]
        candidate = f"{previous.content}\n\n{segment.content}"
        can_merge = (
            previous.content_type == segment.content_type
            and previous.heading_path == segment.heading_path
            and previous.section_code == segment.section_code
            and segment.page_start <= previous.page_end + 1
            and token_count(encoding, candidate) <= TARGET_TOKENS
        )
        if can_merge:
            previous.content = candidate
            previous.page_end = segment.page_end
        else:
            merged.append(segment)
    return merged


def split_oversized_segments(
    segments: list[Segment], encoding: tiktoken.Encoding
) -> list[Segment]:
    result: list[Segment] = []
    for segment in segments:
        encoded = encoding.encode(segment.content)
        if len(encoded) <= MAX_TOKENS or segment.content_type == "table":
            result.append(segment)
            continue

        windows: list[str] = []
        start = 0
        while start < len(encoded):
            end = min(start + MAX_TOKENS, len(encoded))
            windows.append(encoding.decode(encoded[start:end]).strip())
            if end == len(encoded):
                break
            start = max(end - OVERLAP_TOKENS, start + 1)

        for index, window in enumerate(windows, start=1):
            result.append(
                Segment(
                    content=window,
                    page_start=segment.page_start,
                    page_end=segment.page_end,
                    heading_path=segment.heading_path,
                    mode=segment.mode,
                    section_code=segment.section_code,
                    content_type=segment.content_type,
                    recommendation_ids=segment.recommendation_ids,
                    table_id=segment.table_id,
                    part_index=index,
                    part_count=len(windows),
                )
            )
    return result


def section_and_subsection(path: list[str]) -> tuple[str | None, str | None]:
    section = next((title for title in path if SECTION_RE.match(title)), None)
    ignored = {
        "colorectal cancer",
        "recommendations",
        "rationale and impact",
        "recommendations for research",
        "overview",
        "update information",
    }
    candidates = [
        title
        for title in path
        if title.lower() not in ignored
        and title != section
        and not TABLE_RE.match(title)
    ]
    subsection = candidates[-1] if candidates else None
    return section, subsection


def unique_chunk_id(base: str, used: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}-part-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def build_chunk_records(
    segments: list[Segment], manifest: dict, encoding: tiktoken.Encoding
) -> list[dict]:
    chunks: list[dict] = []
    used_ids: set[str] = set()
    guideline_code = manifest["guideline_code"].lower()

    for segment in segments:
        section, subsection = section_and_subsection(segment.heading_path)
        context_lines = [
            f"Document: {manifest['title']} ({manifest['guideline_code']})",
            f"Pages: {segment.page_start}-{segment.page_end}",
        ]
        if section:
            context_lines.append(f"Section: {section}")
        if subsection:
            context_lines.append(f"Subsection: {subsection}")
        if segment.recommendation_ids:
            context_lines.append(
                f"Recommendation: {', '.join(segment.recommendation_ids)}"
            )
        embedding_text = "\n".join(context_lines) + f"\n\n{segment.content}"

        if segment.recommendation_ids:
            recommendation_slug = "-".join(segment.recommendation_ids[0].split("."))
            base_id = f"{guideline_code}-rec-{recommendation_slug}"
        elif segment.table_id:
            base_id = f"{segment.table_id}-p{segment.page_start}"
        else:
            digest = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()[:8]
            base_id = (
                f"{guideline_code}-{segment.content_type}-p{segment.page_start}-{digest}"
            )
        chunk_id = unique_chunk_id(base_id, used_ids)

        linked_table_ids = []
        if segment.recommendation_ids:
            linked_table_ids = [
                f"{guideline_code}-table-{number}"
                for number in re.findall(r"\btable\s+(\d+)\b", segment.content, re.I)
            ]

        in_initial_scope = (
            segment.mode == "recommendations"
            and segment.section_code in INITIAL_SCOPE_SECTIONS
            and segment.content_type in {"recommendation", "table"}
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_id": manifest["source_id"],
                "guideline_code": manifest["guideline_code"],
                "document_title": manifest["title"],
                "source_url": manifest["source_url"],
                "page_start": segment.page_start,
                "page_end": segment.page_end,
                "heading_path": segment.heading_path,
                "section": section,
                "subsection": subsection,
                "section_code": segment.section_code,
                "recommendation_ids": segment.recommendation_ids,
                "content_type": segment.content_type,
                "in_initial_scope": in_initial_scope,
                "table_id": segment.table_id,
                "linked_table_ids": linked_table_ids,
                "linked_chunk_ids": [],
                "part_index": segment.part_index,
                "part_count": segment.part_count,
                "token_count": token_count(encoding, embedding_text),
                "content": segment.content,
                "text": embedding_text,
            }
        )

    table_groups: dict[str, list[str]] = defaultdict(list)
    table_recommendations: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        if chunk["table_id"]:
            table_groups[chunk["table_id"]].append(chunk["chunk_id"])
        for table_id in chunk["linked_table_ids"]:
            table_recommendations[table_id].append(chunk["chunk_id"])

    for chunk in chunks:
        linked: list[str] = []
        if chunk["table_id"]:
            linked.extend(table_groups[chunk["table_id"]])
            linked.extend(table_recommendations[chunk["table_id"]])
        for table_id in chunk["linked_table_ids"]:
            linked.extend(table_groups.get(table_id, []))
        chunk["linked_chunk_ids"] = sorted(
            chunk_id for chunk_id in set(linked) if chunk_id != chunk["chunk_id"]
        )

    return chunks


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = round((len(values) - 1) * fraction)
    return sorted(values)[index]


def validate(chunks: list[dict]) -> None:
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")
    if any(not chunk["content"].strip() for chunk in chunks):
        raise ValueError("Empty chunk detected.")
    if any(chunk["page_start"] > chunk["page_end"] for chunk in chunks):
        raise ValueError("Invalid page range detected.")

    known_ids = set(chunk_ids)
    broken_links = {
        linked_id
        for chunk in chunks
        for linked_id in chunk["linked_chunk_ids"]
        if linked_id not in known_ids
    }
    if broken_links:
        raise ValueError(f"Broken linked_chunk_ids detected: {sorted(broken_links)}")


def main() -> None:
    args = parse_args()
    pages_path = args.pages.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = load_jsonl(pages_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoding = tiktoken.get_encoding("cl100k_base")

    segments = segment_document(pages, manifest["guideline_code"])
    segments = merge_small_segments(segments, encoding)
    segments = split_oversized_segments(segments, encoding)
    chunks = build_chunk_records(segments, manifest, encoding)
    validate(chunks)

    chunks_path = output_dir / f"{manifest['source_id']}_chunks.jsonl"
    report_path = output_dir / f"{manifest['source_id']}_chunks_report.json"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as file_handle:
        for chunk in chunks:
            file_handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    token_counts = [chunk["token_count"] for chunk in chunks]
    content_type_counts = Counter(chunk["content_type"] for chunk in chunks)
    recommendation_ids = {
        recommendation_id
        for chunk in chunks
        for recommendation_id in chunk["recommendation_ids"]
    }
    report = {
        "source_id": manifest["source_id"],
        "input_pages": len(pages),
        "skipped_pages": sorted(SKIP_PAGES),
        "total_chunks": len(chunks),
        "initial_scope_chunks": sum(chunk["in_initial_scope"] for chunk in chunks),
        "recommendations_found": len(recommendation_ids),
        "table_groups": len({chunk["table_id"] for chunk in chunks if chunk["table_id"]}),
        "content_types": dict(sorted(content_type_counts.items())),
        "tokens": {
            "minimum": min(token_counts),
            "average": round(statistics.mean(token_counts), 1),
            "median": percentile(token_counts, 0.5),
            "p95": percentile(token_counts, 0.95),
            "maximum": max(token_counts),
            "chunks_over_max": sum(count > MAX_TOKENS for count in token_counts),
        },
        "settings": {
            "tokenizer": "cl100k_base",
            "target_tokens": TARGET_TOKENS,
            "max_tokens": MAX_TOKENS,
            "overlap_tokens_for_oversized_text": OVERLAP_TOKENS,
            "initial_scope_sections": sorted(INITIAL_SCOPE_SECTIONS),
        },
        "output": str(chunks_path),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
