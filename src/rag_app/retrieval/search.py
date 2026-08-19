"""Multi-search retrieval with reranking."""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from .. import config
from .query_understanding import keyword_score, understand_question


NUMBERED_SECTIONS = {
    "1.1": "1.1 Reduction in risk of colorectal cancer in people with Lynch syndrome",
    "1.2": "1.2 Information for people with colorectal cancer",
    "1.3": "1.3 Management of local disease",
    "1.4": "1.4 Molecular biomarkers to guide systemic anticancer therapy",
    "1.5": "1.5 Management of advanced or metastatic colorectal cancer",
    "1.6": "1.6 Ongoing care and support",
}


def is_patient_information_question(question: str) -> bool:
    """Recognise questions that belong to NICE section 1.2."""
    question = question.lower()
    patient_information_cues = (
        "معلومات",
        "يشرح",
        "شرح",
        "فريق الرعاية",
        "فريق العلاج",
        "اتكلم مع مين",
        "أتكلم مع مين",
        "استشير مين",
        "مين يساعدني",
        "المفروض اعمل اي",
        "اعمل اي",
        "أعمل إيه",
        "i have colon cancer",
        "i have colorectal cancer",
        "what should i do",
        "newly diagnosed",
        "care team explain",
    )
    return any(cue in question for cue in patient_information_cues)


def expand_question(question: str) -> str:
    """Add the guideline's wording for patient-information questions.

    The source guideline calls this topic "Information for people with
    colorectal cancer".  Arabic questions such as "what should the care team
    explain?" may use different wording, so this small bilingual expansion
    helps the embedding model retrieve recommendations 1.2.1-1.2.7.
    """
    if is_patient_information_question(question):
        return (
            f"{question}\n"
            "Information for people with colorectal cancer: treatment options, "
            "benefits, risks, side effects and treatment plan."
        )
    return question


def correct_section_title(section_title: str, text: str) -> str:
    """Correct a page-boundary heading using the recommendation number.

    A PDF page can end with the next heading.  When that happens, the page-level
    metadata may say 1.3 while the chunk itself starts with recommendation
    1.2.7.  Citations should follow the recommendation, not the later heading.
    """
    # Regex101: ^\s*-?\s*(1\.\d+)\.\d+
    match = re.match(r"^\s*-?\s*(1\.\d+)\.\d+", text)
    if not match:
        return section_title
    expected = NUMBERED_SECTIONS.get(match.group(1))
    # Regex101: ^1\.\d+\b
    if expected and re.match(r"^1\.\d+\b", section_title) and not section_title.startswith(match.group(1)):
        return expected
    return section_title


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once per running application."""
    return SentenceTransformer(
        config.EMBEDDING_MODEL,
        local_files_only=config.EMBEDDING_LOCAL_FILES_ONLY,
    )


@lru_cache(maxsize=1)
def get_collection():
    """Open the persistent Chroma collection once per running application."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
    return client.get_collection(
        name=config.COLLECTION_NAME, embedding_function=None
    )


def search(question: str, top_k: int = config.TOP_K) -> list[dict]:
    """Understand, multi-search and rerank a bilingual question."""
    model = get_embedding_model()
    understanding = understand_question(str(question), model)
    query_texts = [f"query: {text}" for text in understanding["queries"]]
    query_vectors = model.encode(
        query_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    collection = get_collection()
    candidate_count = min(max(top_k * 4, 20), collection.count())
    results = collection.query(
        query_embeddings=query_vectors.tolist(),
        n_results=candidate_count,
        where={"content_type": "recommendation"},
        include=["documents", "metadatas", "distances"],
    )

    candidates = {}
    for result_index in range(len(results["ids"])):
        for chunk_id, document, metadata, distance in zip(
            results["ids"][result_index], results["documents"][result_index],
            results["metadatas"][result_index], results["distances"][result_index],
        ):
            semantic_score = 1 - float(distance)
            existing = candidates.get(chunk_id)
            if existing and existing["score"] >= semantic_score:
                continue
            candidates[chunk_id] = {
                "chunk_id": chunk_id,
                "score": semantic_score,
                "document_name": metadata["document_name"],
                "page_number": metadata["page_number"],
                "section_title": correct_section_title(metadata["section_title"], document),
                "content_type": metadata.get("content_type", "unknown"),
                "source_url": metadata["source_url"],
                "text": document,
                "intent": understanding["intent"],
            }

    rows = list(candidates.values())
    for row in rows:
        lexical = max(keyword_score(row["text"], text) for text in understanding["queries"])
        boost = 0.0
        if understanding["intent"] == "symptoms_referral" and row["chunk_id"].startswith("ng12-"):
            boost = 0.12
        elif understanding["intent"] == "newly_diagnosed_information":
            if re.match(r"^\s*-?\s*1\.2\.1\b", row["text"]):
                boost = 0.12
            elif re.match(r"^\s*-?\s*1\.2\.\d+", row["text"]):
                boost = 0.06
        row["rerank_score"] = row["score"] + (0.08 * lexical) + boost

    rows.sort(key=lambda row: row["rerank_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows[:top_k]


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
