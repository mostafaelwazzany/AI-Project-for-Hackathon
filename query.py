"""Search the local Chroma index.

Run:
    python query.py "What follow-up is recommended after surgery?"
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

import config


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
    patient_information_cues = (
        "معلومات",
        "يشرح",
        "شرح",
        "فريق الرعاية",
        "فريق العلاج",
    )
    return any(cue in question for cue in patient_information_cues)


def expand_question(question: str) -> str:
    """Add the guideline's wording for patient-information questions.

    The source guideline calls this topic "Information for people with
    colorectal cancer".  Arabic questions such as "what should the care team
    explain?" may use different wording, so this small bilingual expansion
    helps the embedding model retrieve recommendations 1.2.1–1.2.7.
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
    match = re.match(r"^\s*-?\s*(1\.\d+)\.\d+", text)
    if not match:
        return section_title
    expected = NUMBERED_SECTIONS.get(match.group(1))
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
    """Embed a question and return the closest chunks from Chroma."""
    model = get_embedding_model()
    query_vector = model.encode_query(
        f"query: {expand_question(question)}", normalize_embeddings=True
    )

    collection = get_collection()
    # Retrieve a few extra candidates for the patient-information intent.  We
    # then rerank section 1.2 recommendations without replacing semantic
    # search; this prevents a generic treatment chunk from pushing the direct
    # patient-information recommendation just outside Top-k.
    patient_information = is_patient_information_question(question)
    candidate_count = max(top_k, 20) if patient_information else top_k
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(candidate_count, collection.count()),
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
        section_title = correct_section_title(metadata["section_title"], document)
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk_id,
                "score": 1 - float(distance),
                "document_name": metadata["document_name"],
                "page_number": metadata["page_number"],
                "section_title": section_title,
                "content_type": metadata.get("content_type", "unknown"),
                "source_url": metadata["source_url"],
                "text": document,
            }
        )

    if patient_information:
        rows.sort(
            key=lambda row: (
                1 if re.match(r"^\s*-?\s*1\.2\.\d+", row["text"]) else 0,
                row["score"],
            ),
            reverse=True,
        )
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


if __name__ == "__main__":
    main()
