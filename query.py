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
from query_understanding import keyword_score, normalize_question, understand_question


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


def is_diet_discharge_question(question: str) -> bool:
    """Recognise diet/food questions that map to NICE recommendation 1.2.7."""
    question = normalize_question(question)
    cues = (
        "اكل",
        "الاكل",
        "غذا",
        "نظام غذائي",
        "مسموح اكله",
        "اكل ايه",
        "diet",
        "food",
        "eat",
        "allowed food",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_physical_activity_question(question: str) -> bool:
    """Recognise exercise questions that map to NICE recommendation 1.2.7."""
    question = normalize_question(question)
    cues = (
        "رياضه",
        "رياضة",
        "تمرين",
        "اتمرن",
        "نشاط بدني",
        "العب رياضه",
        "exercise",
        "sport",
        "physical activity",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_early_rectal_treatment_question(question: str) -> bool:
    question = normalize_question(question)
    cues = (
        "سرطان المستقيم المبكر",
        "مرحلة مبكرة",
        "early rectal cancer",
        "early stage rectal cancer",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_preoperative_rectal_radiotherapy_question(question: str) -> bool:
    """Recognise radiotherapy before rectal-cancer surgery questions."""
    question = normalize_question(question)
    has_rectal = "المستقيم" in question or "rectal" in question
    has_before_surgery = (
        "قبل جراحة" in question
        or "قبل العمليه" in question
        or "قبل العملية" in question
        or "preoperative" in question
        or "before surgery" in question
    )
    has_radiotherapy = (
        "اشعاعي" in question
        or "إشعاعي" in question
        or "radiotherapy" in question
        or "chemoradiotherapy" in question
    )
    return has_rectal and has_before_surgery and has_radiotherapy


def is_bowel_obstruction_stent_question(question: str) -> bool:
    question = normalize_question(question)
    cues = (
        "انسداد",
        "دعامة",
        "stent",
        "stenting",
        "bowel obstruction",
        "large bowel obstruction",
        "palliative intent",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_liver_metastases_question(question: str) -> bool:
    question = normalize_question(question)
    cues = (
        "انتشر للكبد",
        "ثانوي في الكبد",
        "نقائل الكبد",
        "spread to the liver",
        "liver metastases",
        "secondary liver tumour",
        "metastatic colorectal cancer in the liver",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_resectable_rectal_surgery_question(question: str) -> bool:
    question = normalize_question(question)
    if "غير قابل للاستئصال" in question or "unresectable" in question:
        return False
    cues = (
        "سرطان المستقيم القابل للاستئصال",
        "قابل للاستئصال",
        "resectable rectal cancer",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_lung_metastases_question(question: str) -> bool:
    question = normalize_question(question)
    cues = (
        "نقائل الرئة",
        "انتشر للرئة",
        "الرئة",
        "lung metastases",
        "spread to the lung",
        "pulmonary metastases",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_peritoneal_metastases_question(question: str) -> bool:
    question = normalize_question(question)
    cues = (
        "نقائل الصفاق",
        "الصفاق",
        "البريتون",
        "peritoneum",
        "peritoneal metastases",
        "peritoneal carcinomatosis",
    )
    return any(normalize_question(cue) in question for cue in cues)


def is_msi_mmr_immunotherapy_question(question: str) -> bool:
    question = normalize_question(question)
    cues = (
        "msi",
        "mmr",
        "العلاج المناعي",
        "مناعي",
        "immunotherapy",
        "pembrolizumab",
        "nivolumab",
        "ipilimumab",
    )
    return any(normalize_question(cue) in question for cue in cues)


def expand_question(question: str) -> str:
    """Add the guideline's wording for common user phrasings.

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
    if is_diet_discharge_question(question):
        return (
            f"{question}\n"
            "Colorectal cancer discharge diet advice. Advice on diet including "
            "foods that can cause or contribute to bowel problems such as "
            "diarrhoea, flatulence, incontinence and difficulty emptying the bowels. "
            "Recommendation 1.2.7."
        )
    if is_physical_activity_question(question):
        return (
            f"{question}\n"
            "Colorectal cancer discharge advice: adapting physical activity to "
            "maintain quality of life. Recommendation 1.2.7."
        )
    if is_early_rectal_treatment_question(question):
        return (
            f"{question}\n"
            "Early rectal cancer treatment choices in table 1: transanal excision "
            "TAE TAMIS TEMS, endoscopic submucosal dissection ESD, total "
            "mesorectal excision TME."
        )
    if is_preoperative_rectal_radiotherapy_question(question):
        return (
            f"{question}\n"
            "Preoperative treatment for people with rectal cancer. "
            "Recommendation 1.3.4: do not offer preoperative radiotherapy to "
            "early rectal cancer cT1-T2 cN0 M0 unless part of a clinical trial. "
            "Recommendation 1.3.5: offer preoperative radiotherapy or "
            "chemoradiotherapy to rectal cancer cT1-T2 cN1-N2 M0 or cT3-T4 any cN M0."
        )
    if is_bowel_obstruction_stent_question(question):
        normalized = normalize_question(question)
        intent_text = (
            "Offer either stenting or emergency surgery for people presenting "
            "with acute left-sided large bowel obstruction if potentially "
            "curative treatment is suitable for them."
            if "curative" in normalized or "شافي" in normalized
            else "Acute left-sided large bowel obstruction: consider stenting "
            "for people going to have treatment with palliative intent."
        )
        return (
            f"{question}\n"
            f"{intent_text}"
        )
    if is_liver_metastases_question(question):
        return (
            f"{question}\n"
            "People with metastatic colorectal cancer in the liver: liver "
            "resection, perioperative systemic anticancer therapy, chemotherapy "
            "with local ablative techniques, colorectal liver metastases."
        )
    if is_msi_mmr_immunotherapy_question(question):
        return (
            f"{question}\n"
            "Untreated unresectable or metastatic colorectal cancer with high "
            "microsatellite instability MSI or mismatch repair MMR deficiency: "
            "pembrolizumab, nivolumab plus ipilimumab immunotherapy. "
            "Recommendations 1.5.1 and 1.5.2."
        )
    if is_resectable_rectal_surgery_question(question):
        return (
            f"{question}\n"
            "Recommendation 1.3.6. Offer surgery to people with rectal cancer "
            "cT1-T2 cN1-N2 M0 or cT3-T4 any cN M0 who have a resectable tumour. "
            "Resectable rectal cancer surgery."
        )
    if is_lung_metastases_question(question):
        return (
            f"{question}\n"
            "Colorectal cancer lung metastases: consider metastasectomy, "
            "stereotactic ablative body radiotherapy SABR, or thermal ablation."
        )
    if is_peritoneal_metastases_question(question):
        return (
            f"{question}\n"
            "Colorectal cancer metastases limited to the peritoneum: offer "
            "systemic anticancer therapy and discuss referral to a specialist "
            "cytoreductive surgery and HIPEC centre."
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
        elif understanding["intent"] == "diet_discharge":
            if re.match(r"^\s*-?\s*1\.2\.7\b", row["text"]):
                boost = 0.16
        elif understanding["intent"] == "physical_activity":
            if re.match(r"^\s*-?\s*1\.2\.7\b", row["text"]):
                boost = 0.16
        elif understanding["intent"] == "early_rectal_treatment":
            if row["chunk_id"].startswith("ng151-p10-12") or re.match(r"^\s*-?\s*1\.3\.3\b", row["text"]):
                boost = 0.12
        elif understanding["intent"] == "preoperative_rectal_radiotherapy":
            if re.match(r"^\s*-?\s*1\.3\.[45]\b", row["text"]):
                boost = 0.22
        elif understanding["intent"] == "bowel_obstruction_stent":
            if re.match(r"^\s*-?\s*1\.3\.1\b", row["text"]):
                boost = 0.14
        elif understanding["intent"] == "liver_metastases":
            if re.search(r"\b1\.5\.(1[5-8])\b", row["text"]) or "liver metastases" in row["text"].lower():
                boost = 0.12
        elif understanding["intent"] == "resectable_rectal_surgery":
            if re.match(r"^\s*-?\s*1\.3\.6\b", row["text"]):
                boost = 0.14
        elif understanding["intent"] == "lung_metastases":
            if re.match(r"^\s*-?\s*1\.5\.19\b", row["text"]):
                boost = 0.14
        elif understanding["intent"] == "peritoneal_metastases":
            if re.match(r"^\s*-?\s*1\.5\.21\b", row["text"]):
                boost = 0.14
        elif understanding["intent"] == "msi_mmr_immunotherapy":
            if re.match(r"^\s*-?\s*1\.5\.[12]\b", row["text"]):
                boost = 0.14
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


if __name__ == "__main__":
    main()
