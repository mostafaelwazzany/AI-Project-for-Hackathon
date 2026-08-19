"""Day 4 safety evaluation: threshold, citations, and faithfulness.

Run:
    python day4_evaluation.py

This runs the complete question -> retrieval -> generation -> citation flow.
Faithfulness is a transparent lexical proxy: it checks whether meaningful
answer terms occur in the retrieved guideline passages.
"""

from __future__ import annotations

import csv
import argparse
import re
import sys
import time

from dotenv import load_dotenv

import config
import evaluate
from generate import (
    ARABIC_REFUSAL,
    REFUSAL,
    citation_for,
    generate,
    is_arabic,
)
from query import search


def extract_citations(answer: str) -> list[str]:
    # Regex101: \[[^\]]+\]
    return re.findall(r"\[[^\]]+\]", answer)


def answer_section(answer: str, arabic: bool) -> str:
    """Extract only the recommendation text, not citation or excerpt labels."""
    label = "التوصية:" if arabic else "Recommendation:"
    start = answer.find(label)
    if start < 0:
        return ""
    body = answer[start + len(label) :]
    end_labels = ("\n\nالنص الداعم:", "\n\nExcerpt:")
    for end_label in end_labels:
        if end_label in body:
            body = body.split(end_label, 1)[0]
    return body.strip()


def words(text: str) -> set[str]:
    # Regex101: [A-Za-z0-9\u0600-\u06FF]+
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", text)
        if len(word) > 2
    }


def faithfulness(answer: str, passages: str, arabic: bool) -> tuple[float, int]:
    """Score supported claims and return (score, unsupported_claim_count)."""
    recommendation = answer_section(answer, arabic)
    if not recommendation:
        return 0.0, 0

    evidence_words = words(passages)
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", passages))
    # Regex101: [.!؟\n]+
    claims = [part.strip() for part in re.split(r"[.!؟\n]+", recommendation) if part.strip()]
    supported = 0
    unsupported = 0
    for claim in claims:
        claim_words = words(claim)
        overlap = len(claim_words & evidence_words) / max(len(claim_words), 1)
        claim_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim))
        numbers_supported = claim_numbers.issubset(evidence_numbers)
        unknown_words = len(claim_words - evidence_words)
        unknown_limit = max(1, int(len(claim_words) * 0.40))
        if overlap >= 0.30 and numbers_supported and unknown_words <= unknown_limit:
            supported += 1
        else:
            unsupported += 1
    return round(supported / max(len(claims), 1), 4), unsupported


def is_refusal(answer: str) -> bool:
    return (
        REFUSAL in answer
        or ARABIC_REFUSAL in answer
        or "[No citation]" in answer
        or "[لا يوجد مصدر]" in answer
    )


def citation_accuracy(answer: str, rows: list[dict], arabic: bool) -> float:
    citations = extract_citations(answer)
    if is_refusal(answer):
        return 1.0
    allowed = {citation_for(row, arabic) for row in rows}
    return 1.0 if citations and set(citations).issubset(allowed) else 0.0


def evaluate_question(question: dict) -> dict:
    text = question["text"]
    arabic = is_arabic(text)
    out_of_scope = question["expected_source"].startswith("NOT COVERED")
    rows = search(text, config.TOP_K)
    top_score = rows[0]["score"] if rows else 0.0
    threshold_refusal = not rows or top_score < config.MIN_RETRIEVAL_SCORE

    if threshold_refusal:
        answer = "[لا يوجد مصدر]" if arabic else "[No citation]"
    else:
        answer = generate_cached(text, question["expected_source"], question["language"])

    retrieved_text = "\n".join(row["text"] for row in rows)
    expected = evaluate.expected_recommendations(question["expected_source"])
    retrieved_expected = any(
        evaluate.is_relevant(row["text"], row["page_number"], question["expected_source"], expected)
        for row in rows
    )
    refusal = is_refusal(answer)
    citation_score = citation_accuracy(answer, rows, arabic)
    if refusal:
        faithfulness_value = ""
        unsupported = 0
    else:
        faithfulness_score, unsupported = faithfulness(answer, retrieved_text, arabic)
        faithfulness_value = faithfulness_score

    return {
        "id": question["id"],
        "variant": question["variant"],
        "language": question["language"],
        "question": text,
        "out_of_scope": "yes" if out_of_scope else "no",
        "top_score": round(top_score, 4),
        "threshold": config.MIN_RETRIEVAL_SCORE,
        "threshold_refusal": "yes" if threshold_refusal else "no",
        "retrieval_found": "yes" if retrieved_expected else "no",
        "refused": "yes" if refusal else "no",
        "correct_decision": "yes" if refusal == out_of_scope else "no",
        "citation_accuracy": citation_score,
        "faithfulness": faithfulness_value,
        "unsupported_claims": unsupported,
        "answer": answer,
    }


_GENERATION_CACHE: dict[tuple[str, str], str] = {}


def generate_cached(question: str, expected_source: str, language: str) -> str:
    """Reuse identical guideline-topic generations across paraphrase variants."""
    key = (expected_source, language)
    if key not in _GENERATION_CACHE:
        _GENERATION_CACHE[key] = generate(question)
    return _GENERATION_CACHE[key]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test question limit")
    parser.add_argument("--language", choices=["ar", "en"], help="Optional language filter")
    parser.add_argument("--out-of-scope", action="store_true", help="Evaluate refusal questions only")
    args = parser.parse_args()
    questions = evaluate.load_questions()
    if args.language:
        questions = [question for question in questions if question["language"] == args.language]
    if args.out_of_scope:
        questions = [question for question in questions if question["expected_source"].startswith("NOT COVERED")]
    if args.limit:
        questions = questions[: args.limit]
    rows = []
    for number, question in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {question['language']} | {question['text']}")
        try:
            rows.append(evaluate_question(question))
        except Exception as error:
            rows.append({
                "id": question["id"],
                "variant": question["variant"],
                "language": question["language"],
                "question": question["text"],
                "error": str(error),
            })
        # Stay below Groq's free-tier per-minute request limit.
        time.sleep(2.1)

    suffix = "_out_of_scope" if args.out_of_scope else "_sample" if args.limit else ""
    output = config.ROOT / "data" / "evaluation" / f"day4_safety_results{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    scored = [row for row in rows if "error" not in row and row["out_of_scope"] == "no"]
    in_scope_refusals = sum(row["refused"] == "yes" for row in scored)
    correct_decisions = sum(row.get("correct_decision") == "yes" for row in rows if "error" not in row)
    answered = [row for row in scored if row["refused"] == "no"]
    mean_citation = sum(float(row["citation_accuracy"]) for row in answered) / max(len(answered), 1)
    faithfulness_rows = [row for row in answered if row["faithfulness"] != ""]
    mean_faithfulness = sum(float(row["faithfulness"]) for row in faithfulness_rows) / max(len(faithfulness_rows), 1)
    summary = {
        "threshold": config.MIN_RETRIEVAL_SCORE,
        "questions": len(rows),
        "in_scope_questions": len(scored),
        "in_scope_refusals": in_scope_refusals,
        "correct_decisions_all_questions": f"{correct_decisions}/{len(rows)}",
        "answered_questions": len(answered),
        "citation_accuracy": round(mean_citation, 4),
        "faithfulness_proxy": round(mean_faithfulness, 4),
        "unsupported_claims_total": sum(int(row.get("unsupported_claims", 0)) for row in rows),
    }
    summary_path = config.ROOT / "data" / "evaluation" / f"day4_safety_summary{suffix}.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    print("\nDay 4 summary")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved details: {output}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
