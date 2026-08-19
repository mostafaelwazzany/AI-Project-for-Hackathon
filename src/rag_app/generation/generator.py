"""Grounded RAG generation pipeline: retrieve -> generate -> cite or refuse."""

from __future__ import annotations

import argparse
import os
import re
import sys
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .. import config
from ..retrieval.search import get_embedding_model, search
from ..utils.text import is_arabic, terminal_display
from .citation import citation_for
from .prompt_builder import SYSTEM_PROMPT, build_context, output_instructions


REFUSAL = (
    "I couldn't find enough information in the indexed colorectal cancer "
    "guideline to answer this confidently. Please rephrase the question or "
    "consult a clinician."
)

ARABIC_REFUSAL = (
    "لم أجد معلومات كافية في دليل سرطان القولون والمستقيم المفهرس للإجابة "
    "عن هذا السؤال بثقة. يُرجى إعادة صياغة السؤال أو استشارة طبيب."
)

DISCLAIMER = (
    "Disclaimer: This guideline-based answer is for information only and does "
    "not replace advice from a qualified clinician."
)
ARABIC_DISCLAIMER = (
    "تنبيه: هذه الإجابة المبنية على الدليل للمعلومات فقط، ولا تغني عن استشارة "
    "طبيب أو ممارس صحي مؤهل."
)


def refusal_output(arabic: bool) -> str:
    if arabic:
        return (
            f"التوصية:\n{ARABIC_REFUSAL}\n\n"
            "النص الداعم:\nلم يتم العثور على نص داعم.\n\n"
            "المصدر:\n[لا يوجد مصدر]"
        )
    return (
        f"Recommendation:\n{REFUSAL}\n\n"
        "Excerpt:\nNo supporting passage found.\n\n"
        "Citation:\n[No citation]"
    )


def add_disclaimer(answer: str, arabic: bool) -> str:
    """Add a visible clinical-safety disclaimer once to every answer."""
    disclaimer = ARABIC_DISCLAIMER if arabic else DISCLAIMER
    if disclaimer in answer:
        return answer
    return f"{answer.rstrip()}\n\n{disclaimer}"


def quota_output(arabic: bool) -> str:
    """Return a clear message when Groq's free quota is temporarily full."""
    if arabic:
        return (
            "التوصية:\n"
            "تعذر إنشاء الإجابة لأن مشروع Groq وصل إلى حد الاستخدام المجاني. "
            "افتح Groq Console ثم Limits لمعرفة الحد الذي تم تجاوزه. إذا كان "
            "الحد اليومي، انتظر حتى يتجدد عند منتصف الليل بتوقيت Pacific أو فعّل "
            "Billing؛ وإذا كان الحد في الدقيقة، انتظر ثم أعد السؤال.\n\n"
            "النص الداعم:\n"
            "تم العثور على نتائج من الدليل، لكن خدمة التوليد لم تكن متاحة.\n\n"
            "المصدر:\n[لا يوجد مصدر]"
        )
    return (
        "Recommendation:\n"
            "An answer cannot be generated because this Groq project reached a "
            "free-tier limit. Check Groq Console > Limits. For a daily limit, wait "
        "until the Pacific-time reset or enable billing; for a per-minute limit, "
        "wait briefly and try again.\n\n"
        "Excerpt:\n"
        "Guideline results were found, but the generation service was unavailable.\n\n"
        "Citation:\n[No citation]"
    )


def asks_for_stage_symptoms(question: str, rows: list[dict]) -> bool:
    """Detect stage-specific symptom wording after retrieval identifies the intent."""
    if not rows or rows[0].get("intent") != "symptoms_referral":
        return False
    # Regex101: مرحل(?:ة|ه)|stage
    return bool(re.search(r"مرحل(?:ة|ه)|stage", question, re.IGNORECASE))


def stage_symptoms_output(row: dict, arabic: bool) -> str:
    """Give a safe NG12 answer without assigning warning signs to a cancer stage."""
    citation = citation_for(row, arabic)
    if arabic:
        return (
            "التوصية:\n"
            "النص المسترجع من NICE لا يصنّف هذه العلامات حسب مرحلة السرطان. "
            "وبالنسبة لسرطان القولون والمستقيم، يوصي باستخدام اختبار FIT لتوجيه "
            "الإحالة عند وجود كتلة بالبطن، أو تغير في عادات الإخراج، أو أنيميا نقص "
            "الحديد، أو بعض حالات نزيف المستقيم والألم بالبطن وفقدان الوزن غير "
            "المفسر، بحسب العمر والأعراض المصاحبة.\n\n"
            "النص الداعم:\n"
            "توصي NICE بإجراء اختبار FIT لتوجيه الإحالة عند الاشتباه بسرطان القولون "
            "والمستقيم لدى البالغين الذين لديهم كتلة بالبطن، أو تغير في عادات "
            "الإخراج، أو أنيميا نقص الحديد، مع معايير إضافية مرتبطة بالعمر ونزيف "
            "المستقيم والألم بالبطن وفقدان الوزن.\n\n"
            f"المصدر:\n{citation}"
        )
    return (
        "Recommendation:\n"
        "The retrieved NICE passage does not classify these warning signs by "
        "cancer stage. For colorectal cancer, it recommends FIT to guide referral "
        "for an abdominal mass, a change in bowel habit, iron-deficiency anaemia, "
        "and specified combinations of rectal bleeding, abdominal pain or "
        "unexplained weight loss depending on age.\n\n"
        "Excerpt:\n"
        "Offer quantitative faecal immunochemical testing (FIT) to guide referral "
        "for suspected colorectal cancer in adults with the listed warning signs.\n\n"
        f"Citation:\n{citation}"
    )


def is_valid_answer(answer: str, allowed_citations: set[str], arabic: bool) -> bool:
    """Fail safely if the LLM ignores the required grounded answer format."""
    labels = ("التوصية", "النص الداعم", "المصدر") if arabic else (
        "Recommendation",
        "Excerpt",
        "Citation",
    )
    # Regex101 English example:
    # ^Recommendation:\s*.+?\n\s*Excerpt:\s*.+?\n\s*Citation:\s*(.+)$
    # Regex101 Arabic example:
    # ^التوصية:\s*.+?\n\s*النص الداعم:\s*.+?\n\s*المصدر:\s*(.+)$
    pattern = re.compile(
        rf"^{re.escape(labels[0])}:\s*.+?\n\s*{re.escape(labels[1])}:\s*.+?\n\s*{re.escape(labels[2])}:\s*(.+)$",
        re.DOTALL,
    )
    match = pattern.match(answer.strip())
    if not match:
        return False

    citation_text = match.group(1).strip()
    if citation_text in {"[No citation]", "[لا يوجد مصدر]"}:
        return True
    # Regex101: \[[^\]]+\]
    cited = re.findall(r"\[[^\]]+\]", citation_text)
    return bool(cited) and all(citation in allowed_citations for citation in cited)


def repair_answer_citation(
    answer: str,
    allowed_citations: set[str],
    primary_citation: str,
    arabic: bool,
) -> str | None:
    """Keep a grounded answer when the LLM formats its citation imperfectly.

    The LLM occasionally gives a useful answer and excerpt but slightly changes an
    Arabic section name in the citation.  The old code rejected the whole reply
    in that case.  Here we keep only a correctly structured, supported answer
    and replace its source line with the exact citation from our metadata.
    A real "no evidence" reply is still refused.
    """
    labels = ("التوصية", "النص الداعم", "المصدر") if arabic else (
        "Recommendation",
        "Excerpt",
        "Citation",
    )
    # Regex101 English example:
    # ^(Recommendation:\s*.+?\n\s*Excerpt:\s*.+?\n\s*Citation:\s*)(.+)$
    # Regex101 Arabic example:
    # ^(التوصية:\s*.+?\n\s*النص الداعم:\s*.+?\n\s*المصدر:\s*)(.+)$
    pattern = re.compile(
        rf"^({re.escape(labels[0])}:\s*.+?\n\s*{re.escape(labels[1])}:\s*.+?\n\s*{re.escape(labels[2])}:\s*)(.+)$",
        re.DOTALL,
    )
    match = pattern.match(answer.strip())
    if not match:
        return None

    source_text = match.group(2).strip()
    no_evidence_markers = (
        "[لا يوجد مصدر]",
        "[No citation]",
        "لم يتم العثور على نص داعم",
        "No supporting passage found",
    )
    if any(marker in answer for marker in no_evidence_markers):
        return None

    # Regex101: \[[^\]]+\]
    cited = re.findall(r"\[[^\]]+\]", source_text)
    if cited and all(citation in allowed_citations for citation in cited):
        return answer.strip()

    # The model produced an answer/excerpt from retrieved context, but its
    # displayed citation was not an exact metadata copy.  Never let it invent
    # a source: replace it with the exact top retrieved source.
    return match.group(1) + primary_citation


def response_text(content: object) -> str:
    """Read the LLM's text whether its response is a string or content blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content).strip()


def answer_recommendation(answer: str, arabic: bool) -> str:
    """Extract the recommendation field for the live claim guard."""
    label = "التوصية:" if arabic else "Recommendation:"
    start = answer.find(label)
    if start < 0:
        return ""
    body = answer[start + len(label) :]
    for end_label in ("\n\nالنص الداعم:", "\n\nExcerpt:"):
        if end_label in body:
            body = body.split(end_label, 1)[0]
    return body.strip()


def claim_words(text: str) -> set[str]:
    # Regex101: [A-Za-z0-9\u0600-\u06FF]+
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", text)
        if len(word) > 2
    }


def claims_are_supported(answer: str, rows: list[dict], arabic: bool) -> bool:
    """Independent lexical safety check against retrieved evidence.

    This is intentionally conservative: missing numbers or too many unseen
    terms make the answer fail closed rather than silently reaching the user.
    """
    recommendation = answer_recommendation(answer, arabic)
    if not recommendation:
        return False
    evidence = "\n".join(row["text"] for row in rows)
    if arabic:
        # The evidence is English while the answer is Arabic. A lexical overlap
        # check would reject correct translations, so use our multilingual E5
        # model for a cross-language semantic support check instead.
        model = get_embedding_model()
        claim_vector = model.encode(
            [f"query: {recommendation}"], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )[0]
        evidence_vectors = model.encode(
            [f"passage: {row['text']}" for row in rows],
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        similarity = max(float(claim_vector @ vector) for vector in evidence_vectors)
        answer_numbers = set(re.findall(r"\d+(?:\.\d+)?", recommendation))
        evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
        return similarity >= 0.72 and answer_numbers.issubset(evidence_numbers)
    evidence_words = claim_words(evidence)
    evidence_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence))
    # Regex101: [.!؟\n]+
    claims = [part.strip() for part in re.split(r"[.!؟\n]+", recommendation) if part.strip()]
    for claim in claims:
        current_words = claim_words(claim)
        overlap = len(current_words & evidence_words) / max(len(current_words), 1)
        current_numbers = set(re.findall(r"\d+(?:\.\d+)?", claim))
        unseen = len(current_words - evidence_words)
        if (
            overlap < 0.30
            or not current_numbers.issubset(evidence_numbers)
            or unseen > max(1, int(len(current_words) * 0.40))
        ):
            return False
    return True


@lru_cache(maxsize=1)
def get_generation_model(api_key: str) -> ChatGroq:
    """Create the Groq client once per running application."""
    return ChatGroq(
        model=config.GENERATION_MODEL,
        api_key=api_key,
        max_tokens=config.GENERATION_MAX_OUTPUT_TOKENS,
        timeout=config.GENERATION_TIMEOUT_SECONDS,
        max_retries=0,
        temperature=0.2,
        # Qwen 3.6 defaults to a long reasoning trace. RAG answers need a
        # concise, citation-bound final response, so disable thinking tokens.
        reasoning_effort="none",
        reasoning_format="hidden",
    )


def generate(question: str, rows: list[dict] | None = None) -> str:
    """Retrieve evidence, then produce a grounded answer or a safe refusal."""
    rows = search(question) if rows is None else rows
    arabic = is_arabic(question)
    if not rows or rows[0]["score"] < config.MIN_RETRIEVAL_SCORE:
        return add_disclaimer(refusal_output(arabic), arabic)
    if asks_for_stage_symptoms(question, rows):
        return add_disclaimer(stage_symptoms_output(rows[0], arabic), arabic)
    if rows[0].get("intent") == "symptoms_referral":
        rows = rows[:1]

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    context, allowed_citations = build_context(rows, arabic)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n" + output_instructions(arabic)),
            ("human", "Question:\n{question}\n\nPassages:\n{context}"),
        ]
    )
    model = get_generation_model(api_key)
    try:
        response = (prompt | model).invoke({"question": question, "context": context})
    except Exception as error:
        message = str(error)
        if "RESOURCE_EXHAUSTED" in message or "429" in message:
            return add_disclaimer(quota_output(arabic), arabic)
        raise
    answer = response_text(response.content)
    if is_valid_answer(answer, allowed_citations, arabic):
        if claims_are_supported(answer, rows, arabic):
            return add_disclaimer(answer, arabic)
        return add_disclaimer(refusal_output(arabic), arabic)

    repaired = repair_answer_citation(
        answer,
        allowed_citations,
        citation_for(rows[0], arabic),
        arabic,
    )
    if repaired and claims_are_supported(repaired, rows, arabic):
        return add_disclaimer(repaired, arabic)
    return add_disclaimer(refusal_output(arabic), arabic)


def interactive_mode() -> None:
    """Keep the process alive so loaded models stay in RAM between questions."""
    print("Interactive mode. Type 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("\n" + generate(question))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print normal Unicode text for web/API clients without terminal shaping.",
    )
    args = parser.parse_args()
    if args.interactive:
        interactive_mode()
    elif args.question:
        answer = generate(args.question)
        print(answer if args.raw_output else terminal_display(answer))
    else:
        parser.error("Write a question or use --interactive")
