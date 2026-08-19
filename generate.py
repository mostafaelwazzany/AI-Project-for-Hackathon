"""Day 3 grounded RAG pipeline: retrieve -> generate -> cite or refuse.

Run:
    python generate.py "What follow-up is recommended after surgery?"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from functools import lru_cache

import arabic_reshaper
from bidi.algorithm import get_display
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import config
from query import get_embedding_model, search


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

ARABIC_SECTIONS = {
    "1.1 Reduction in risk of colorectal cancer in people with Lynch syndrome": "1.1 تقليل خطر سرطان القولون والمستقيم لدى الأشخاص المصابين بمتلازمة لينش",
    "1.2 Information for people with colorectal cancer": "1.2 معلومات للأشخاص المصابين بسرطان القولون والمستقيم",
    "1.3 Management of local disease": "1.3 علاج المرض الموضعي",
    "1.3 Colorectal cancer recognition and referral": "1.3 التعرّف على سرطان القولون والمستقيم والإحالة",
    "1.4 Molecular biomarkers to guide systemic anticancer therapy": "1.4 المؤشرات الحيوية الجزيئية لتوجيه العلاج الجهازي المضاد للسرطان",
    "1.5 Management of advanced or metastatic colorectal cancer": "1.5 علاج سرطان القولون والمستقيم المتقدم أو المنتشر",
    "Follow-up for detection of local recurrence and distant metastases": "المتابعة لاكتشاف الانتكاس الموضعي والنقائل البعيدة",
    "People with rectal cancer": "الأشخاص المصابون بسرطان المستقيم",
    "People with colon cancer": "الأشخاص المصابون بسرطان القولون",
    "People with locally advanced or recurrent rectal cancer": "الأشخاص المصابون بسرطان المستقيم المتقدم موضعياً أو الناكس",
    "Surgery for people with rectal cancer": "الجراحة للأشخاص المصابين بسرطان المستقيم",
    "Surgical technique for people with rectal cancer": "التقنية الجراحية لسرطان المستقيم",
    "Surgical technique for people with colon cancer": "التقنية الجراحية لسرطان القولون",
    "Preoperative treatment for people with rectal cancer": "العلاج قبل الجراحة لسرطان المستقيم",
    "Other systemic anticancer therapy for untreated disease": "العلاج الجهازي المضاد للسرطان الآخر للمرض غير المعالج",
    "BRAF V600E mutation-positive disease": "المرض الإيجابي لطفرة BRAF V600E",
    "Neurotrophic tyrosine receptor kinase (NTRK) fusion-positive solid tumours": "الأورام الصلبة الإيجابية لاندماج NTRK",
    "People with metastatic colorectal cancer in the lung": "سرطان القولون والمستقيم المنتشر إلى الرئة",
    "People with metastatic colorectal cancer in the peritoneum": "سرطان القولون والمستقيم المنتشر إلى الصفاق",
}

SYSTEM_PROMPT = """You are a citation-bound clinical guideline assistant.
Answer only from the retrieved guideline passages. Do not use medical knowledge
outside the passages. Do not guess diagnoses, dosages, thresholds, or intervals.

If the passages do not directly answer the question, refuse. Do not try to be
helpful by adding outside information.

A question may be broad. If one or more passages contain directly relevant
recommendations, answer by combining only those recommendations. In particular,
for a question about information that a care team should explain, passages about
treatment options, benefits, risks, side effects, and treatment-plan changes
are direct supporting evidence. Do not refuse merely because the wording of the
question differs from the wording of a recommendation.

This assistant is exclusively about colorectal cancer. If a short question
mentions symptoms, stages, treatment, surgery or follow-up without repeating
the cancer type, interpret it as referring to colorectal cancer unless the user
explicitly names a different disease.

If a question asks for symptoms of a particular stage but the passage lists
warning symptoms without assigning them to stages, clearly say that the
retrieved NICE passage does not classify those symptoms by stage, then give the
relevant warning symptoms from the passage. Never label a symptom as belonging
to a stage unless the passage explicitly does so.

Use calibrated language: say “the guideline recommends” when the passage is a
direct recommendation; say “the guideline suggests” only when the passage is
partial or indirect. Never present an inference as a definite clinical fact.

Be concise. The recommendation must be at most 70 words. Use only one short
supporting excerpt of at most 80 words and then immediately provide the citation.
Do not list every retrieved passage or repeat the same point.

"""


def is_arabic(text: str) -> bool:
    # Regex101: [\u0600-\u06FF]
    return bool(re.search(r"[\u0600-\u06FF]", text))


def terminal_display(text: str) -> str:
    """Make Arabic readable in VS Code's LTR-only integrated terminal.

    VS Code's terminal renderer does not consistently apply Arabic shaping or
    bidirectional layout.  This affects display only: generate() still returns
    normal Unicode Arabic for a future web interface or file output.
    """
    if not is_arabic(text):
        return text
    return "\n".join(
        get_display(arabic_reshaper.reshape(line)) if line else ""
        for line in text.splitlines()
    )


def output_instructions(arabic: bool) -> str:
    if arabic:
        return """Reply in Arabic and use exactly this format:
التوصية:
<إجابة قصيرة مباشرة، أو رسالة الرفض>

النص الداعم:
<ترجمة عربية أمينة للنص الداعم فقط، أو "لم يتم العثور على نص داعم.">

المصدر:
<مصدر واحد أو أكثر من المصادر المقدمة كما هو تماماً، أو "[لا يوجد مصدر]">"""
    return """Reply in English and use exactly this format:
Recommendation:
<a short direct answer, or the refusal message>

Excerpt:
<the exact supporting text from one retrieved passage, or "No supporting passage found.">

Citation:
<one or more citations copied exactly from the provided passages, or "[No citation]">"""


def citation_for(row: dict, arabic: bool) -> str:
    """Create the required document / section / page citation."""
    if arabic:
        section = ARABIC_SECTIONS.get(row["section_title"], row["section_title"])
        if row["document_name"] == config.NG12_DOCUMENT_NAME:
            return f"[NICE NG12: الاشتباه بالسرطان والتعرّف عليه والإحالة، القسم: {section}، الصفحة: {row['page_number']}]"
        return f"[NICE NG151: سرطان القولون والمستقيم، القسم: {section}، الصفحة: {row['page_number']}]"
    return (
        f"[{row['document_name']}, Section: {row['section_title']}, "
        f"Page: {row['page_number']}]"
    )


def build_context(rows: list[dict], arabic: bool) -> tuple[str, set[str]]:
    """Pass the retrieved chunks to Gemini exactly as they were indexed."""
    passages = []
    allowed_citations = set()
    for row in rows:
        citation = citation_for(row, arabic)
        allowed_citations.add(citation)
        passages.append(
            f"PASSAGE {row['rank']}\n"
            f"{citation}\n"
            f"Text:\n{row['text'].strip()}"
        )
    return "\n\n---\n\n".join(passages), allowed_citations


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
    """Return a clear, formatted message when Groq's free quota is temporarily full."""
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
    """Fail safely if Gemini ignores the required grounded answer format."""
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
    """Keep a grounded answer when Gemini formats its citation imperfectly.

    Gemini occasionally gives a useful answer and excerpt but slightly changes an
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
    """Read Gemini's text whether its response is a string or content blocks."""
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
        # Groq returns 429 when the free-tier request
        # limit is reached.  Keep the interactive application alive instead of
        # exposing a long SDK traceback to the user.
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


if __name__ == "__main__":
    main()
