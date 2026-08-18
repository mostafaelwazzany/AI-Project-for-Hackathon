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
from query import search


REFUSAL = (
    "I couldn't find enough information in the indexed colorectal cancer "
    "guideline to answer this confidently. Please rephrase the question or "
    "consult a clinician."
)

ARABIC_REFUSAL = (
    "لم أجد معلومات كافية في دليل سرطان القولون والمستقيم المفهرس للإجابة "
    "عن هذا السؤال بثقة. يُرجى إعادة صياغة السؤال أو استشارة طبيب."
)

ARABIC_SECTIONS = {
    "1.1 Reduction in risk of colorectal cancer in people with Lynch syndrome": "1.1 تقليل خطر سرطان القولون والمستقيم لدى الأشخاص المصابين بمتلازمة لينش",
    "1.2 Information for people with colorectal cancer": "1.2 معلومات للأشخاص المصابين بسرطان القولون والمستقيم",
    "1.3 Management of local disease": "1.3 علاج المرض الموضعي",
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


@lru_cache(maxsize=1)
def get_generation_model(api_key: str) -> ChatGroq:
    """Create the Groq client once per running application."""
    return ChatGroq(
        model=config.GENERATION_MODEL,
        api_key=api_key,
        max_tokens=config.GENERATION_MAX_OUTPUT_TOKENS,
        temperature=0.2,
        # Qwen 3.6 defaults to a long reasoning trace. RAG answers need a
        # concise, citation-bound final response, so disable thinking tokens.
        reasoning_effort="none",
        reasoning_format="hidden",
    )


def generate(question: str) -> str:
    """Retrieve evidence, then produce a grounded answer or a safe refusal."""
    rows = search(question)
    arabic = is_arabic(question)
    if not rows or rows[0]["score"] < config.MIN_RETRIEVAL_SCORE:
        return refusal_output(arabic)

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
            return quota_output(arabic)
        raise
    answer = response_text(response.content)
    if is_valid_answer(answer, allowed_citations, arabic):
        return answer

    repaired = repair_answer_citation(
        answer,
        allowed_citations,
        citation_for(rows[0], arabic),
        arabic,
    )
    return repaired if repaired else refusal_output(arabic)


def interactive_mode() -> None:
    """Keep the process alive so loaded models stay in RAM between questions."""
    print("Interactive mode. Type 'exit' to stop.")
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            print("\n" + terminal_display(generate(question)))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.interactive:
        interactive_mode()
    elif args.question:
        print(terminal_display(generate(args.question)))
    else:
        parser.error("Write a question or use --interactive")


if __name__ == "__main__":
    main()
