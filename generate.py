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

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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

"""


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


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


def is_valid_answer(answer: str, allowed_citations: set[str], arabic: bool) -> bool:
    """Fail safely if Gemini ignores the required grounded answer format."""
    labels = ("التوصية", "النص الداعم", "المصدر") if arabic else (
        "Recommendation",
        "Excerpt",
        "Citation",
    )
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
    cited = re.findall(r"\[[^\]]+\]", citation_text)
    return bool(cited) and all(citation in allowed_citations for citation in cited)


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
def get_generation_model(api_key: str) -> ChatGoogleGenerativeAI:
    """Create the Gemini client once per running application."""
    return ChatGoogleGenerativeAI(
        model=config.GENERATION_MODEL,
        google_api_key=api_key,
        max_output_tokens=config.GENERATION_MAX_OUTPUT_TOKENS,
    )


def generate(question: str) -> str:
    """Retrieve evidence, then produce a grounded answer or a safe refusal."""
    rows = search(question)
    arabic = is_arabic(question)
    if not rows or rows[0]["score"] < config.MIN_RETRIEVAL_SCORE:
        return refusal_output(arabic)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Copy .env.example to .env and add your key."
        )

    context, allowed_citations = build_context(rows, arabic)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT + "\n" + output_instructions(arabic)),
            ("human", "Question:\n{question}\n\nPassages:\n{context}"),
        ]
    )
    model = get_generation_model(api_key)
    response = (prompt | model).invoke({"question": question, "context": context})
    answer = response_text(response.content)
    return (
        answer
        if is_valid_answer(answer, allowed_citations, arabic)
        else refusal_output(arabic)
    )


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
    args = parser.parse_args()
    if args.interactive:
        interactive_mode()
    elif args.question:
        print(generate(args.question))
    else:
        parser.error("Write a question or use --interactive")


if __name__ == "__main__":
    main()
