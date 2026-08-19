"""Understand Arabic/English questions before retrieval without another API call."""

from __future__ import annotations

import re
import unicodedata


INTENTS = {
    "symptoms_referral": {
        "description": "colorectal cancer warning symptoms signs recognition FIT testing and referral",
        "cues": ("اعراض", "أعراض", "علامات", "سخونه", "سخونة", "حرارة", "حمى", "نزيف", "دم في البراز", "symptom", "sign", "fever", "temperature", "rectal bleeding", "bowel habit"),
    },
    "newly_diagnosed_information": {
        "description": "information and treatment options for a person newly diagnosed with colorectal cancer",
        "cues": ("اتشخصت", "تشخصت", "عندي سرطان", "اعمل اي", "أعمل إيه", "اتكلم مع مين", "diagnosed", "i have colon", "what should i do"),
    },
    "diet_discharge": {
        "description": "colorectal cancer discharge diet advice foods that can cause bowel problems diarrhoea flatulence incontinence difficulty emptying bowels",
        "cues": ("اكل", "أكل", "الاكل", "الأكل", "غذا", "نظام غذائي", "مسموح اكله", "مسموح أكله", "اكل ايه", "آكل إيه", "diet", "food", "eat", "allowed food"),
    },
    "physical_activity": {
        "description": "colorectal cancer discharge advice adapting physical activity to maintain quality of life",
        "cues": ("رياضه", "رياضة", "تمرين", "اتمرن", "نشاط بدني", "مجهود", "العب رياضه", "ألعب رياضة", "exercise", "sport", "physical activity", "work out"),
    },
    "follow_up": {
        "description": "follow-up surveillance after curative colorectal cancer surgery recurrence CEA CT",
        "cues": ("متابعة", "هتابع", "بعد الجراحة", "بعد العملية", "follow-up", "follow up", "after surgery", "after curative", "surveillance", "recurrence"),
    },
    "early_rectal_treatment": {
        "description": "early rectal cancer treatment choices table 1 transanal excision TAE TAMIS TEMS endoscopic submucosal dissection ESD total mesorectal excision TME",
        "cues": ("سرطان المستقيم المبكر", "مرحلة مبكرة", "early rectal cancer", "early-stage rectal cancer"),
    },
    "preoperative_rectal_radiotherapy": {
        "description": "preoperative radiotherapy or chemoradiotherapy before surgery for rectal cancer recommendations 1.3.4 and 1.3.5",
        "cues": ("علاج إشعاعي قبل جراحة سرطان المستقيم", "علاج اشعاعي قبل جراحة سرطان المستقيم", "قبل جراحة سرطان المستقيم", "preoperative radiotherapy", "chemoradiotherapy before surgery", "before rectal cancer surgery"),
    },
    "bowel_obstruction_stent": {
        "description": "acute left-sided large bowel obstruction stenting palliative intent",
        "cues": ("انسداد", "دعامة", "stent", "stenting", "bowel obstruction", "large bowel obstruction", "palliative intent"),
    },
    "liver_metastases": {
        "description": "colorectal cancer liver metastases liver resection perioperative systemic anticancer therapy local ablative techniques chemotherapy",
        "cues": ("انتشر للكبد", "ثانوي في الكبد", "نقائل الكبد", "metastatic colorectal cancer in the liver", "spread to the liver", "liver metastases", "secondary liver tumour"),
    },
    "resectable_rectal_surgery": {
        "description": "resectable rectal cancer offer surgery after discussion by a multidisciplinary team MDT",
        "cues": ("سرطان المستقيم القابل للاستئصال", "قابل للاستئصال", "resectable rectal cancer"),
    },
    "lung_metastases": {
        "description": "colorectal cancer lung metastases consider metastasectomy stereotactic ablative body radiotherapy SABR thermal ablation",
        "cues": ("نقائل الرئة", "انتشر للرئة", "الرئة", "lung metastases", "spread to the lung", "pulmonary metastases"),
    },
    "peritoneal_metastases": {
        "description": "colorectal cancer metastases limited to the peritoneum systemic anticancer therapy referral to specialist cytoreductive surgery and HIPEC centre",
        "cues": ("نقائل الصفاق", "الصفاق", "البريتون", "peritoneum", "peritoneal metastases", "peritoneal carcinomatosis"),
    },
    "treatment": {
        "description": "colorectal cancer treatment options surgery radiotherapy systemic anticancer therapy palliative care",
        "cues": ("علاج", "كيماوي", "اشعاع", "treatment", "chemotherapy", "radiotherapy"),
    },
    "side_effects": {
        "description": "side effects and quality of life after colorectal cancer treatment",
        "cues": ("اثار جانبية", "آثار جانبية", "مضاعفات", "side effect", "complication"),
    },
    "surgery": {
        "description": "colorectal cancer surgery surgical technique colon rectal cancer",
        "cues": ("جراحة", "عملية", "استئصال", "surgery", "operation", "resection"),
    },
    "biomarkers": {
        "description": "colorectal cancer molecular biomarkers RAS BRAF mismatch repair NTRK",
        "cues": ("طفرة", "تحليل جيني", "مؤشرات", "biomarker", "braf", "ras", "ntrk"),
    },
    "msi_mmr_immunotherapy": {
        "description": "untreated unresectable or metastatic colorectal cancer high microsatellite instability MSI or mismatch repair MMR deficiency pembrolizumab nivolumab ipilimumab immunotherapy",
        "cues": ("msi", "mmr", "العلاج المناعي", "مناعي", "immunotherapy", "pembrolizumab", "nivolumab", "ipilimumab"),
    },
}

DOMAIN_CUES = (
    "سرطان القولون",
    "سرطان المستقيم",
    "سرطان الامعاء",
    "القولون والمستقيم",
    "colorectal cancer",
    "colon cancer",
    "rectal cancer",
    "bowel cancer",
)


def normalize_question(text: str) -> str:
    """Normalize common Arabic spelling and punctuation differences."""
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    # Regex101: [\u064B-\u065F\u0670]
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ـ": ""}))
    # Regex101: [^\w\s\u0600-\u06FF-]
    text = re.sub(r"[^\w\s\u0600-\u06FF-]", " ", text)
    # Regex101: \s+
    return re.sub(r"\s+", " ", text).strip()


def add_domain_context(question: str) -> str:
    """Assume short questions refer to colorectal cancer in this specialist app."""
    normalized = normalize_question(question)
    if any(normalize_question(cue) in normalized for cue in DOMAIN_CUES):
        return question
    # Regex101: [\u0600-\u06FF]
    if re.search(r"[\u0600-\u06FF]", question):
        return f"في سياق سرطان القولون والمستقيم: {question}"
    return f"In the context of colorectal cancer: {question}"


def understand_question(question: str, model) -> dict:
    """Return intent plus several search formulations for robust retrieval."""
    normalized = normalize_question(question)
    cue_scores = {
        name: sum(normalize_question(cue) in normalized for cue in data["cues"])
        for name, data in INTENTS.items()
    }
    cue_intent = max(cue_scores, key=cue_scores.get)
    if cue_scores[cue_intent] > 0:
        intent, confidence = cue_intent, 1.0
    else:
        texts = [f"query: {normalized}"] + [f"passage: {data['description']}" for data in INTENTS.values()]
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        similarities = [float(vectors[0] @ vector) for vector in vectors[1:]]
        best = max(range(len(similarities)), key=similarities.__getitem__)
        intent = list(INTENTS)[best] if similarities[best] >= 0.73 else "general"
        confidence = similarities[best]

    domain_question = add_domain_context(question)
    queries = [domain_question, question, normalize_question(domain_question), normalized]
    if intent != "general":
        queries.append(INTENTS[intent]["description"])
    return {
        "original": question,
        "normalized": normalized,
        "domain_question": domain_question,
        "intent": intent,
        "confidence": confidence,
        "queries": list(dict.fromkeys(queries)),
    }


def keyword_score(text: str, query: str) -> float:
    """Simple lexical score used alongside vector similarity."""
    # Regex101: [A-Za-z0-9\u0600-\u06FF]+
    query_words = {word for word in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", normalize_question(query)) if len(word) > 2}
    text_words = set(re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", normalize_question(text)))
    return len(query_words & text_words) / max(len(query_words), 1)
