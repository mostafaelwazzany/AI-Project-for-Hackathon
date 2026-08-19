"""Understand Arabic/English questions before retrieval without another API call."""

from __future__ import annotations

import re
import unicodedata


INTENTS = {
    "symptoms_referral": {
        "description": "colorectal cancer warning symptoms signs recognition FIT testing and referral",
        "cues": ("اعراض", "أعراض", "علامات", "نزيف", "دم في البراز", "symptom", "sign", "rectal bleeding", "bowel habit"),
    },
    "newly_diagnosed_information": {
        "description": "information and treatment options for a person newly diagnosed with colorectal cancer",
        "cues": ("اتشخصت", "تشخصت", "عندي سرطان", "اعمل اي", "أعمل إيه", "اتكلم مع مين", "diagnosed", "i have colon", "what should i do"),
    },
    "follow_up": {
        "description": "follow-up surveillance after curative colorectal cancer surgery recurrence CEA CT",
        "cues": ("متابعة", "هتابع", "بعد الجراحة", "بعد العملية", "follow-up", "follow up", "after surgery", "after curative", "surveillance", "recurrence"),
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
}


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

    queries = [question, normalized]
    if intent != "general":
        queries.append(INTENTS[intent]["description"])
    return {"original": question, "normalized": normalized, "intent": intent, "confidence": confidence, "queries": list(dict.fromkeys(queries))}


def keyword_score(text: str, query: str) -> float:
    """Simple lexical score used alongside vector similarity."""
    # Regex101: [A-Za-z0-9\u0600-\u06FF]+
    query_words = {word for word in re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", normalize_question(query)) if len(word) > 2}
    text_words = set(re.findall(r"[A-Za-z0-9\u0600-\u06FF]+", normalize_question(text)))
    return len(query_words & text_words) / max(len(query_words), 1)
