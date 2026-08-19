"""Keep the Python RAG pipeline loaded for the Next.js chat.

The process reads one JSON question per line and writes one JSON result per line.
Keeping it alive avoids reloading the embedding model for every chat message.
"""

import json
import sys
import unicodedata

from dotenv import load_dotenv

from generate import generate, is_arabic
from query import get_collection, get_embedding_model, search


def source_from_row(row: dict) -> dict:
    """Return source metadata needed by the web UI."""
    return {
        "url": row["source_url"],
        "text": row["text"],
        "document": row["document_name"],
        "page": str(row["page_number"]),
        "section": row["section_title"],
        "chunk_id": row["chunk_id"],
    }


def friendly_error(question: str) -> str:
    if is_arabic(question):
        return "تعذر تجهيز الإجابة مؤقتًا. حاول إرسال السؤال مرة أخرى."
    return "The answer could not be prepared temporarily. Please try again."


def main() -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="strict")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    load_dotenv()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("warmup"):
                # Load the local model, run one tiny embedding and open Chroma
                # before the user sends the first real question.
                model = get_embedding_model()
                model.encode(
                    ["query: warmup"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                get_collection().count()
                print(json.dumps({"ready": True}), flush=True)
                continue
            question = unicodedata.normalize(
                "NFC", str(request.get("question", ""))
            ).strip()
            if not question:
                raise ValueError("Question is empty.")
            try:
                rows = search(question)
                result = {
                    "answer": generate(question, rows),
                    "source": source_from_row(rows[0]) if rows else None,
                    "sources": [source_from_row(row) for row in rows],
                }
            except Exception as error:
                # A fast tokenizer can rarely keep a bad state in a long-lived
                # process. Reload it once and retry without exposing SDK errors.
                if "TextEncodeInput" not in str(error):
                    raise
                get_embedding_model.cache_clear()
                rows = search(question)
                result = {
                    "answer": generate(question, rows),
                    "source": source_from_row(rows[0]) if rows else None,
                    "sources": [source_from_row(row) for row in rows],
                }
        except Exception as error:
            print(f"Chat bridge error: {error}", file=sys.stderr, flush=True)
            result = {"error": friendly_error(question if 'question' in locals() else "")}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
