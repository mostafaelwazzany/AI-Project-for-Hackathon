"""Test varied Arabic and English phrasings against the expected intent/source."""

import csv
import sys

import config
from query import search


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = config.ROOT / "data" / "evaluation" / "intent_paraphrases.csv"
    with path.open(encoding="utf-8-sig") as file:
        tests = list(csv.DictReader(file))

    passed = 0
    for test in tests:
        result = search(test["question"], top_k=1)[0]
        ok = (
            result["intent"] == test["expected_intent"]
            and result["chunk_id"].startswith(test["expected_chunk_prefix"])
        )
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} | {test['question']} | {result['intent']} | {result['chunk_id']}")

    print(f"\nIntent/paraphrase accuracy: {passed}/{len(tests)} ({passed / len(tests):.1%})")


if __name__ == "__main__":
    main()
