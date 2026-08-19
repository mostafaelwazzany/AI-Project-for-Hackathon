"""Re-run Day 2 comparisons against the current cleaned 66-question dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

import config
import evaluate
import ingest


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(rows: list[dict]) -> dict:
    result = evaluate.build_summary(rows)
    return {
        "top_k": result["top_k"],
        "total_questions": result["total_questions"],
        "scored_questions": result["scored_questions"],
        "found_expected_evidence": result["found_expected_evidence"],
        "found_rate": result["found_rate"],
        "mean_precision_at_k": result["mean_precision_at_k"],
        "map_at_k": result["map_at_k"],
        "mrr": result["mrr"],
    }


def main() -> None:
    original = {
        name: getattr(config, name)
        for name in (
            "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNKS_PATH", "CHROMA_PATH",
            "COLLECTION_NAME", "EMBEDDING_MODEL", "EVALUATION_RESULTS_PATH",
            "EVALUATION_SUMMARY_PATH",
        )
    }
    experiment_root = config.ROOT / "data" / "experiments_current"
    pages = ingest.load_pdf()
    base_model = SentenceTransformer(config.EMBEDDING_MODEL, local_files_only=True)

    try:
        chunk_rows = []
        for size, overlap in ((300, 50), (450, 80), (600, 100)):
            folder = experiment_root / f"chunks_{size}_{overlap}"
            config.CHUNK_SIZE, config.CHUNK_OVERLAP = size, overlap
            config.CHUNKS_PATH = folder / "chunks.jsonl"
            config.CHROMA_PATH = folder / "chroma"
            config.COLLECTION_NAME = f"current_chunks_{size}_{overlap}"
            config.EVALUATION_RESULTS_PATH = folder / "evaluation_results.csv"
            config.EVALUATION_SUMMARY_PATH = folder / "evaluation_summary.csv"
            chunks = ingest.chunk_pages(pages, base_model.tokenizer)
            ingest.build_index(chunks, base_model)
            rows = evaluate.evaluate(config.TOP_K)
            write_csv(config.EVALUATION_RESULTS_PATH, rows)
            item = summary(rows)
            item.update({"chunk_size": size, "overlap": overlap, "chunks": len(chunks)})
            chunk_rows.append(item)
            print(f"chunk {size}/{overlap}: {item['found_rate']:.1%}")
        write_csv(experiment_root / "chunking_comparison.csv", chunk_rows)

        # Model comparison uses the current final 450/80 chunks, not an old baseline.
        with original["CHUNKS_PATH"].open(encoding="utf-8") as file:
            chunks = [json.loads(line) for line in file if line.strip()]
        model_rows = []
        for name, model_id in (("e5_small", "intfloat/multilingual-e5-small"), ("e5_base", "intfloat/multilingual-e5-base")):
            folder = experiment_root / f"model_{name}"
            config.EMBEDDING_MODEL = model_id
            config.CHROMA_PATH = folder / "chroma"
            config.COLLECTION_NAME = f"current_model_{name}"
            config.EVALUATION_RESULTS_PATH = folder / "evaluation_results.csv"
            config.EVALUATION_SUMMARY_PATH = folder / "evaluation_summary.csv"
            model = SentenceTransformer(model_id, local_files_only=True)
            ingest.build_index(chunks, model)
            rows = evaluate.evaluate(config.TOP_K)
            write_csv(config.EVALUATION_RESULTS_PATH, rows)
            item = summary(rows)
            item.update({"model_name": name, "model_id": model_id, "chunks": len(chunks)})
            model_rows.append(item)
            print(f"model {name}: {item['found_rate']:.1%}")
        write_csv(experiment_root / "model_comparison.csv", model_rows)

        # Top-k comparison against the restored final index.
        config.CHROMA_PATH = original["CHROMA_PATH"]
        config.COLLECTION_NAME = original["COLLECTION_NAME"]
        config.EMBEDDING_MODEL = original["EMBEDDING_MODEL"]
        topk_rows = []
        for top_k in (1, 3, 5):
            item = summary(evaluate.evaluate(top_k))
            item["top_k"] = top_k
            topk_rows.append(item)
        write_csv(experiment_root / "topk_comparison.csv", topk_rows)
        print("Saved current comparisons to", experiment_root)
    finally:
        for name, value in original.items():
            setattr(config, name, value)


if __name__ == "__main__":
    main()
