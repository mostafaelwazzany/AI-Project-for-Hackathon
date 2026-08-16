# Day 1 — Embeddings and vector index decision

## Chosen embedding model

`intfloat/multilingual-e5-small`, loaded through the `sentence-transformers` library.

Why:

- The NICE source text is English, while demo questions may be Arabic or English.
- It is a multilingual model that supports Arabic and has an MIT license.
- Unlike a general sentence-similarity model, E5 was trained specifically for query-to-passage retrieval.
- It runs locally, so the MVP does not need an embedding API key or paid service.
- Document chunks use the required `passage: ` prefix with `encode_document`; user questions use `query: ` with `encode_query`.
- Both outputs are normalized and compared using cosine distance.

The first model download is cached locally. Later runs reuse the cached files.

## Chunk strategy

Chunks follow the guideline structure rather than arbitrary fixed windows. Each numbered NICE recommendation is kept as its own chunk. The multi-page early rectal cancer table is split by source page and linked back to recommendation 1.3.3. Every embedded text includes document, heading, and page context before the original content.

The Day 1 MVP indexes the 33 chunks marked `in_initial_scope=true`. The script can later index all recommendation/table chunks or the complete 178-chunk corpus by changing `--scope`.

## Vector database

Chroma is used with a persistent local client and a cosine HNSW index. Each record stores the chunk text, its embedding, and metadata. Required presentation fields are:

- `document_name`
- `section_title`
- `page_number`
- `chunk_id`

Extra metadata preserves pages, recommendation IDs, section code, content type, source URL, and table links for citations and filtering.

## Commands

```powershell
.\.venv\Scripts\python.exe scripts\03_build_vector_index.py
.\.venv\Scripts\python.exe scripts\04_test_retrieval.py
.\.venv\Scripts\python.exe scripts\04_test_retrieval.py --query "What treatment options are recommended for early rectal cancer?"
```

Generated outputs:

- `data/vector_store/chroma/` — persistent vector database
- `data/vector_store/index_report.json` — model, dimension, count, and build provenance
- `data/evaluation/retrieval_report.json` — bilingual retrieval test results
