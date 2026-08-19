# Colorectal Cancer Guideline Assistant

A bilingual (Arabic/English) Retrieval-Augmented Generation system grounded in official NICE colorectal cancer guidance. Ask natural-language questions and receive short, citation-bound answers with source references.

---

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Notebooks](#notebooks)
- [Configuration](#configuration)
- [Setup](#setup)
- [Running the project](#running-the-project)
- [Evaluation results](#evaluation-results)
- [Architecture decisions](#architecture-decisions)

---

## How it works

```
                         INGESTION
                         =========
NICE NG151 PDF  -->  PDF Loader  -->  Clean Markdown  -->  Structure-Aware Chunking
NICE NG12 PDF   -->  (supplementary colorectal section)         |
                                                                v
                                                    Token Count Validation (450 tokens)
                                                                |
                                                                v
                                                    Embedding (intfloat/multilingual-e5-base)
                                                                |
                                                                v
                                                    Chroma Vector Store (cosine HNSW)

                         RETRIEVAL
                         =========
User Question  -->  Query Understanding
                     |  - Normalize (Arabic/English)
                     |  - Classify intent (cue match or semantic)
                     |  - Expand domain context
                     |  - Generate multiple queries
                     v
              Multi-Query Encoding  -->  Chroma Search  -->  Dedup + Rerank
                                                                 |
                                                            Top-5 Results

                         GENERATION
                         ==========
Top-5 Results  -->  Build Context (PASSAGEs with citations)
                         |
                         v
                  System Prompt (citation-bound, calibrated, concise)
                         |
                         v
                  LLM Call (Groq: qwen3.6-27b, temperature=0.2)
                         |
                         v
                  Citation Validation  -->  Claim Support Check  -->  Grounded Answer
```

---

## Project structure

```
.
├── src/rag_app/                        # Main Python package
│   ├── config.py                       # All paths, model names, chunk size, Top-k
│   ├── ingestion/
│   │   ├── pdf_loader.py               # PDF loading, markdown cleaning, NG12 extraction
│   │   ├── chunker.py                  # Structure-aware recursive chunking
│   │   └── indexer.py                  # Embedding and Chroma indexing
│   ├── retrieval/
│   │   ├── query_understanding.py      # Bilingual intent classification and query reformulation
│   │   └── search.py                   # Multi-search retrieval with reranking
│   ├── generation/
│   │   ├── prompt_builder.py           # System prompt and context construction
│   │   ├── citation.py                 # Bilingual citation formatting
│   │   └── generator.py               # Grounded generation pipeline
│   ├── evaluation/
│   │   ├── metrics.py                  # Relevance matching, AP@k, MRR
│   │   └── evaluator.py               # Batch retrieval evaluation
│   └── utils/
│       ├── text.py                     # Arabic detection and terminal display
│       └── io.py                       # JSONL file I/O
│
├── notebooks/                          # Jupyter notebooks (guided walkthroughs)
│   ├── 01_data_ingestion.ipynb         # PDF -> chunks -> Chroma pipeline
│   ├── 02_query_understanding.ipynb    # Bilingual intent classification demo
│   ├── 03_retrieval_reranking.ipynb    # Search + reranking + score analysis
│   ├── 04_generation.ipynb             # Grounded answer generation demo
│   ├── 05_evaluation.ipynb             # Retrieval metrics + experiments
│   └── 06_bilingual_architecture.ipynb # End-to-end bilingual walkthrough
│
├── data/
│   ├── raw/                            # Original NICE PDFs
│   ├── processed/                      # Clean Markdown and page data
│   ├── chunks/chunks.jsonl             # Final chunk records
│   ├── vector_store/chroma/            # Persistent Chroma database
│   └── evaluation/test_questions.csv   # 66 bilingual test questions
│
├── web/                                # Next.js frontend (untouched)
├── run_ingest.py                       # CLI: rebuild chunks and vector database
├── run_query.py                        # CLI: search the index
├── run_generate.py                     # CLI: grounded generation with citations
├── run_evaluate.py                     # CLI: evaluate retrieval metrics
├── web_chat_bridge.py                  # Python-Next.js bridge (stdin/stdout JSON)
├── requirements.txt                    # Python dependencies
└── .env.example                        # Environment template
```

---

## Notebooks

The `notebooks/` directory contains 6 guided walkthroughs. Each notebook:

- Has a clear title and overview
- Explains every step before running it
- Shows intermediate outputs for understanding
- Documents design decisions and rationale
- Requires no external explanation to follow

| # | Notebook | What it covers |
|---|----------|---------------|
| 01 | Data Ingestion | PDF loading, markdown cleaning, chunk creation, token analysis, Chroma indexing |
| 02 | Query Understanding | Arabic normalization, intent classification (cue + semantic), domain expansion |
| 03 | Retrieval & Reranking | Multi-query encoding, Chroma search, deduplication, lexical/intent boosts |
| 04 | Generation | Context building, system prompt, LLM call, citation validation, claim support |
| 05 | Evaluation | Test questions, relevance matching, MAP/MRR, per-language breakdown, experiments |
| 06 | Bilingual Architecture | Same question in AR vs EN, side-by-side comparison, architecture summary |

To run a notebook:

```powershell
cd "path\to\project"
jupyter notebook notebooks/01_data_ingestion.ipynb
```

---

## Configuration

All settings are in `src/rag_app/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding model | `intfloat/multilingual-e5-base` | Multilingual embedding (Arabic + English) |
| Chunk size | 450 tokens | Under E5's 512-token limit |
| Chunk overlap | 80 tokens | Context preservation across chunks |
| Top-k | 5 | Number of retrieved chunks |
| Generation model | `qwen/qwen3.6-27b` | Via Groq free API |
| Temperature | 0.2 | Deterministic for medical answers |
| MIN_RETRIEVAL_SCORE | 0.75 | Refusal threshold for weak evidence |

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the web frontend)
- A Groq API key (free at https://console.groq.com)

### Python environment

```powershell
cd "path\to\project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Web frontend (optional)

```powershell
cd web
npm install
```

Create `web/.env.local`:

```env
ANALYSIS_PASSWORD=your_private_password
```

---

## Running the project

### Ingest data (build chunks and vector index)

```powershell
python run_ingest.py
```

### Search the index

```powershell
python run_query.py "What follow-up is recommended after surgery?"
```

### Generate grounded answers

```powershell
python run_generate.py "What follow-up is recommended after surgery?"
python run_generate.py --interactive
```

### Evaluate retrieval

```powershell
python run_evaluate.py --top-k 5
```

### Run the web interface

```powershell
cd web
npm run dev
```

- Chat: http://localhost:3000
- Analysis: http://localhost:3000/analysis

---

## Evaluation results

Current baseline with production settings (k=5, e5-base, chunk 450/80):

| Metric | Value |
|--------|-------|
| Found rate | 90.6% |
| MAP@5 | 67.0% |
| MRR | 69.0% |
| Arabic found rate | ~90% |
| English found rate | ~91% |

---

## Architecture decisions

1. **Multilingual E5 embeddings** - Shared vector space for Arabic and English, no translation needed
2. **Structure-aware chunking** - Splits at NICE recommendation boundaries (`1.x.x` numbers) for clean citations
3. **Multi-query retrieval** - Each question generates 4-5 query formulations for robust matching
4. **Three-signal reranking** - Semantic similarity + lexical overlap + intent-based boosts
5. **Cross-language claim check** - Arabic answers verified via E5 semantic similarity (threshold 0.72)
6. **Citation validation** - Every citation must match metadata exactly; invented sources are rejected
7. **Safe refusal** - System refuses when evidence is insufficient rather than guessing

---

# مساعد إرشادات سرطان القولون والمستقيم

نظام RAG عربي وإنجليزي يعتمد على إرشادات NICE الرسمية لسرطان القولون والمستقيم. اطرح أسئلة بلغتك واحصل على إجابة قصيرة مع مصدر موثق.

## كيف يعمل النظام؟

1. `run_ingest.py` يقرأ ملفات NICE ويقسم النص إلى chunks معرفية
2. موديل `multilingual-e5-base` يحول الـ chunks إلى embeddings
3. Chroma تخزن المتجهاتlocally
4. `run_query.py` يفهم السؤال ويسترجع أفضل 5 قطع
5. `run_generate.py` ينشئ إجابة قصيرة مع مصدر NICE
6. موقع Next.js يعرض الشات وصفحة التحليل

## الملفات الرئيسية

- `src/rag_app/config.py`: إعدادات المشروع كلها
- `src/rag_app/ingestion/`: Parsing وCleaning وChunking وIndexing
- `src/rag_app/retrieval/`: فهم الأسئلة والاسترجاع والترتيب
- `src/rag_app/generation/`: الإجابة والمصدر والرفض الآمن
- `src/rag_app/evaluation/`: حساب Found Rate وPrecision وMAP وMRR
- `notebooks/`: 6 د Notebooks تشرح كل خطوة بالتفصيل
- `web/`: الموقع وصفحة Analysis

## أوامر مفيدة

```powershell
python run_ingest.py                    # بناء الـ chunks وقاعدة البيانات
python run_query.py "سؤالك"             # البحث في الفهرس
python run_generate.py "سؤالك"          # إجابة مبنية على الدليل
python run_evaluate.py --top-k 5        # تقييم جودة الاسترجاع
```

صفحة Analysis تجرب أي قيمة k من 1 إلى 20، لكن القيمة الأساسية للنظام تظل `k=5`.
