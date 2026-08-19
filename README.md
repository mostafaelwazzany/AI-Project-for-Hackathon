# Colorectal Cancer Guideline Assistant

A bilingual RAG assistant grounded in official NICE colorectal cancer guidance.

## What the project does

1. `ingest.py` reads NICE NG151 and the colorectal section of NG12.
2. It cleans the text and creates structure-aware recursive chunks.
3. `intfloat/multilingual-e5-base` converts chunks into embeddings.
4. Chroma stores the vectors locally.
5. `query.py` retrieves the best five chunks for a question.
6. `generate.py` creates a short grounded answer with a NICE citation.
7. The Next.js website provides the bilingual chat and a private analysis page.

## Main architecture

```text
NICE PDFs
   ↓
ingest.py → clean text → chunks → embeddings → Chroma
                                              ↓
Question → query.py → Top-5 evidence → generate.py → cited answer
                                              ↓
                                      Next.js website
```

## Important files

```text
config.py                 All paths, model names, chunk size and Top-k
ingest.py                 PDF parsing, cleaning, chunking and Chroma indexing
supplementary_sources.py  Extracts the colorectal section from NICE NG12
query_understanding.py    Understands short Arabic/English questions and intent
query.py                  Embedding, retrieval and reranking
generate.py               Grounded answer, citation and safe refusal
evaluate.py               Calculates retrieval metrics without saving report files
web_chat_bridge.py        Keeps the Python model loaded for the website

data/raw/                 Original NICE PDFs
data/processed/           Clean Markdown and page data
data/chunks/chunks.jsonl  Final chunks
data/vector_store/chroma/ Final vector database
data/evaluation/test_questions.csv  The 66 evaluation questions

web/                      Next.js frontend
web/components/chat-panel.tsx       Chat interface
web/components/dashboard.tsx        Private analysis dashboard
web/app/api/chat/route.ts            Chat API bridge
web/app/api/analysis/evaluate/route.ts  Temporary Top-k evaluation API
```

## Current configuration

```text
Embedding model: intfloat/multilingual-e5-base
Vector database: Chroma
Chunking: structure-aware recursive chunking
Chunk size: 450 tokens
Chunk overlap: 80 tokens
Production Top-k: 5
Generation model: qwen/qwen3.6-27b through Groq
```

Temporary Top-k tests on the private analysis page do not change the production
`Top-k = 5`. Evaluation results are shown in the page and are not written to
extra CSV report files.

## Install and run

```powershell
cd "F:\Creativa Hackathon"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_key
```

Create `web/.env.local`:

```env
ANALYSIS_PASSWORD=your_private_password
```

Run the website:

```powershell
cd "F:\Creativa Hackathon\web"
npm install
npm run dev
```

- Chat: `http://localhost:3000`
- Private analysis: `http://localhost:3000/analysis`

## Useful commands

Rebuild the chunks and vector database:

```powershell
python ingest.py
```

Test retrieval:

```powershell
python query.py "What follow-up is needed after surgery?"
```

Run retrieval evaluation in the terminal:

```powershell
python evaluate.py --top-k 5
```

---

# مساعد إرشادات سرطان القولون والمستقيم

المشروع عبارة عن نظام RAG عربي وإنجليزي يعتمد على إرشادات NICE الرسمية.

## النظام يعمل إزاي؟

1. `ingest.py` يقرأ ملفات NICE وينظف النص.
2. يقسم النص باستخدام Structure-aware Recursive Chunking.
3. موديل `multilingual-e5-base` يحول الـChunks إلى Embeddings.
4. Chroma تخزن الـVectors.
5. `query.py` يسترجع أفضل خمس قطع للسؤال.
6. `generate.py` ينشئ إجابة قصيرة مبنية على النص مع مصدر NICE.
7. موقع Next.js يعرض الشات وصفحة التحليل الخاصة.

## الملفات التي تحتاج تعرفها

- `config.py`: إعدادات المشروع كلها.
- `ingest.py`: Parsing وCleaning وChunking وIndexing.
- `query_understanding.py`: يفهم صياغات الأسئلة المختصرة.
- `query.py`: Retrieval وReranking.
- `generate.py`: الإجابة والمصدر والرفض الآمن.
- `evaluate.py`: يحسب Found Rate وPrecision وMAP وMRR.
- `web/`: الموقع وصفحة Analysis.

صفحة Analysis تستطيع تجربة أي قيمة `k` من 1 إلى 20، لكن القيمة الأساسية
للنظام تظل دائمًا `k=5`. النتائج تظهر داخل الصفحة فقط ولا تنشئ ملفات تقارير
إضافية.
