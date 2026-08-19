# Colorectal Cancer RAG | نظام RAG لسرطان القولون والمستقيم

## English

### Overview

This is a Retrieval-Augmented Generation (RAG) system for colorectal cancer using **NICE NG151** for management and the colorectal recommendations in **NICE NG12** for recognition and referral. Each answer is grounded in retrieved guideline evidence and includes a citation.

> Educational project only. It is not a diagnostic tool and does not replace clinical advice.

### One-line pipeline

`NICE PDF → parse & clean → chunks → embeddings → Chroma → retrieve evidence → grounded answer + citation`

### Architecture

```text
NICE PDF
   │
   ▼
ingest.py ──► cleaned Markdown + chunks.jsonl ──► Chroma vector database
                                                    │
User question ──► query understanding ──► multi-query + hybrid retrieval ──┘
                                                │
                                                ▼
                                           generate.py
                              │
                              ▼
                    Groq-hosted Qwen model
                              │
                              ▼
                  Answer + excerpt + citation
```

### Project structure

```text
Creativa Hackathon/
├── config.py                  # All settings in one place
├── ingest.py                  # Build the local knowledge base
├── query.py                   # Test retrieval only
├── query_understanding.py     # Normalize, classify intent and expand queries
├── supplementary_sources.py   # Extract colorectal-only recommendations from NG12
├── intent_tests.py            # Test varied Arabic/English phrasings
├── generate.py                # Run the complete RAG flow
├── evaluate.py                # Measure retrieval quality
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                   # Original NICE PDF
│   ├── processed/             # Clean Markdown and pages.jsonl
│   ├── chunks/                # chunks.jsonl
│   ├── vector_store/chroma/   # Persistent local Chroma database
│   └── evaluation/            # Test questions and reports
│
└── experiments/               # Optional model/chunk comparison scripts
```

### Libraries and why we use them

| Stage | Library | Use in this project | Why |
|---|---|---|---|
| Parsing | `pymupdf4llm` | Converts the PDF to page-level Markdown | Preserves text structure and tables better than plain-text extraction. |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` | Splits recursively by paragraph, line, sentence, then word. |
| Embeddings | `sentence-transformers` | `intfloat/multilingual-e5-base` | Creates multilingual Arabic/English vectors for questions and evidence. |
| Vector database | `chromadb` | Persistent local Chroma collection | Stores vectors with page, section, source, and chunk ID metadata. |
| Prompting | `langchain-core` | `ChatPromptTemplate` | Creates a structured prompt that requires answer, excerpt, and citation. |
| Generation | `langchain-groq` | `ChatGroq` with Qwen | Generates the final answer using retrieved evidence. |
| Secrets | `python-dotenv` | Reads `GROQ_API_KEY` from `.env` | Keeps the API key outside the code. |

**Why not FAISS or LlamaParse?** They are alternatives, not extra requirements. Chroma was selected because it persists vectors and metadata locally. `pymupdf4llm` was selected because it preserves this guideline's tables without another cloud API key. LangChain is used for recursive chunking and prompt construction.

### RAG stages

1. **Parse** the NICE PDF page by page into Markdown.
2. **Clean** conversion noise such as HTML tags, repeated headings, and excess whitespace.
3. **Chunk** the text with structure-aware recursive splitting.
4. **Embed** every chunk using `intfloat/multilingual-e5-base`.
5. **Store** vectors and metadata in Chroma.
6. **Retrieve** the closest evidence chunks for a user question.
7. **Generate** a grounded answer using only those chunks.
8. **Evaluate** the retrieval result against manually defined expected evidence.

Before retrieval, the system normalizes Arabic spelling, classifies the intent
with multilingual E5, creates multiple search formulations, combines semantic
and keyword signals, and reranks the candidates. This handles dialect, formal
Arabic and English without calling another LLM.

### Chunking approach

We use **structure-aware document chunking with LangChain recursive token-based splitting**.

- Chunk size: **450 real E5 tokens**.
- Chunk overlap: **80 tokens**.
- Metadata retained: page, section title, content type, source URL, and table ID.
- Tables are kept intact by row group; rows are not cut in the middle.
- A multi-page table keeps a page-range citation such as `10-12`.

### Setup and commands

Activate the environment and install dependencies once:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then add:

```text
GROQ_API_KEY=your_key_here
```

Build/rebuild the knowledge base after changing the PDF or chunk settings:

```powershell
python ingest.py
```

Test retrieval only:

```powershell
python query.py "What follow-up is recommended after curative colorectal cancer surgery?"
```

Run the complete RAG chat:

```powershell
python generate.py --interactive
```

Evaluate retrieval quality:

```powershell
python evaluate.py --top-k 5
```

لمقارنة أحجام الـchunks والـembedding models والـTop-k:

```powershell
python current_experiments.py
```

لاختبارات الأمان والأسئلة خارج النطاق:

```powershell
python adversarial_tests.py
```

Run the Day 4 safety suite (confidence threshold, citation checks, unsupported
claim detection, and faithfulness proxy):

```powershell
python day4_evaluation.py
```

Run the current Day 2 comparisons (chunk size/overlap, embedding model, and
Top-k) on the same 66-question dataset:

```powershell
python current_experiments.py
```

Run the adversarial safety checks:

```powershell
python adversarial_tests.py
```

Test intent detection across varied Arabic and English phrasings:

```powershell
python intent_tests.py
```

Evaluation files:

- `data/evaluation/evaluation_results.csv`: result for every test question.
- `data/evaluation/evaluation_summary.csv`: overall Found Rate, MAP, and MRR.
- `data/evaluation/day4_safety_results.csv`: per-question Day 4 safety checks.
- `data/evaluation/day4_safety_summary.csv`: Day 4 safety summary.
- `data/evaluation/adversarial_results.csv`: adversarial safety results.
- `data/experiments_current/`: current reproducible Day 2 comparisons.

The Day 4 script accepts `--limit`, `--language ar|en`, and `--out-of-scope` so
cloud checks can be run in small batches when the provider is slow or rate-limited.

Close these CSV files in Excel or VS Code before rerunning evaluation, otherwise Windows may lock the file.

### Evaluation metrics

| Metric | Meaning |
|---|---|
| Found Rate | Percentage of in-scope questions where expected evidence appears in Top-k. |
| Precision@k | Fraction of retrieved results that directly match expected evidence. It can be low at Top-5 even when the correct chunk is found. |
| MAP@k | Average ranking quality across all expected evidence. |
| MRR | How early the first correct evidence appears; higher is better. |

### Regular expressions

Every regular expression in the Python files has a comment beginning with `Regex101:` immediately above it. Copy the expression after that label into [regex101](https://regex101.com/) for a visual explanation.

```python
# Regex101: ^\s*-?\s*\d+\.\d+\.\d+\b
```

The example matches a recommendation number at the beginning of a line, such as `1.6.1` or `- 1.6.1`.

### Presentation questions

**Why multilingual E5?** It supports both Arabic and English retrieval and uses E5 query/passage prefixes.

**Why Chroma?** It stores the vectors together with citation-ready metadata locally.

**Does the LLM answer from its own knowledge?** No. It receives retrieved NICE evidence and is instructed to refuse when the evidence is insufficient.

**What is the difference between `query.py` and `generate.py`?** `query.py` returns evidence chunks only. `generate.py` turns them into a cited answer.

---

## العربية

### نظرة عامة

ده نظام **RAG** لأسئلة سرطان القولون والمستقيم، ويستخدم **NICE NG151** للعلاج والمتابعة، وتوصيات القولون من **NICE NG12** للأعراض والتعرّف والإحالة. كل إجابة مبنية على دليل مسترجع وتظهر معها citation.

> المشروع تعليمي فقط، وليس أداة تشخيص أو بديلًا عن الطبيب.

### الفكرة في سطر واحد

`NICE PDF → parse & clean → chunks → embeddings → Chroma → retrieve evidence → grounded answer + citation`

### هيكل المشروع

```text
Creativa Hackathon/
├── config.py                  # جميع الإعدادات في مكان واحد
├── ingest.py                  # يبني قاعدة المعرفة المحلية
├── query.py                   # يختبر الاسترجاع فقط
├── query_understanding.py     # يوحد السؤال ويفهم الـIntent ويولد صيغ بحث
├── supplementary_sources.py  # يستخرج قسم القولون فقط من NG12
├── intent_tests.py            # يختبر الصياغات العربية والإنجليزية
├── generate.py                # يشغل الـRAG كاملًا
├── evaluate.py                # يقيس جودة الاسترجاع
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                   # ملف NICE PDF الأصلي
│   ├── processed/             # Markdown نظيف وpages.jsonl
│   ├── chunks/                # chunks.jsonl
│   ├── vector_store/chroma/   # قاعدة Chroma المحلية الدائمة
│   └── evaluation/            # أسئلة وتقارير التقييم
│
└── experiments/               # تجارب اختيارية للموديلات والـchunks
```

### المكتبات ولماذا نستخدمها

| المرحلة | المكتبة | استخدامها | السبب |
|---|---|---|---|
| Parsing | `pymupdf4llm` | يحول الـPDF إلى Markdown لكل صفحة | يحافظ على بنية النص والجداول أفضل من استخراج النص العادي. |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` | يقسم بالترتيب: paragraph ثم line ثم sentence ثم word. |
| Embeddings | `sentence-transformers` | `intfloat/multilingual-e5-base` | ينشئ vectors للأسئلة والأدلة بالعربي والإنجليزي. |
| Vector DB | `chromadb` | Chroma local persistent collection | يخزن الـvectors مع الصفحة والقسم والمصدر وchunk ID. |
| Prompting | `langchain-core` | `ChatPromptTemplate` | يبني prompt يفرض الإجابة والنص الداعم والمصدر. |
| Generation | `langchain-groq` | `ChatGroq` مع Qwen | يولد الإجابة النهائية من الأدلة المسترجعة. |
| Secrets | `python-dotenv` | يقرأ `GROQ_API_KEY` من `.env` | يحفظ المفتاح خارج الكود. |

**لماذا ليس FAISS أو LlamaParse؟** هما بدائل وليسا مكتبتين يجب استخدامهما معًا. اخترنا Chroma لأنها تحفظ الـvectors والـmetadata محليًا. واخترنا `pymupdf4llm` لأنه يحافظ على جداول هذا الـPDF ولا يحتاج API key إضافيًا. نستخدم LangChain للـrecursive chunking وبناء الـprompt.

### مراحل الـRAG

1. **Parsing**: نقرأ NICE PDF صفحة بصفحة ونحوله إلى Markdown.
2. **Cleaning**: نحذف HTML tags والعناوين المكررة والمسافات الزائدة.
3. **Chunking**: نقسم النص بطريقة structure-aware recursive.
4. **Embedding**: نحول كل chunk إلى vector باستخدام `intfloat/multilingual-e5-base`.
5. **Store**: نخزن الـvectors والـmetadata في Chroma.
6. **Retrieval**: نرجع أقرب evidence للسؤال.
7. **Generation**: ننتج إجابة مبنية على الـevidence فقط.
8. **Evaluation**: نقارن الاسترجاع بالدليل المتوقع لكل سؤال.

قبل الاسترجاع، النظام يوحد اختلافات الكتابة العربية، ويفهم الـIntent باستخدام
multilingual E5، وينشئ أكثر من صيغة بحث، ثم يجمع semantic similarity مع keyword
matching ويعيد ترتيب النتائج. لذلك يفهم العامية والفصحى والإنجليزية بدون API
إضافي.

### نوع الـChunking

نستخدم:

> **Structure-aware document chunking + LangChain recursive token-based splitting**

- حجم القطعة: **450 E5 token حقيقية**.
- Overlap: **80 token**.
- نحتفظ بـpage وsection وcontent type وsource URL وtable ID.
- الجداول تُقسّم بين مجموعات الصفوف ولا نقطع الصف في منتصفه.
- الجدول متعدد الصفحات يحتفظ بمصدر مثل `10-12`.

### التشغيل

فعّل البيئة وثبّت المكتبات مرة واحدة:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

انسخ `.env.example` إلى `.env` ثم ضع المفتاح:

```text
GROQ_API_KEY=your_key_here
```

لبناء أو إعادة بناء قاعدة المعرفة بعد تغيير الـPDF أو الـchunk settings:

```powershell
python ingest.py
```

لاختبار الـretrieval فقط:

```powershell
python query.py "What follow-up is recommended after curative colorectal cancer surgery?"
```

لتشغيل الـRAG كاملًا:

```powershell
python generate.py --interactive
```

للتقييم:

```powershell
python evaluate.py --top-k 5
```

لاختبار اختلاف صياغة الأسئلة والـIntent المتوقع:

```powershell
python intent_tests.py
```

ملفات النتائج:

- `data/evaluation/evaluation_results.csv`: نتيجة كل سؤال.
- `data/evaluation/evaluation_summary.csv`: الملخص النهائي مثل Found Rate وMAP وMRR.
- `data/evaluation/adversarial_results.csv`: نتائج اختبارات الأمان.
- `data/experiments_current/`: المقارنات الحالية القابلة لإعادة التشغيل.

اقفل ملفات CSV في Excel أو VS Code قبل إعادة التقييم حتى لا يقفلها Windows.

### Metrics التقييم

| Metric | المعنى |
|---|---|
| Found Rate | نسبة الأسئلة التي ظهر دليلها المتوقع في Top-k. |
| Precision@k | نسبة النتائج المسترجعة المطابقة للدليل. قد تكون منخفضة مع Top-5 حتى لو ظهر الدليل الصحيح. |
| MAP@k | متوسط جودة ترتيب الأدلة الصحيحة. |
| MRR | مدى قرب أول دليل صحيح من Rank 1؛ كلما زاد كان أفضل. |

### Regular Expressions

فوق كل Regular Expression في ملفات Python يوجد comment يبدأ بـ`Regex101:`. انسخ التعبير الذي بعده وضعه في [regex101](https://regex101.com/) لرؤية شرحه.

```python
# Regex101: ^\s*-?\s*\d+\.\d+\.\d+\b
```

المثال يلتقط رقم توصية في بداية السطر مثل `1.6.1` أو `- 1.6.1`.

### أسئلة متوقعة في العرض

**لماذا multilingual E5؟** لأنه يدعم الاسترجاع بالعربي والإنجليزي ويستخدم prefixes مناسبة للأسئلة والنصوص.

**لماذا Chroma؟** لأنها تحفظ الـvectors مع metadata جاهزة للـcitation بشكل محلي.

**هل الـLLM يجيب من معلوماته؟** لا. نرسل له أدلة NICE المسترجعة فقط، ونطلب منه الرفض لو الأدلة غير كافية.

**ما الفرق بين `query.py` و`generate.py`؟** `query.py` يعرض الـevidence chunks فقط، بينما `generate.py` يحولها لإجابة مفهومة مع citation.
