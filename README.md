# Colorectal Cancer RAG

نظام RAG بسيط للإجابة عن أسئلة تخص **Colorectal cancer** بالاعتماد على دليل
**NICE NG151** فقط. كل إجابة مبنية على جزء مسترجع من الدليل وتظهر معه citation.

> هذا المشروع demonstrator تعليمي، وليس أداة تشخيص أو بديلًا عن الطبيب.

## الفكرة في سطر واحد

`PDF guideline → parse & clean → chunks → embeddings → Chroma → retrieve evidence → grounded answer + citation`

## Architecture

```text
NICE PDF
   │
   ▼
ingest.py ──► cleaned Markdown + chunks.jsonl ──► Chroma vector database
                                                    │
User question ──► generate.py ──► retrieval ──────┘
                              │
                              ▼
                Groq-hosted Qwen model
                              │
                              ▼
                 Answer + supporting text + source
```

## Project structure

```text
Creativa Hackathon/
│
├── config.py                  # كل الإعدادات المهمة في مكان واحد
├── ingest.py                  # أمر بسيط لبناء قاعدة المعرفة
├── query.py                   # أمر لتجربة الاسترجاع فقط
├── generate.py                # أمر لتشغيل RAG كاملًا
├── evaluate.py                # أمر لقياس جودة الاسترجاع
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                    # الـPDF الأصلي
│   ├── processed/              # Markdown نظيف وpages.jsonl
│   ├── chunks/                 # chunks.jsonl
│   ├── vector_store/chroma/    # قاعدة الـvector database المحلية
│   └── evaluation/             # الأسئلة وتقارير التقييم
│
└── experiments/               # تجارب اختيارية فقط، ليست مطلوبة للتشغيل
```

## Libraries المستخدمة ولماذا

| المرحلة | المكتبة | استخدامها في المشروع | لماذا اخترناها؟ |
|---|---|---|---|
| Parsing | `pymupdf4llm` | `rag_pipeline/ingestion.py` | يحول الـPDF إلى Markdown ويحافظ على بنية النص والجداول أفضل من استخراج نص عادي. |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` | Recursive chunking: يفضّل الفصل عند الفقرات ثم الأسطر ثم الجمل ثم الكلمات. |
| Embeddings | `sentence-transformers` | `intfloat/multilingual-e5-base` | يفهم العربية والإنجليزية، ويحوّل السؤال والـchunks إلى vectors. |
| Vector DB | `chromadb` | Chroma persistent local database | يخزّن vectors مع metadata مثل الصفحة والقسم وchunk ID، ويعيد أقرب evidence. |
| Prompting | `langchain-core` | `ChatPromptTemplate` | يبني prompt منظم يفرض: recommendation + excerpt + citation. |
| Generation | `langchain-groq` | `ChatGroq` مع Qwen | يستدعي نموذجًا cloud-hosted لإنتاج الإجابة من الأدلة المسترجعة. |
| Secrets | `python-dotenv` | يقرأ `GROQ_API_KEY` من `.env` | لا نكتب الـAPI key داخل الكود. |

### لماذا لا نستخدم FAISS أو LlamaParse الآن؟

هما بدائل، وليس مطلوبًا وضع كل المكتبات في مشروع واحد:

- اخترنا **Chroma** بدل FAISS لأنه يخزّن الـvectors والـmetadata محليًا بشكل دائم، وهو مناسب لعرض المشروع.
- اخترنا **pymupdf4llm** بدل LlamaParse لأنه لا يحتاج API key إضافيًا ويحافظ على جداول هذا الـPDF كـMarkdown.
- ما زلنا نستخدم **LangChain** فعليًا للـchunking والـprompt؛ وهو النمط المطلوب في الـRAG pipeline.

## مراحل الـRAG ببساطة

1. **Parsing**: نقرأ NICE PDF ونحوّله لنص Markdown صفحة بصفحة.
2. **Cleaning**: نحذف وسوم HTML، headings المكررة، المسافات والرموز الناتجة عن التحويل.
3. **Chunking**: نقسم النص إلى قطع لا تزيد عن 450 tokens، مع overlap = 80 tokens. الجداول الطويلة تقسم بين مجموعات الصفوف ولا نقطع الصف نفسه.
4. **Embedding**: نحول كل chunk إلى vector باستخدام `intfloat/multilingual-e5-base`.
5. **Store**: نخزّن الـvector والنص والـmetadata في Chroma.
6. **Retrieval**: نحول سؤال المستخدم إلى vector ونطلب أقرب 5 chunks.
7. **Generation**: نرسل السؤال + الـchunks المسترجعة إلى Qwen ونطلب إجابة لا تعتمد إلا على evidence وتحتوي citation.
8. **Evaluation**: نختبر أسئلة معروفة ونقيس هل ظهر الدليل الصحيح ضمن Top-k أم لا.

## Chunking المستخدم

نستخدم:

> **Structure-aware document chunking + LangChain recursive token-based splitting**

- Document-aware: نحتفظ برقم الصفحة، عنوان القسم، نوع المحتوى، وtable ID.
- Recursive: يفصل بالترتيب `paragraph → line → sentence → word` عند الحاجة.
- Token-aware: حجم الـchunk يقاس بـtokens حقيقية من tokenizer الخاص بـE5، وليس بعدد الحروف.
- Table-safe: الجداول لا تتكسر في منتصف الصف، والجداول عبر أكثر من صفحة تحمل citation مثل `10-12`.

## التشغيل

### 1. أول مرة فقط

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

انسخ `.env.example` إلى `.env` وضع المفتاح:

```text
GROQ_API_KEY=your_key_here
```

### 2. بناء قاعدة المعرفة

نفذه عندما تغيّر الـPDF أو إعدادات الـchunking:

```powershell
python ingest.py
```

ينتج:

- `data/processed/guideline.md`
- `data/chunks/chunks.jsonl`
- `data/vector_store/chroma/`

### 3. تجربة الـretrieval فقط

```powershell
python query.py "What follow-up is recommended after curative colorectal cancer surgery?"
```

### 4. تشغيل RAG كاملًا

```powershell
python generate.py --interactive
```

اكتب السؤال، ثم `exit` للخروج.

### 5. تقييم الجودة

```powershell
python evaluate.py --top-k 5
```

النتائج تخرج في ملفين:

- `data/evaluation/evaluation_results.csv`: نتيجة كل سؤال.
- `data/evaluation/evaluation_summary.csv`: الملخص النهائي مثل Found Rate وMAP وMRR.

> اقفل ملف CSV من Excel أو VS Code قبل تشغيل التقييم مرة أخرى، لأن Windows قد يمنع Python من استبداله وهو مفتوح.

## كيف نفسر Metrics؟

| Metric | المعنى |
|---|---|
| Found Rate | نسبة الأسئلة التي ظهر دليلها الصحيح ضمن Top-k. |
| Precision@k | من النتائج المسترجعة، كم نتيجة مطابقة مباشرة للدليل المتوقع. قد تكون منخفضة مع Top-5 حتى لو الدليل الصحيح موجود. |
| MAP@k | متوسط جودة ترتيب كل الأدلة الصحيحة. أفضل مقياس هنا مع أكثر من نتيجة. |
| MRR | مدى قرب أول evidence صحيح من Rank 1. |

## Regular Expressions

فوق كل Regex في ملفات المشروع يوجد comment يبدأ بـ`Regex101:`. انسخ ما بعده وضعه في [regex101](https://regex101.com/) لرؤية شرح مرئي.

مثال:

```python
# Regex101: ^\s*-?\s*\d+\.\d+\.\d+\b
```

هذا يلتقط بداية توصية مثل `1.6.1` أو `- 1.6.1`.

## أسئلة متوقعة في العرض

**لماذا E5؟**
لأنه multilingual فيفهم الأسئلة العربية والإنجليزية، ويستخدم prefixes مناسبة: `query:` للسؤال و`passage:` للنص.

**لماذا Chroma؟**
لأنها vector database محلية persistent وتحتفظ بالنص مع source/page/chunk ID، ولذلك نستطيع إظهار citation.

**هل الـLLM يعرف الإجابة من نفسه؟**
لا. نرسل له فقط الأدلة المسترجعة من NICE NG151 ونطلب عدم اختراع معلومات. إذا لم نجد evidence كافيًا نرجع refusal.

**ما الفرق بين `query.py` و`generate.py`؟**
`query.py` يعرض الـchunks فقط. `generate.py` يستخدم الـchunks نفسها لإنتاج إجابة مفهومة مع citation.
