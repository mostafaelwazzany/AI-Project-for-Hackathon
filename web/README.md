# NICE RAG Evaluation Console

## English

This Next.js + TypeScript + Tailwind dashboard presents the retrieval evaluation
results from the Python RAG pipeline. It is intentionally a read-only analytics
view: the values come from the current evaluation artifacts and can later be
replaced by an API endpoint.

### Run

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

The **Chat** page keeps the conversation visible as message bubbles and shows a
loading bubble while retrieval/generation is running. `app/api/chat/route.ts`
keeps `web_chat_bridge.py` alive, so the embedding model loads only on the first
question and is reused afterwards. The Python RAG pipeline remains the source of
truth for the answer, excerpt, citation, and disclaimer.

### Design direction

- Dark clinical/financial analytics dashboard with navy surfaces and blue data
  accents; green means healthy/grounded and amber marks trade-offs.
- Fira Sans for UI text and Fira Code for technical configuration values.
- Mobile-first responsive layout: drawer navigation below 1024px, stacked cards,
  and touch targets of at least 44px.
- Charts include legends and text summaries so meaning is not communicated by
  color alone.

### Component choices

The component search was done through 21st.dev for analytics KPI/stat cards and
through the shadcn registry for Card, Tabs, Progress, and chart patterns. The
final components are small local TypeScript components so the team can explain
them easily and avoid paid component retrieval.

## العربية

هذه واجهة Next.js + TypeScript + Tailwind لعرض نتائج تقييم نظام الـRAG. هي شاشة
تحليل للقراءة فقط، والقيم الحالية مأخوذة من ملفات التقييم ويمكن لاحقًا استبدالها
بـAPI.

التشغيل:

```powershell
cd web
npm install
npm run dev
```

ثم افتح `http://localhost:3000`.

صفحة **Chat** تعرض الأسئلة والإجابات كبابلز، وتعرض Loading أثناء البحث. الـAPI
يبقي عملية Python مفتوحة في الخلفية، ولذلك يتم تحميل موديل الـembedding في أول
سؤال فقط ثم يُعاد استخدامه في الأسئلة التالية.
