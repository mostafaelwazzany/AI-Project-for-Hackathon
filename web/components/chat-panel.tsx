"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Bot,
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  FileText,
  MessageCircle,
  Send,
  User,
} from "lucide-react";

type Source = {
  url: string;
  text: string;
  document: string;
  page?: string;
  section?: string;
  chunk_id?: string;
};
type Message = {
  id: number;
  role: "user" | "assistant";
  text: string;
  source?: Source | null;
  sources?: Source[];
};

type AnswerParts = {
  recommendation: string;
  excerpt: string;
  citation: string;
  disclaimer: string;
};

function parseAnswer(text: string): AnswerParts | null {
  const disclaimerStart = text.search(/\n\s*(?:تنبيه:|Disclaimer:)/);
  const body = disclaimerStart >= 0 ? text.slice(0, disclaimerStart) : text;
  const disclaimer = disclaimerStart >= 0 ? text.slice(disclaimerStart).trim() : "";
  const arabic = body.trimStart().startsWith("التوصية:");
  const pattern = arabic
    ? /^التوصية:\s*([\s\S]*?)\n\s*النص الداعم:\s*([\s\S]*?)\n\s*المصدر:\s*([\s\S]*)$/
    : /^Recommendation:\s*([\s\S]*?)\n\s*Excerpt:\s*([\s\S]*?)\n\s*Citation:\s*([\s\S]*)$/;
  const match = body.trim().match(pattern);
  if (!match) return null;
  return {
    recommendation: match[1].trim(),
    excerpt: match[2].trim(),
    citation: match[3].trim(),
    disclaimer,
  };
}

function citationGuideline(citation: string) {
  return citation.includes("NG12") ? "ng12" : "ng151";
}

function citationPage(citation: string) {
  return citation.match(/(?:الصفحة|Page):\s*([^\];\n]+)/)?.[1]?.trim();
}

function recommendationNumber(text: string) {
  return text.match(/\b(\d+\.\d+\.\d+)\b/)?.[1] ?? "";
}

function sourceMatchesCitation(source: Source, citation: string) {
  const guideline = citationGuideline(citation);
  const sourceGuideline = source.document.toLowerCase().includes("suspected")
    ? "ng12"
    : "ng151";
  if (sourceGuideline !== guideline) return false;

  const page = citationPage(citation);
  if (page && source.page && page === source.page) return true;

  const citedRecommendation = recommendationNumber(citation);
  if (citedRecommendation && source.text.includes(citedRecommendation)) return true;

  return false;
}

function selectedSource(citation: string, sources?: Source[], fallback?: Source | null) {
  return sources?.find((source) => sourceMatchesCitation(source, citation))
    ?? fallback
    ?? null;
}

function niceArticleUrl(citation: string, source?: Source | null) {
  if (source?.url?.includes("/chapter/")) return source.url;

  const guideline = citationGuideline(citation);
  return guideline === "ng12"
    ? "https://www.nice.org.uk/guidance/ng12/chapter/Recommendations-organised-by-site-of-cancer"
    : "https://www.nice.org.uk/guidance/ng151/chapter/Recommendations";
}

function sourceUrl(citation: string, source?: Source | null) {
  const guideline = citationGuideline(citation);
  const article = niceArticleUrl(citation, source);

  if (!source?.text) return article;
  const numberFromCitation = recommendationNumber(citation);
  const numberFromSource = recommendationNumber(source.text);
  const number = numberFromCitation || numberFromSource;
  let cleanText = source.text
    .replace(/<[^>]+>/g, " ")
    .replace(/[#*_`]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const recommendationStart = cleanText.match(/\b\d+\.\d+\.\d+\s+-?\s*(.*)/);
  if (recommendationStart?.[1]) cleanText = recommendationStart[1].trim();
  if (!cleanText) return article;

  // NICE gives every recommendation a stable HTML id such as ng151-1_6_1.
  // The anchor guarantees the correct location; a short Text Fragment adds
  // native browser highlighting without failing on small PDF/HTML differences.
  const anchor = number
    ? `${guideline}-${number.replaceAll(".", "_")}`
    : "";
  const highlight = cleanText.split(/\s+/).slice(0, 10).join(" ");
  const fragment = anchor ? `${anchor}:~:text=` : ":~:text=";
  return `${article}#${fragment}${encodeURIComponent(highlight)}`;
}

function AssistantAnswer({
  text,
  source,
  sources,
}: {
  text: string;
  source?: Source | null;
  sources?: Source[];
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const answer = parseAnswer(text);

  if (!answer) return <div className="whitespace-pre-wrap">{text}</div>;

  const noSource = /لا يوجد مصدر|No citation/.test(answer.citation);
  const arabic = /[\u0600-\u06ff]/.test(answer.recommendation);
  const matchedSource = selectedSource(answer.citation, sources, source);

  return (
    <div dir={arabic ? "rtl" : "ltr"} className="space-y-3">
      <p className="whitespace-pre-wrap text-start">{answer.recommendation}</p>

      {!noSource && (
        <div className="flex justify-start">
          <button
            type="button"
            onClick={() => setShowEvidence((current) => !current)}
            aria-expanded={showEvidence}
            className="inline-flex min-h-11 cursor-pointer items-center gap-1.5 rounded-full border border-[#3b536c] bg-[#111b28] px-3 text-xs font-medium text-[#d6e3f1] shadow-sm transition-colors duration-200 hover:border-[#5ba7ff] hover:bg-[#172638] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#5ba7ff]"
          >
            <span className="grid size-5 place-items-center rounded-full bg-[#25384c] text-[10px]">
              1
            </span>
            <span>NICE source</span>
            <ChevronDown
              aria-hidden="true"
              size={14}
              className={`transition-transform duration-200 ${showEvidence ? "rotate-180" : ""}`}
            />
          </button>
        </div>
      )}

      {showEvidence && !noSource && (
        <aside
          className="rounded-xl border border-[#294864] bg-[#091a2b] p-4 text-start shadow-lg"
          aria-label={arabic ? "تفاصيل المصدر والنص الداعم" : "Source and supporting evidence"}
        >
          <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-[#8dc8ff]">
            <FileText aria-hidden="true" size={15} />
            {arabic ? "الدليل الداعم" : "Supporting evidence"}
          </div>
          <mark
            dir="auto"
            className="block rounded-lg border-s-2 border-[#46d6a0] bg-[#12332f] px-3 py-2 text-start text-inherit [color:inherit]"
          >
            {answer.excerpt}
          </mark>
          <p className="mt-3 text-xs leading-6 text-[#a9bdd1]">
            {answer.citation.replace(/^\[|\]$/g, "")}
          </p>
          <a
            href={sourceUrl(answer.citation, matchedSource)}
            target="_blank"
            rel="noreferrer"
            className="mt-3 inline-flex min-h-11 items-center gap-2 rounded-lg border border-[#315a7e] px-3 text-xs font-semibold text-[#8dc8ff] transition-colors duration-200 hover:bg-[#102d49]"
          >
            <ExternalLink aria-hidden="true" size={15} />
            {arabic
              ? "فتح NICE وتحديد مكان المعلومة"
              : "Open NICE and highlight this evidence"}
          </a>
        </aside>
      )}

      {answer.disclaimer && (
        <p className="border-t border-[#1d5045] pt-3 text-start text-xs leading-6 text-[#9db8b0]">
          {answer.disclaimer}
        </p>
      )}
    </div>
  );
}

export default function ChatPanel({ language = "ar" }: { language?: "ar" | "en" }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const arabicUi = language === "ar";

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function ask(event: FormEvent) {
    event.preventDefault();
    const value = question.trim();
    if (!value || loading) return;
    setMessages((current) => [
      ...current,
      { id: Date.now(), role: "user", text: value },
    ]);
    setQuestion("");
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: data.answer,
          source: data.source,
          sources: data.sources ?? [],
        },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="mx-auto flex max-w-5xl flex-col">
      <div className="mb-6">
        <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[.18em] text-[#5ba7ff]">
          <span className="size-2 rounded-full bg-[#46d6a0]" />
          {arabicUi ? "مساعد مبني على الدليل" : "Guideline assistant"}
        </div>
        <h1 className="text-3xl font-semibold sm:text-4xl">
          {arabicUi ? "اسأل عن سرطان القولون والمستقيم" : "Ask about colorectal cancer"}
        </h1>
        <p className="mt-2 text-sm text-[#8ca6c1]">
          {arabicUi
            ? "اسأل بالعربية أو الإنجليزية، وستكون الإجابة من إرشادات NICE فقط."
            : "Ask in Arabic or English. Answers are grounded only in NICE guidance."}
        </p>
      </div>
      <div className="flex min-h-[62vh] flex-col overflow-hidden rounded-2xl border border-[#203b58] bg-[#0d1b2d]/90">
        <div
          className="flex-1 space-y-5 overflow-y-auto p-4 sm:p-6"
          aria-live="polite"
        >
          {messages.length === 0 && (
            <div className="grid min-h-72 place-items-center text-center">
              <div>
                <div className="mx-auto grid size-14 place-items-center rounded-2xl bg-[#153c63] text-[#8dc8ff]">
                  <Bot size={26} />
                </div>
                <h2 className="mt-4 font-semibold">
                  {arabicUi
                    ? "ابدأ بسؤال عن سرطان القولون والمستقيم"
                    : "Start with a colorectal cancer question"}
                </h2>
                <p className="mt-2 text-sm text-[#8ca6c1]">
                  {arabicUi
                    ? "مثال: ما المتابعة المطلوبة بعد الجراحة العلاجية؟"
                    : "Example: What follow-up is needed after curative surgery?"}
                </p>
              </div>
            </div>
          )}
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`mt-1 grid size-8 shrink-0 place-items-center rounded-full ${message.role === "user" ? "order-2 bg-[#1e5a91]" : "bg-[#0b3a32] text-[#71e5b9]"}`}
              >
                {message.role === "user" ? (
                  <User size={15} />
                ) : (
                  <Bot size={16} />
                )}
              </div>
              <div
                dir="auto"
                className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-start text-sm leading-7 ${message.role === "user" ? "rounded-tr-sm bg-[#1e5a91] text-white" : "rounded-tl-sm border border-[#2b805f] bg-[#0a2826]"}`}
              >
                {message.role === "assistant" ? (
                  <AssistantAnswer
                    text={message.text}
                    source={message.source}
                    sources={message.sources}
                  />
                ) : (
                  message.text
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex gap-3">
              <div className="grid size-8 shrink-0 place-items-center rounded-full bg-[#0b3a32] text-[#71e5b9]">
                <Bot size={16} />
              </div>
              <div
                role="status"
                className="rounded-2xl rounded-tl-sm border border-[#203b58] bg-[#10243a] px-4 py-3"
              >
                <span className="sr-only">
                  {arabicUi ? "جارٍ البحث في الدليل" : "Searching the guideline"}
                </span>
                <div className="flex gap-1.5">
                  <span className="size-2 animate-bounce rounded-full bg-[#5ba7ff] [animation-delay:-.3s]" />
                  <span className="size-2 animate-bounce rounded-full bg-[#5ba7ff] [animation-delay:-.15s]" />
                  <span className="size-2 animate-bounce rounded-full bg-[#5ba7ff]" />
                </div>
                <p className="mt-2 text-xs text-[#8ca6c1]">
                  {arabicUi ? "جارٍ البحث في الدليل…" : "Searching the guideline…"}
                </p>
              </div>
            </div>
          )}
          {error && (
            <div
              role="alert"
              className="rounded-xl border border-[#6d3544] bg-[#3a1824] p-4 text-sm text-[#ffb1bd]"
            >
              {error}
            </div>
          )}
          <div ref={endRef} />
        </div>
        <form
          onSubmit={ask}
          className="border-t border-[#203b58] bg-[#091525] p-3 sm:p-4"
        >
          <div className="flex items-end gap-2">
            <label htmlFor="chat-question" className="sr-only">
              {arabicUi ? "سؤالك" : "Your question"}
            </label>
            <textarea
              id="chat-question"
              dir="auto"
              rows={1}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder={arabicUi ? "اكتب سؤالك هنا…" : "Write your question here…"}
              className="max-h-36 min-h-12 flex-1 resize-none rounded-xl border border-[#294864] bg-[#0d1b2d] px-4 py-3 text-sm text-[#e7f0fb] placeholder:text-[#66809a] focus:border-[#5ba7ff] focus:outline-none"
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className="grid size-12 shrink-0 place-items-center rounded-xl bg-[#1e5a91] text-white hover:bg-[#276fae] disabled:cursor-not-allowed disabled:opacity-40"
              aria-label={arabicUi ? "إرسال السؤال" : "Send question"}
            >
              {loading ? (
                <MessageCircle className="animate-pulse" size={19} />
              ) : (
                <Send size={19} />
              )}
            </button>
          </div>
          <div className="mt-2 flex items-center gap-2 text-[11px] text-[#8ca6c1]">
            <CheckCircle2 size={12} className="text-[#46d6a0]" />
            {arabicUi
              ? "Enter للإرسال، وShift + Enter لسطر جديد"
              : "Press Enter to send, or Shift + Enter for a new line"}
          </div>
        </form>
      </div>
    </section>
  );
}
