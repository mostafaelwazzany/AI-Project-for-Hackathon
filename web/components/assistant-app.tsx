"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Languages, LoaderCircle, Sparkles } from "lucide-react";
import ChatPanel from "./chat-panel";

type Language = "ar" | "en";

export default function AssistantApp() {
  const [language, setLanguage] = useState<Language>("ar");
  const [ragReady, setRagReady] = useState(false);
  const [warmupSlow, setWarmupSlow] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setWarmupSlow(true), 12_000);
    void fetch("/api/chat")
      .then((response) => {
        if (response.ok) {
          setRagReady(true);
          setWarmupSlow(false);
        }
      })
      .catch(() => setWarmupSlow(true))
      .finally(() => window.clearTimeout(timer));
    return () => window.clearTimeout(timer);
  }, []);

  const arabic = language === "ar";

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-[#203b58]/80 bg-[#07111f]/90 px-4 py-3 backdrop-blur-xl sm:px-8">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-10 shrink-0 place-items-center rounded-xl bg-[#1e5a91] text-[#b9ddff]">
              <Sparkles aria-hidden="true" size={20} />
            </div>
            <div className="min-w-0">
              <p className="text-xs font-semibold leading-4 sm:text-base sm:leading-5">
                {arabic
                  ? "مساعد سرطان القولون والمستقيم"
                  : "Colorectal Cancer Assistant"}
              </p>
              <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-[#8ca6c1]">
                {ragReady ? (
                  <CheckCircle2 aria-hidden="true" size={12} className="text-[#46d6a0]" />
                ) : (
                  <LoaderCircle aria-hidden="true" size={12} className="animate-spin text-[#5ba7ff]" />
                )}
                {ragReady
                  ? arabic ? "جاهز" : "Ready"
                  : warmupSlow
                    ? arabic ? "جاهز للأسئلة، أول رد قد يستغرق قليلًا" : "Ready to ask; first answer may take a moment"
                    : arabic ? "جارٍ تجهيز المساعد…" : "Preparing assistant…"}
              </div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setLanguage(arabic ? "en" : "ar")}
            className="inline-flex min-h-11 shrink-0 items-center gap-2 rounded-xl border border-[#294864] bg-[#0d1b2d] px-3 text-sm font-medium transition-colors duration-200 hover:bg-[#102d49]"
            aria-label={arabic ? "Switch to English" : "التبديل إلى العربية"}
          >
            <Languages aria-hidden="true" size={17} className="text-[#8dc8ff]" />
            {arabic ? "English" : "العربية"}
          </button>
        </div>
      </header>
      <main className="px-4 py-7 sm:px-8 sm:py-10">
        <ChatPanel language={language} />
      </main>
    </div>
  );
}
