"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Languages, LoaderCircle } from "lucide-react";
import ChatPanel from "./chat-panel";

type Language = "ar" | "en";

function SiteLogo() {
  return (
    <div className="site-logo-3d grid size-11 shrink-0 place-items-center rounded-2xl border border-[#58d7ff]/35 bg-gradient-to-br from-[#0b3a55] via-[#123c63] to-[#0b3a32] text-[#b9f7e1] shadow-lg shadow-[#07111f]/40">
      <svg
        viewBox="0 0 48 48"
        aria-hidden="true"
        className="size-7"
        fill="none"
      >
        <path
          d="M24 7c-5.5 0-10 4.4-10 9.8 0 3 1.4 5.8 3.7 7.6L13 41h8l1.3-7.2h3.4L27 41h8l-4.7-16.6A9.6 9.6 0 0 0 34 16.8C34 11.4 29.5 7 24 7Z"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinejoin="round"
        />
        <path
          d="M19 18h10M24 13v10"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
        />
        <path
          d="M12 30c-4.5 1-7 3-7 5.5C5 39.1 10.8 42 18 42m18-12c4.5 1 7 3 7 5.5 0 3.6-5.8 6.5-13 6.5"
          stroke="#5ba7ff"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

export default function AssistantApp() {
  const [language, setLanguage] = useState<Language>("en");
  const [ragReady, setRagReady] = useState(false);
  const [warmupSlow, setWarmupSlow] = useState(false);
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const splashTimer = window.setTimeout(() => setShowSplash(false), 2_800);
    const timer = window.setTimeout(() => setWarmupSlow(true), 12_000);
    void fetch("/api/chat")
      .then((response) => {
        if (response.ok) {
          setRagReady(true);
          setWarmupSlow(false);
          window.setTimeout(() => setShowSplash(false), 450);
        }
      })
      .catch(() => setWarmupSlow(true))
      .finally(() => window.clearTimeout(timer));
    return () => {
      window.clearTimeout(timer);
      window.clearTimeout(splashTimer);
    };
  }, []);

  const arabic = language === "ar";

  return (
    <div className="premium-shell min-h-screen overflow-x-hidden">
      <div aria-hidden="true" className="scene-3d fixed inset-0 pointer-events-none">
        <div className="orb orb-blue" />
        <div className="orb orb-green" />
        <div className="orb orb-violet" />
        <div className="mesh-floor" />
        <div className="floating-card floating-card-a" />
        <div className="floating-card floating-card-b" />
      </div>
      {showSplash && (
        <div
          role="status"
          aria-label={arabic ? "جارٍ تجهيز المساعد" : "Preparing assistant"}
          className="fixed inset-0 z-50 grid place-items-center bg-[#07111f]"
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_35%,rgba(91,167,255,.22),transparent_34%),radial-gradient(circle_at_55%_60%,rgba(70,214,160,.14),transparent_30%)]" />
          <div className="relative flex flex-col items-center text-center">
            <div className="splash-logo">
              <SiteLogo />
            </div>
            <h1 className="mt-6 text-xl font-semibold text-[#e7f0fb] sm:text-2xl">
              {arabic
                ? "مساعد سرطان القولون والمستقيم"
                : "Colorectal Cancer Assistant"}
            </h1>
            <p className="mt-2 text-sm text-[#8ca6c1]">
              {ragReady
                ? arabic ? "المساعد جاهز" : "Assistant is ready"
                : arabic ? "نجهز الدليل الطبي الآن…" : "Preparing the guideline assistant…"}
            </p>
            <div className="mt-5 h-1.5 w-48 overflow-hidden rounded-full bg-[#10243a]">
              <div className="splash-progress h-full rounded-full bg-gradient-to-r from-[#5ba7ff] to-[#46d6a0]" />
            </div>
          </div>
        </div>
      )}
      <header className="sticky top-0 z-20 border-b border-[#6fe9ff]/10 bg-[#07111f]/55 px-4 py-3 shadow-2xl shadow-black/20 backdrop-blur-2xl sm:px-8">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <SiteLogo />
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
            className="interactive-lift inline-flex min-h-11 shrink-0 items-center gap-2 rounded-xl border border-[#58d7ff]/25 bg-[#0d1b2d]/75 px-3 text-sm font-medium shadow-lg shadow-black/20 backdrop-blur-xl transition-colors duration-200 hover:bg-[#102d49]"
            aria-label={arabic ? "Switch to English" : "التبديل إلى العربية"}
          >
            <Languages aria-hidden="true" size={17} className="text-[#8dc8ff]" />
            {arabic ? "English" : "العربية"}
          </button>
        </div>
      </header>
      <main className="relative z-10 px-4 py-5 sm:px-8 lg:h-[calc(100vh-4.5rem)] lg:overflow-hidden">
        <ChatPanel language={language} />
      </main>
    </div>
  );
}
