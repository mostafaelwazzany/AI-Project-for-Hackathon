"use client";

import { FormEvent, useState } from "react";
import { LoaderCircle, LockKeyhole, ShieldCheck } from "lucide-react";

export default function AnalysisLogin({ configured }: { configured: boolean }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function login(event: FormEvent) {
    event.preventDefault();
    if (!password || loading) return;
    setLoading(true);
    setError("");
    try {
      const response = await fetch("/api/analysis/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      window.location.reload();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="grid min-h-screen place-items-center px-4 py-10">
      <section className="w-full max-w-md rounded-3xl border border-[#203b58] bg-[#0d1b2d]/95 p-6 shadow-2xl sm:p-8">
        <div className="grid size-12 place-items-center rounded-2xl bg-[#153c63] text-[#8dc8ff]">
          <LockKeyhole aria-hidden="true" size={23} />
        </div>
        <p className="mt-6 text-xs font-semibold uppercase tracking-[.18em] text-[#5ba7ff]">
          Private workspace
        </p>
        <h1 className="mt-2 text-2xl font-semibold">Evaluation analysis</h1>
        <p className="mt-2 text-sm leading-6 text-[#8ca6c1]">
          صفحة خاصة لمراجعة أداء الاسترجاع وتشغيل تجارب Top-k دون تغيير إعداد k=5 الأساسي.
        </p>
        {!configured ? (
          <div role="alert" className="mt-6 rounded-xl border border-[#735528] bg-[#342814] p-4 text-sm text-[#f4d28e]">
            Set ANALYSIS_PASSWORD in web/.env.local, then restart the server.
          </div>
        ) : (
          <form onSubmit={login} className="mt-7 space-y-4">
            <div>
              <label htmlFor="analysis-password" className="mb-2 block text-sm font-medium">
                Admin password
              </label>
              <input
                id="analysis-password"
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="current-password"
                className="min-h-12 w-full rounded-xl border border-[#294864] bg-[#091525] px-4 text-base focus:border-[#5ba7ff] focus:outline-none"
              />
            </div>
            {error && <p role="alert" className="text-sm text-[#ffb1bd]">{error}</p>}
            <button
              type="submit"
              disabled={loading || !password}
              className="flex min-h-12 w-full items-center justify-center gap-2 rounded-xl bg-[#1e5a91] font-semibold text-white transition-colors hover:bg-[#276fae] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? <LoaderCircle aria-hidden="true" size={18} className="animate-spin" /> : <ShieldCheck aria-hidden="true" size={18} />}
              {loading ? "Checking…" : "Open private analysis"}
            </button>
          </form>
        )}
      </section>
    </main>
  );
}
