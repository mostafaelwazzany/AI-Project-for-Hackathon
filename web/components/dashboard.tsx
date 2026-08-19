"use client";

import { FormEvent, useState } from "react";
import { Activity, BarChart3, CheckCircle2, Database, FlaskConical, Gauge, Languages, LoaderCircle, LockKeyhole, LogOut, Settings2, ShieldCheck } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { chunkData, modelData, overview, topKData } from "../lib/data";

type Summary = { top_k: number; total_questions: number; scored_questions: number; out_of_scope_questions: number; found_expected_evidence: number; found_rate: number; mean_precision_at_k: number; map_at_k: number; mrr: number; ar_found_rate: number; ar_found_count: string; en_found_rate: number; en_found_count: string };
const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

function MetricCard({ label, value, hint, icon: Icon, accent = "blue" }: { label: string; value: string; hint: string; icon: typeof Activity; accent?: "blue" | "green" | "amber" }) {
  const colors = accent === "green" ? "bg-[#0b3a32] text-[#71e5b9]" : accent === "amber" ? "bg-[#3b2a12] text-[#f4c879]" : "bg-[#102e4d] text-[#8dc8ff]";
  return <article className="rounded-2xl border border-[#203b58] bg-[#0d1b2d]/95 p-5"><div className="flex items-start justify-between gap-3"><p className="text-sm text-[#8ca6c1]">{label}</p><span className={`rounded-xl p-2 ${colors}`}><Icon aria-hidden="true" size={18} /></span></div><p className="mt-4 text-3xl font-semibold">{value}</p><p className="mt-2 text-xs leading-5 text-[#8ca6c1]">{hint}</p></article>;
}

function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return <section className="rounded-2xl border border-[#203b58] bg-[#0d1b2d]/95 p-5 sm:p-6"><h2 className="text-lg font-semibold">{title}</h2><p className="mt-1 text-sm leading-6 text-[#8ca6c1]">{description}</p><div className="mt-6">{children}</div></section>;
}

export default function AnalysisDashboard() {
  const [topK, setTopK] = useState(5);
  const [result, setResult] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function evaluate(event: FormEvent) {
    event.preventDefault();
    if (loading) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const response = await fetch("/api/analysis/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ topK }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setResult(data.summary);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Evaluation failed."); }
    finally { setLoading(false); }
  }

  async function logout() { await fetch("/api/analysis/logout", { method: "POST" }); window.location.reload(); }

  return <div className="min-h-screen">
    <header className="sticky top-0 z-20 border-b border-[#203b58]/80 bg-[#07111f]/90 px-4 py-3 backdrop-blur-xl sm:px-8"><div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4"><div className="flex min-w-0 items-center gap-3"><div className="grid size-10 shrink-0 place-items-center rounded-xl bg-[#153c63] text-[#8dc8ff]"><LockKeyhole aria-hidden="true" size={19} /></div><div className="min-w-0"><p className="truncate font-semibold">Private retrieval analysis</p><p className="text-[11px] text-[#8ca6c1]">Admin only · fixed production Top-k = 5</p></div></div><button onClick={logout} className="inline-flex min-h-11 items-center gap-2 rounded-xl border border-[#294864] px-3 text-sm transition-colors hover:bg-[#10243a]"><LogOut aria-hidden="true" size={16} /><span className="hidden sm:inline">Log out</span></button></div></header>

    <main className="mx-auto max-w-[1440px] space-y-6 px-4 py-8 sm:px-8 sm:py-10">
      <section className="flex flex-col justify-between gap-5 md:flex-row md:items-end"><div><div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-[#5ba7ff]"><ShieldCheck aria-hidden="true" size={15} /> Verified baseline</div><h1 className="mt-3 text-3xl font-semibold sm:text-4xl">Complete RAG evaluation</h1><p className="mt-3 max-w-2xl text-sm leading-6 text-[#8ca6c1]">The k=5 baseline, language coverage, model and chunk comparisons, plus a temporary Top-k runner.</p></div><div className="rounded-xl border border-[#2b805f] bg-[#0a2826] px-4 py-3 text-sm text-[#71e5b9]">Production configuration remains k=5</div></section>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="Found rate" value={`${overview.foundRate}%`} hint="58 of 64 in-scope questions" icon={CheckCircle2} accent="green" /><MetricCard label="Precision@5" value={`${overview.precision}%`} hint="Mean strict relevance within five results" icon={Gauge} /><MetricCard label="MAP@5" value={`${overview.map}%`} hint="Average ranking quality" icon={BarChart3} /><MetricCard label="MRR" value={`${overview.mrr}%`} hint="How early the first correct result appears" icon={Activity} accent="amber" /></section>

      <Section title="Temporary Top-k evaluation" description="Choose any k from 1 to 20. The result is displayed here only and never changes config.TOP_K = 5.">
        <form onSubmit={evaluate} className="flex flex-col gap-4 rounded-xl border border-[#294864] bg-[#091525] p-4 sm:flex-row sm:items-end"><div className="flex-1"><label htmlFor="evaluation-k" className="mb-2 block text-sm font-medium">Experimental k value</label><input id="evaluation-k" type="number" min={1} max={20} value={topK} onChange={(event) => setTopK(Number(event.target.value))} className="min-h-12 w-full rounded-xl border border-[#294864] bg-[#0d1b2d] px-4 text-base focus:border-[#5ba7ff] focus:outline-none" /></div><button disabled={loading || topK < 1 || topK > 20} className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-[#1e5a91] px-5 font-semibold text-white transition-colors hover:bg-[#276fae] disabled:cursor-not-allowed disabled:opacity-50">{loading ? <LoaderCircle aria-hidden="true" size={18} className="animate-spin" /> : <FlaskConical aria-hidden="true" size={18} />}{loading ? "Running evaluation…" : `Evaluate k=${topK}`}</button></form>
        {loading && <p role="status" className="mt-3 text-sm text-[#8ca6c1]">Loading the embedding model and scoring all 66 questions. This can take around 20–40 seconds.</p>}
        {error && <p role="alert" className="mt-4 rounded-xl border border-[#6d3544] bg-[#3a1824] p-4 text-sm text-[#ffb1bd]">{error}</p>}
        {result && <div className="mt-5 space-y-4"><div className="flex flex-wrap items-center justify-between gap-3"><h3 className="font-semibold">Temporary result for k={result.top_k}</h3><span className="rounded-full border border-[#735528] bg-[#342814] px-3 py-1 text-xs text-[#f4d28e]">Not saved · baseline still k=5</span></div><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><MetricCard label={`Found rate @${result.top_k}`} value={percent(result.found_rate)} hint={`${result.found_expected_evidence}/${result.scored_questions} found`} icon={CheckCircle2} accent="green" /><MetricCard label={`Precision@${result.top_k}`} value={percent(result.mean_precision_at_k)} hint="Mean strict precision" icon={Gauge} /><MetricCard label={`MAP@${result.top_k}`} value={percent(result.map_at_k)} hint="Average precision by rank" icon={BarChart3} /><MetricCard label="MRR" value={percent(result.mrr)} hint="First correct result rank" icon={Activity} accent="amber" /></div><div className="grid gap-3 sm:grid-cols-2"><LanguageResult label="Arabic" rate={result.ar_found_rate} count={result.ar_found_count} /><LanguageResult label="English" rate={result.en_found_rate} count={result.en_found_count} /></div></div>}
      </Section>

      <div className="grid gap-6 xl:grid-cols-[1.35fr_1fr]">
        <Section title="Top-k comparison" description="Reference results showing the recall/precision trade-off."><div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={topKData}><CartesianGrid stroke="#203b58" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#8ca6c1", fontSize: 12 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tick={{ fill: "#8ca6c1", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(value) => `${value}%`} /><Tooltip contentStyle={{ background: "#0d1b2d", border: "1px solid #203b58", borderRadius: 12 }} /><Legend /><Bar dataKey="found" name="Found rate" fill="#46d6a0" radius={[5, 5, 0, 0]} /><Bar dataKey="precision" name="Precision" fill="#5ba7ff" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div></Section>
        <Section title="Coverage and languages" description="Baseline k=5 evaluation coverage."><div className="space-y-5">{[{ label: "Overall", value: 90.62, count: "58/64" }, { label: "Arabic", value: 87.5, count: "28/32" }, { label: "English", value: 93.75, count: "30/32" }].map((item) => <div key={item.label}><div className="mb-2 flex justify-between text-sm"><span>{item.label}</span><span className="text-[#8ca6c1]">{item.count} · {item.value}%</span></div><div className="h-2 overflow-hidden rounded-full bg-[#1a3049]"><div className="h-full rounded-full bg-[#46d6a0]" style={{ width: `${item.value}%` }} /></div></div>)}<div className="grid grid-cols-3 gap-3 pt-2 text-center"><Count value="66" label="Total" /><Count value="64" label="In scope" /><Count value="2" label="Out of scope" /></div></div></Section>
      </div>

      <div className="grid gap-6 lg:grid-cols-2"><ExperimentTable title="Embedding model experiments" description="Why multilingual-e5-base remains selected." heading="Model" rows={modelData.map((row) => [row.name, `${row.found}%`, `${row.map}%`])} /><ExperimentTable title="Chunk experiments" description="Structure-aware recursive chunks compared by token budget." heading="Size / overlap" rows={chunkData.map((row) => [row.name, `${row.found}%`, `${row.map}%`])} /></div>

      <div className="grid gap-6 lg:grid-cols-2"><Section title="Current production configuration" description="Temporary evaluation runs never change these values."><dl className="grid gap-3 text-sm sm:grid-cols-2">{[["Embedding", "intfloat/multilingual-e5-base"], ["Vector database", "Chroma"], ["Top-k", "5 (fixed)"], ["Chunk size", "450 tokens"], ["Overlap", "80 tokens"], ["Documents", "NICE NG151 + NG12"]].map(([term, value]) => <div key={term} className="rounded-xl bg-[#091525] p-3"><dt className="text-xs text-[#8ca6c1]">{term}</dt><dd className="mt-1 break-words font-medium">{value}</dd></div>)}</dl></Section><Section title="Metric guide" description="Short explanations for presentation and judging."><dl className="space-y-4 text-sm">{[["Found Rate", "Expected evidence appears anywhere in Top-k."], ["Precision@k", "Share of returned chunks that are strictly relevant."], ["MAP@k", "Rewards retrieving all expected evidence and ranking it early."], ["MRR", "Measures how early the first correct chunk appears."]].map(([term, text]) => <div key={term}><dt className="font-semibold text-[#8dc8ff]">{term}</dt><dd className="mt-1 text-[#8ca6c1]">{text}</dd></div>)}</dl></Section></div>

      <div className="flex items-center gap-3 rounded-2xl border border-[#203b58] bg-[#0d1b2d] p-5 text-sm"><Database aria-hidden="true" className="text-[#46d6a0]" size={19} /><div><p className="font-semibold">Evaluation data is local</p><p className="mt-1 text-xs text-[#8ca6c1]">Temporary experiments use the existing Chroma index and 66 test questions without changing production retrieval.</p></div><Settings2 aria-hidden="true" className="ml-auto hidden text-[#5ba7ff] sm:block" size={19} /></div>
    </main>
  </div>;
}

function LanguageResult({ label, rate, count }: { label: string; rate: number; count: string }) { return <div className="rounded-xl border border-[#203b58] bg-[#0a1828] p-4"><p className="flex items-center gap-2 text-sm text-[#8ca6c1]"><Languages aria-hidden="true" size={16} />{label} found rate</p><p className="mt-2 text-2xl font-semibold">{percent(rate)}</p><p className="mt-1 text-xs text-[#8ca6c1]">{count}</p></div>; }
function Count({ value, label }: { value: string; label: string }) { return <div className="rounded-xl bg-[#091525] p-3"><p className="text-xl font-semibold">{value}</p><p className="text-xs text-[#8ca6c1]">{label}</p></div>; }
function ExperimentTable({ title, description, heading, rows }: { title: string; description: string; heading: string; rows: string[][] }) { return <Section title={title} description={description}><div className="overflow-x-auto"><table className="w-full min-w-[420px] text-left text-sm"><thead className="text-[#8ca6c1]"><tr><th className="pb-3 font-medium">{heading}</th><th className="pb-3 font-medium">Found rate</th><th className="pb-3 font-medium">MAP</th></tr></thead><tbody>{rows.map((row) => <tr key={row[0]} className="border-t border-[#203b58]"><td className="py-3 font-medium">{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td></tr>)}</tbody></table></div></Section>; }
