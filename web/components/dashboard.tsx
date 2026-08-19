"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Database,
  FlaskConical,
  Gauge,
  LoaderCircle,
  LockKeyhole,
  LogOut,
  Search,
  ShieldCheck,
  Target,
  XCircle,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { chunkData, modelData, overview, topKData } from "../lib/data";

type Summary = {
  top_k: number;
  total_questions: number;
  scored_questions: number;
  out_of_scope_questions: number;
  found_expected_evidence: number;
  found_rate: number;
  mean_precision_at_k: number;
  map_at_k: number;
  mrr: number;
  ar_found_rate: number;
  ar_found_count: string;
  en_found_rate: number;
  en_found_count: string;
  rank_1_count: number;
  top_3_count: number;
  top_5_count: number;
  missed_count: number;
  average_first_correct_rank: number;
};

type EvaluationRow = {
  id: string;
  variant: string;
  language: string;
  question: string;
  expected_source: string;
  expected_recommendations: string;
  top_k: number;
  status: string;
  found: string;
  best_rank: number | "";
  relevant_in_top_k: number | "";
  precision_at_k: number | "";
  average_precision_at_k: number | "";
  reciprocal_rank: number | "";
  top_score: number;
  top_chunk_id: string;
  top_page: string;
};

type EvaluationReport = {
  summary: Summary;
  rank_distribution: { name: string; count: number; rate: number }[];
  rows: EvaluationRow[];
};

const percent = (value: number) => `${(value * 100).toFixed(2)}%`;

const fallbackSummary: Summary = {
  top_k: 5,
  total_questions: overview.questions,
  scored_questions: overview.scored,
  out_of_scope_questions: overview.questions - overview.scored,
  found_expected_evidence: 64,
  found_rate: overview.foundRate / 100,
  mean_precision_at_k: overview.precision / 100,
  map_at_k: overview.map / 100,
  mrr: overview.mrr / 100,
  ar_found_rate: 1,
  ar_found_count: "32/32",
  en_found_rate: 1,
  en_found_count: "32/32",
  rank_1_count: 44,
  top_3_count: 61,
  top_5_count: 64,
  missed_count: 0,
  average_first_correct_rank: 1.56,
};

const fallbackReport: EvaluationReport = {
  summary: fallbackSummary,
  rank_distribution: [
    { name: "Rank 1", count: fallbackSummary.rank_1_count, rate: fallbackSummary.rank_1_count / fallbackSummary.scored_questions },
    { name: "Top 3", count: fallbackSummary.top_3_count, rate: fallbackSummary.top_3_count / fallbackSummary.scored_questions },
    { name: "Top 5", count: fallbackSummary.top_5_count, rate: fallbackSummary.top_5_count / fallbackSummary.scored_questions },
    { name: "Missed", count: fallbackSummary.missed_count, rate: fallbackSummary.missed_count / fallbackSummary.scored_questions },
  ],
  rows: [],
};

function isEvaluationReport(data: EvaluationReport | { error?: string }): data is EvaluationReport {
  return "summary" in data && "rows" in data && "rank_distribution" in data;
}

export default function AnalysisDashboard() {
  const [topK, setTopK] = useState(5);
  const [baseline, setBaseline] = useState<EvaluationReport | null>(null);
  const [result, setResult] = useState<EvaluationReport | null>(null);
  const [loadingBaseline, setLoadingBaseline] = useState(true);
  const [loadingExperiment, setLoadingExperiment] = useState(false);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [languageFilter, setLanguageFilter] = useState("all");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    void runEvaluation(5, "baseline");
  }, []);

  async function runEvaluation(k: number, mode: "baseline" | "experiment") {
    if (mode === "baseline") setLoadingBaseline(true);
    else setLoadingExperiment(true);
    setError("");
    try {
      const response = await fetch("/api/analysis/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topK: k }),
      });
      const data = (await response.json()) as EvaluationReport | { error?: string };
      if (!response.ok || !isEvaluationReport(data)) {
        throw new Error("error" in data ? data.error : "Evaluation failed.");
      }
      if (mode === "baseline") setBaseline(data);
      else setResult(data);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evaluation failed.");
    } finally {
      if (mode === "baseline") setLoadingBaseline(false);
      else setLoadingExperiment(false);
    }
  }

  async function evaluate(event: FormEvent) {
    event.preventDefault();
    if (loadingExperiment) return;
    await runEvaluation(topK, "experiment");
  }

  async function logout() {
    await fetch("/api/analysis/logout", { method: "POST" });
    window.location.reload();
  }

  const activeReport = baseline ?? fallbackReport;
  const summary = activeReport.summary;
  const comparisonData = topKData.map((row) =>
    row.name === `k=${summary.top_k}`
      ? {
          ...row,
          found: Number((summary.found_rate * 100).toFixed(2)),
          precision: Number((summary.mean_precision_at_k * 100).toFixed(2)),
        }
      : row,
  );
  const filteredRows = useMemo(() => {
    const searchText = query.trim().toLowerCase();
    return activeReport.rows.filter((row) => {
      const matchesSearch =
        !searchText ||
        row.question.toLowerCase().includes(searchText) ||
        row.expected_source.toLowerCase().includes(searchText) ||
        row.top_chunk_id.toLowerCase().includes(searchText);
      const matchesLanguage = languageFilter === "all" || row.language === languageFilter;
      const matchesStatus = statusFilter === "all" || row.status === statusFilter;
      return matchesSearch && matchesLanguage && matchesStatus;
    });
  }, [activeReport.rows, languageFilter, query, statusFilter]);

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-[#203b58]/80 bg-[#07111f]/90 px-4 py-3 backdrop-blur-xl sm:px-8">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-[#153c63] text-[#8dc8ff]">
              <LockKeyhole aria-hidden="true" size={19} />
            </div>
            <div className="min-w-0">
              <p className="truncate font-semibold">Private retrieval analysis</p>
              <p className="text-[11px] text-[#8ca6c1]">Admin only · baseline Top-k = 5</p>
            </div>
          </div>
          <button
            onClick={logout}
            className="inline-flex min-h-11 items-center gap-2 rounded-lg border border-[#294864] px-3 text-sm transition-colors hover:bg-[#10243a]"
          >
            <LogOut aria-hidden="true" size={16} />
            <span className="hidden sm:inline">Log out</span>
          </button>
        </div>
      </header>

      <main className="mx-auto max-w-[1440px] space-y-6 px-4 py-8 sm:px-8 sm:py-10">
        <section className="flex flex-col justify-between gap-5 md:flex-row md:items-end">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[.18em] text-[#5ba7ff]">
              <ShieldCheck aria-hidden="true" size={15} />
              Verified baseline
            </div>
            <h1 className="mt-3 text-3xl font-semibold sm:text-4xl">Complete RAG evaluation</h1>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-[#8ca6c1]">
              Coverage, precision, ranking quality, language split, per-question details, and temporary Top-k experiments.
            </p>
          </div>
          <div className="rounded-lg border border-[#2b805f] bg-[#0a2826] px-4 py-3 text-sm text-[#71e5b9]">
            Production retrieval stays fixed at k=5
          </div>
        </section>

        {loadingBaseline && (
          <p role="status" className="rounded-lg border border-[#294864] bg-[#091525] p-4 text-sm text-[#8ca6c1]">
            Loading baseline evaluation from the local Chroma index...
          </p>
        )}
        {error && (
          <p role="alert" className="rounded-lg border border-[#6d3544] bg-[#3a1824] p-4 text-sm text-[#ffb1bd]">
            {error}
          </p>
        )}

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <MetricCard label="Found Rate@5" value={percent(summary.found_rate)} hint={`${summary.found_expected_evidence}/${summary.scored_questions} in-scope questions found expected evidence`} icon={CheckCircle2} accent="green" />
          <MetricCard label="Mean Precision@5" value={percent(summary.mean_precision_at_k)} hint="Average share of relevant chunks inside the five returned chunks" icon={Gauge} />
          <MetricCard label="MAP@5" value={percent(summary.map_at_k)} hint="Ranking quality across expected evidence, not just whether it appeared" icon={BarChart3} />
          <MetricCard label="MRR" value={percent(summary.mrr)} hint="How early the first correct chunk appears in the ranked list" icon={Activity} accent="amber" />
        </section>

        <div className="grid gap-6 xl:grid-cols-[1.15fr_.85fr]">
          <Section title="Ranking quality" description="This answers the reviewer point: finding evidence is good, but ranking it first is better.">
            <div className="grid gap-3 sm:grid-cols-4">
              <RankBox label="Correct at Rank 1" value={summary.rank_1_count} total={summary.scored_questions} />
              <RankBox label="Correct inside Top 3" value={summary.top_3_count} total={summary.scored_questions} />
              <RankBox label="Correct inside Top 5" value={summary.top_5_count} total={summary.scored_questions} />
              <RankBox label="Missed" value={summary.missed_count} total={summary.scored_questions} danger />
            </div>
            <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_220px]">
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={activeReport.rank_distribution}>
                    <CartesianGrid stroke="#203b58" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" tick={{ fill: "#8ca6c1", fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis tick={{ fill: "#8ca6c1", fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#0d1b2d", border: "1px solid #203b58", borderRadius: 8 }} />
                    <Bar dataKey="count" name="Questions" fill="#46d6a0" radius={[5, 5, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="rounded-lg bg-[#091525] p-4">
                <p className="text-sm text-[#8ca6c1]">Average first correct rank</p>
                <p className="mt-3 text-4xl font-semibold">{summary.average_first_correct_rank}</p>
                <p className="mt-3 text-xs leading-5 text-[#8ca6c1]">
                  Lower is better. Rank 1 means the best retrieved chunk already contains the expected evidence.
                </p>
              </div>
            </div>
          </Section>

          <Section title="Coverage and languages" description="Baseline k=5 evaluation coverage.">
            <div className="space-y-5">
              <ProgressLine label="Overall" value={summary.found_rate} count={`${summary.found_expected_evidence}/${summary.scored_questions}`} />
              <ProgressLine label="Arabic" value={summary.ar_found_rate} count={summary.ar_found_count} />
              <ProgressLine label="English" value={summary.en_found_rate} count={summary.en_found_count} />
              <div className="grid grid-cols-3 gap-3 pt-2 text-center">
                <Count value={String(summary.total_questions)} label="Total" />
                <Count value={String(summary.scored_questions)} label="In scope" />
                <Count value={String(summary.out_of_scope_questions)} label="Out of scope" />
              </div>
            </div>
          </Section>
        </div>

        <Section title="Temporary Top-k evaluation" description="Run a temporary experiment for any k from 1 to 20. It displays results only and does not change the fixed production k=5.">
          <form onSubmit={evaluate} className="flex flex-col gap-4 rounded-lg border border-[#294864] bg-[#091525] p-4 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label htmlFor="evaluation-k" className="mb-2 block text-sm font-medium">
                Experimental k value
              </label>
              <input id="evaluation-k" type="number" min={1} max={20} value={topK} onChange={(event) => setTopK(Number(event.target.value))} className="min-h-12 w-full rounded-lg border border-[#294864] bg-[#0d1b2d] px-4 text-base focus:border-[#5ba7ff] focus:outline-none" />
            </div>
            <button disabled={loadingExperiment || topK < 1 || topK > 20} className="inline-flex min-h-12 items-center justify-center gap-2 rounded-lg bg-[#1e5a91] px-5 font-semibold text-white transition-colors hover:bg-[#276fae] disabled:cursor-not-allowed disabled:opacity-50">
              {loadingExperiment ? <LoaderCircle aria-hidden="true" size={18} className="animate-spin" /> : <FlaskConical aria-hidden="true" size={18} />}
              {loadingExperiment ? "Running evaluation..." : `Evaluate k=${topK}`}
            </button>
          </form>
          {loadingExperiment && <p role="status" className="mt-3 text-sm text-[#8ca6c1]">Loading the embedding model and scoring all questions. This can take around 20-40 seconds.</p>}
          {result && <ExperimentResult report={result} />}
        </Section>

        <div className="grid gap-6 xl:grid-cols-[1.35fr_1fr]">
          <Section title="Top-k comparison" description="Reference results showing the recall/precision trade-off.">
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={comparisonData}>
                  <CartesianGrid stroke="#203b58" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fill: "#8ca6c1", fontSize: 12 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[0, 100]} tick={{ fill: "#8ca6c1", fontSize: 11 }} axisLine={false} tickLine={false} tickFormatter={(value) => `${value}%`} />
                  <Tooltip contentStyle={{ background: "#0d1b2d", border: "1px solid #203b58", borderRadius: 8 }} />
                  <Legend />
                  <Bar dataKey="found" name="Found rate" fill="#46d6a0" radius={[5, 5, 0, 0]} />
                  <Bar dataKey="precision" name="Precision" fill="#5ba7ff" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Section>
          <Section title="Metric guide" description="Short explanations for presentation and judging.">
            <dl className="space-y-4 text-sm">
              <MetricGuide term="Found Rate" text="Expected evidence appears anywhere in Top-k." />
              <MetricGuide term="Precision@k" text="Share of returned chunks that are strictly relevant." />
              <MetricGuide term="MAP@k" text="Rewards retrieving expected evidence and ranking it early." />
              <MetricGuide term="MRR" text="Measures how early the first correct chunk appears." />
            </dl>
          </Section>
        </div>

        <Section title="Question-level evidence" description="Detailed rows used to explain why a question passed or failed and where the first correct chunk appeared.">
          <div className="mb-4 grid gap-3 lg:grid-cols-[1fr_160px_160px]">
            <label className="relative block">
              <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#8ca6c1]" size={17} />
              <span className="sr-only">Search questions</span>
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search question, expected recommendation, or chunk id" className="min-h-11 w-full rounded-lg border border-[#294864] bg-[#091525] pl-10 pr-3 text-sm focus:border-[#5ba7ff] focus:outline-none" />
            </label>
            <Select label="Language" value={languageFilter} onChange={setLanguageFilter} options={[["all", "All languages"], ["ar", "Arabic"], ["en", "English"]]} />
            <Select label="Status" value={statusFilter} onChange={setStatusFilter} options={[["all", "All statuses"], ["PASS", "Pass"], ["FAIL", "Fail"], ["REVIEW_REFUSAL", "Out of scope"]]} />
          </div>
          <QuestionTable rows={filteredRows} />
        </Section>

        <div className="grid gap-6 lg:grid-cols-2">
          <ExperimentTable title="Embedding model experiments" description="Why multilingual-e5-base remains selected." heading="Model" rows={modelData.map((row) => [row.name, `${row.found}%`, `${row.map}%`])} />
          <ExperimentTable title="Chunk experiments" description="Structure-aware recursive chunks compared by token budget." heading="Size / overlap" rows={chunkData.map((row) => [row.name, `${row.found}%`, `${row.map}%`])} />
        </div>

        <div className="flex items-center gap-3 rounded-lg border border-[#203b58] bg-[#0d1b2d] p-5 text-sm">
          <Database aria-hidden="true" className="text-[#46d6a0]" size={19} />
          <div>
            <p className="font-semibold">Evaluation data is local</p>
            <p className="mt-1 text-xs text-[#8ca6c1]">Experiments use the existing Chroma index and test questions without changing production retrieval.</p>
          </div>
        </div>
      </main>
    </div>
  );
}

function MetricCard({ label, value, hint, icon: Icon, accent = "blue" }: { label: string; value: string; hint: string; icon: typeof Activity; accent?: "blue" | "green" | "amber" | "red" }) {
  const colors = {
    blue: "bg-[#102e4d] text-[#8dc8ff]",
    green: "bg-[#0b3a32] text-[#71e5b9]",
    amber: "bg-[#3b2a12] text-[#f4c879]",
    red: "bg-[#421925] text-[#ff9aad]",
  }[accent];
  return (
    <article className="rounded-lg border border-[#203b58] bg-[#0d1b2d]/95 p-4">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-[#8ca6c1]">{label}</p>
        <span className={`rounded-lg p-2 ${colors}`}>
          <Icon aria-hidden="true" size={18} />
        </span>
      </div>
      <p className="mt-3 text-2xl font-semibold sm:text-3xl">{value}</p>
      <p className="mt-2 text-xs leading-5 text-[#8ca6c1]">{hint}</p>
    </article>
  );
}

function Section({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="min-w-0 rounded-lg border border-[#203b58] bg-[#0d1b2d]/95 p-4 sm:p-5">
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="mt-1 text-sm leading-6 text-[#8ca6c1]">{description}</p>
      <div className="mt-5 min-w-0">{children}</div>
    </section>
  );
}

function RankBox({ label, value, total, danger = false }: { label: string; value: number; total: number; danger?: boolean }) {
  return (
    <div className="rounded-lg bg-[#091525] p-4">
      <p className="text-xs text-[#8ca6c1]">{label}</p>
      <p className={`mt-2 text-2xl font-semibold ${danger ? "text-[#ff9aad]" : "text-[#71e5b9]"}`}>{value}</p>
      <p className="mt-1 text-xs text-[#8ca6c1]">{percent(value / total)}</p>
    </div>
  );
}

function ProgressLine({ label, value, count }: { label: string; value: number; count: string }) {
  return (
    <div>
      <div className="mb-2 flex justify-between gap-3 text-sm">
        <span>{label}</span>
        <span className="text-[#8ca6c1]">{count} · {percent(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#1a3049]">
        <div className="h-full rounded-full bg-[#46d6a0]" style={{ width: `${Math.min(value * 100, 100)}%` }} />
      </div>
    </div>
  );
}

function Count({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-lg bg-[#091525] p-3">
      <p className="text-xl font-semibold">{value}</p>
      <p className="text-xs text-[#8ca6c1]">{label}</p>
    </div>
  );
}

function MetricGuide({ term, text }: { term: string; text: string }) {
  return (
    <div>
      <dt className="font-semibold text-[#8dc8ff]">{term}</dt>
      <dd className="mt-1 text-[#8ca6c1]">{text}</dd>
    </div>
  );
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: [string, string][] }) {
  return (
    <label>
      <span className="sr-only">{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} className="min-h-11 w-full rounded-lg border border-[#294864] bg-[#091525] px-3 text-sm focus:border-[#5ba7ff] focus:outline-none">
        {options.map(([optionValue, text]) => (
          <option key={optionValue} value={optionValue}>
            {text}
          </option>
        ))}
      </select>
    </label>
  );
}

function ExperimentResult({ report }: { report: EvaluationReport }) {
  const summary = report.summary;
  return (
    <div className="mt-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-semibold">Temporary result for k={summary.top_k}</h3>
        <span className="rounded-full border border-[#735528] bg-[#342814] px-3 py-1 text-xs text-[#f4d28e]">Not saved · baseline still k=5</span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard label={`Found Rate@${summary.top_k}`} value={percent(summary.found_rate)} hint={`${summary.found_expected_evidence}/${summary.scored_questions} found`} icon={CheckCircle2} accent="green" />
        <MetricCard label={`Precision@${summary.top_k}`} value={percent(summary.mean_precision_at_k)} hint="Mean strict precision" icon={Gauge} />
        <MetricCard label={`MAP@${summary.top_k}`} value={percent(summary.map_at_k)} hint="Average precision by rank" icon={BarChart3} />
        <MetricCard label="MRR" value={percent(summary.mrr)} hint="First correct result rank" icon={Activity} accent="amber" />
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <RankBox label="Rank 1" value={summary.rank_1_count} total={summary.scored_questions} />
        <RankBox label="Top 3" value={summary.top_3_count} total={summary.scored_questions} />
        <RankBox label="Top 5" value={summary.top_5_count} total={summary.scored_questions} />
        <RankBox label="Missed" value={summary.missed_count} total={summary.scored_questions} danger />
      </div>
    </div>
  );
}

function QuestionTable({ rows }: { rows: EvaluationRow[] }) {
  if (rows.length === 0) {
    return <div className="rounded-lg border border-[#294864] bg-[#091525] p-6 text-center text-sm text-[#8ca6c1]">No question rows to display yet. Run the baseline evaluation or change the filters.</div>;
  }
  return (
    <div className="max-h-[620px] max-w-full overflow-auto rounded-lg border border-[#203b58]">
      <table className="w-full min-w-[980px] text-left text-sm">
        <thead className="sticky top-0 bg-[#091525] text-xs uppercase tracking-wide text-[#8ca6c1]">
          <tr>
            <th className="p-3 font-medium">Question</th>
            <th className="p-3 font-medium">Lang</th>
            <th className="p-3 font-medium">Status</th>
            <th className="p-3 font-medium">First correct rank</th>
            <th className="p-3 font-medium">Relevant / k</th>
            <th className="p-3 font-medium">Precision</th>
            <th className="p-3 font-medium">AP</th>
            <th className="p-3 font-medium">Top score</th>
            <th className="p-3 font-medium">Top chunk</th>
            <th className="p-3 font-medium">Expected evidence</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.id}-${row.variant}-${row.language}`} className="border-t border-[#203b58] align-top transition-colors hover:bg-[#10243a]">
              <td className="max-w-[320px] p-3 leading-5">{row.question}</td>
              <td className="p-3 uppercase text-[#8ca6c1]">{row.language}</td>
              <td className="p-3"><StatusBadge status={row.status} /></td>
              <td className="p-3">{row.best_rank === "" ? <span className="text-[#8ca6c1]">-</span> : `#${row.best_rank}`}</td>
              <td className="p-3">{row.relevant_in_top_k === "" ? "-" : `${row.relevant_in_top_k}/${row.top_k}`}</td>
              <td className="p-3">{formatMaybePercent(row.precision_at_k)}</td>
              <td className="p-3">{formatMaybePercent(row.average_precision_at_k)}</td>
              <td className="p-3">{row.top_score.toFixed(4)}</td>
              <td className="p-3"><span className="code text-xs text-[#8dc8ff]">{row.top_chunk_id}</span><p className="mt-1 text-xs text-[#8ca6c1]">page {row.top_page}</p></td>
              <td className="max-w-[260px] p-3 text-xs leading-5 text-[#8ca6c1]">{row.expected_recommendations || row.expected_source}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  if (status === "PASS") {
    return <span className="inline-flex items-center gap-1 rounded-full bg-[#0b3a32] px-2 py-1 text-xs text-[#71e5b9]"><CheckCircle2 size={13} /> Pass</span>;
  }
  if (status === "FAIL") {
    return <span className="inline-flex items-center gap-1 rounded-full bg-[#421925] px-2 py-1 text-xs text-[#ff9aad]"><XCircle size={13} /> Fail</span>;
  }
  return <span className="inline-flex items-center gap-1 rounded-full bg-[#3b2a12] px-2 py-1 text-xs text-[#f4c879]"><Target size={13} /> OOS</span>;
}

function formatMaybePercent(value: number | "") {
  if (value === "") return "-";
  return percent(value);
}

function ExperimentTable({ title, description, heading, rows }: { title: string; description: string; heading: string; rows: string[][] }) {
  return (
    <Section title={title} description={description}>
      <div className="max-w-full overflow-x-auto">
        <table className="w-full min-w-[320px] text-left text-sm">
          <thead className="text-[#8ca6c1]">
            <tr>
              <th className="pb-3 font-medium">{heading}</th>
              <th className="pb-3 font-medium">Found rate</th>
              <th className="pb-3 font-medium">MAP</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row[0]} className="border-t border-[#203b58]">
                <td className="py-3 font-medium">{row[0]}</td>
                <td>{row[1]}</td>
                <td>{row[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}
