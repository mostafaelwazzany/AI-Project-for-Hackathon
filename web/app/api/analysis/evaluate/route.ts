import { spawn } from "node:child_process";
import path from "node:path";
import { cookies } from "next/headers";
import { NextRequest, NextResponse } from "next/server";
import { ANALYSIS_COOKIE, validSession } from "../../../../lib/analysis-auth";

export const runtime = "nodejs";

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

function runEvaluation(topK: number) {
  return new Promise<EvaluationReport>((resolve, reject) => {
    const root = path.resolve(process.cwd(), "..");
    const python = process.platform === "win32"
      ? path.join(root, ".venv", "Scripts", "python.exe")
      : path.join(root, ".venv", "bin", "python");
    const child = spawn(
      python,
      ["evaluate.py", "--top-k", String(topK), "--json"],
      {
        cwd: root,
        env: { ...process.env, PYTHONIOENCODING: "utf-8" },
        windowsHide: true,
      },
    );
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("Evaluation timed out."));
    }, 5 * 60_000);
    child.stdout.on("data", (chunk) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk) => (stderr += chunk.toString()));
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || "Evaluation failed."));
        return;
      }
      try {
        const jsonLine = stdout.trim().split(/\r?\n/).at(-1) ?? "";
        resolve(JSON.parse(jsonLine) as EvaluationReport);
      } catch {
        reject(new Error("The evaluation returned an invalid summary."));
      }
    });
  });
}

export async function POST(request: NextRequest) {
  const session = (await cookies()).get(ANALYSIS_COOKIE)?.value;
  if (!validSession(session)) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }
  const body = (await request.json().catch(() => null)) as { topK?: unknown } | null;
  const topK = Number(body?.topK);
  if (!Number.isInteger(topK) || topK < 1 || topK > 20) {
    return NextResponse.json({ error: "k must be an integer from 1 to 20." }, { status: 400 });
  }
  try {
    return NextResponse.json(await runEvaluation(topK));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Evaluation failed." },
      { status: 500 },
    );
  }
}
