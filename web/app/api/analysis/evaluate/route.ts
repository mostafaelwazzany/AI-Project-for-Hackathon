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
};

function runEvaluation(topK: number) {
  return new Promise<Summary>((resolve, reject) => {
    const root = path.resolve(process.cwd(), "..");
    const python = process.platform === "win32"
      ? path.join(root, ".venv", "Scripts", "python.exe")
      : path.join(root, ".venv", "bin", "python");
    const child = spawn(
      python,
      ["evaluate.py", "--top-k", String(topK), "--no-save", "--json"],
      { cwd: root, windowsHide: true },
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
        resolve(JSON.parse(jsonLine) as Summary);
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
    return NextResponse.json({ summary: await runEvaluation(topK) });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Evaluation failed." },
      { status: 500 },
    );
  }
}
