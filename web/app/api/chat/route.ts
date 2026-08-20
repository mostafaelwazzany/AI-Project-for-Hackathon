import { NextRequest, NextResponse } from "next/server";
import { ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";
import readline from "node:readline";

export const runtime = "nodejs";

type Source = {
  url: string;
  text: string;
  document: string;
  page?: string;
  section?: string;
  chunk_id?: string;
};
type Result = {
  answer?: string;
  error?: string;
  ready?: boolean;
  source?: Source | null;
  sources?: Source[];
};
type Pending = {
  id: string;
  resolve: (value: Result) => void;
  timer: NodeJS.Timeout;
};
type RagState = {
  process?: ChildProcessWithoutNullStreams;
  pending: Pending[];
  nextId?: number;
};

const globalRag = globalThis as typeof globalThis & { ragState?: RagState };
const state = globalRag.ragState ?? { pending: [] };
globalRag.ragState = state;

function hfApiUrl() {
  const base = process.env.HF_SPACE_API_URL?.replace(/\/$/, "");
  return base ?? "";
}

async function sendRemote(question: string): Promise<Result> {
  const base = hfApiUrl();
  if (!base) return { error: "HF_SPACE_API_URL is not configured." };
  const startResponse = await fetch(`${base}/gradio_api/call/v2/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: question }),
    signal: AbortSignal.timeout(60_000),
  });
  const startData = (await startResponse.json().catch(() => null)) as {
    event_id?: string;
  } | null;
  if (!startResponse.ok || !startData?.event_id) {
    return { error: "The hosted RAG service could not start the answer job." };
  }
  const resultResponse = await fetch(
    `${base}/gradio_api/call/answer/${startData.event_id}`,
    { signal: AbortSignal.timeout(90_000) },
  );
  const eventText = await resultResponse.text();
  if (!resultResponse.ok) {
    return { error: "The hosted RAG service failed while preparing the answer." };
  }
  const dataLine = eventText
    .split(/\r?\n/)
    .find((line) => line.startsWith("data: "));
  if (!dataLine) {
    return { error: "The hosted RAG service returned an empty response." };
  }
  try {
    const parsed = JSON.parse(dataLine.slice(6)) as unknown[];
    return {
      answer: typeof parsed[0] === "string" ? parsed[0] : "",
      source: null,
      sources: [],
    };
  } catch {
    return { error: "The hosted RAG service returned an invalid response." };
  }
}

function rejectPending(message: string) {
  while (state.pending.length) {
    const item = state.pending.shift()!;
    clearTimeout(item.timer);
    item.resolve({ error: message });
  }
}

function startRag() {
  if (state.process && !state.process.killed) return state.process;
  const root = path.resolve(process.cwd(), "..");
  const python =
    process.env.PYTHON_EXECUTABLE ||
    (process.platform === "win32"
      ? path.join(root, ".venv", "Scripts", "python.exe")
      : "python3");
  const child = spawn(/* turbopackIgnore: true */ python, ["web_chat_bridge.py"], {
    cwd: root,
    windowsHide: true,
    env: {
      ...process.env,
      EMBEDDING_LOCAL_FILES_ONLY:
        process.env.EMBEDDING_LOCAL_FILES_ONLY ?? "false",
    },
  });
  state.process = child;
  readline.createInterface({ input: child.stdout }).on("line", (line) => {
    let parsed: Result & { id?: string };
    try {
      parsed = JSON.parse(line) as Result & { id?: string };
    } catch {
      parsed = { error: "The RAG service returned an invalid response." };
    }
    const pendingIndex = parsed.id
      ? state.pending.findIndex((pending) => pending.id === parsed.id)
      : 0;
    const item =
      pendingIndex >= 0 ? state.pending.splice(pendingIndex, 1)[0] : undefined;
    if (!item) return;
    clearTimeout(item.timer);
    delete parsed.id;
    item.resolve(parsed);
  });
  child.stderr.on("data", (chunk) =>
    console.error("RAG:", chunk.toString().trim()),
  );
  child.on("close", () => {
    state.process = undefined;
    rejectPending("The RAG service stopped. Please try again.");
  });
  return child;
}

function timeoutMessage(payload: { question: string } | { warmup: true }) {
  if ("warmup" in payload) return "The assistant is still warming up.";
  return /[\u0600-\u06ff]/.test(payload.question)
    ? "الرد استغرق وقتًا طويلًا. غالبًا السيرفر على Render ما زال يبدأ أو خدمة التوليد بطيئة. حاول مرة أخرى بعد لحظات."
    : "The answer took too long. The Render server may still be starting or the generation service is slow. Please try again shortly.";
}

function sendRag(payload: { question: string } | { warmup: true }) {
  return new Promise<Result>((resolve) => {
    const child = startRag();
    const id = String(state.nextId = (state.nextId ?? 0) + 1);
    const timer = setTimeout(() => {
      const index = state.pending.findIndex((item) => item.id === id);
      if (index >= 0) state.pending.splice(index, 1);
      resolve({ error: timeoutMessage(payload) });
    }, "warmup" in payload ? 20_000 : 45_000);
    state.pending.push({ id, resolve, timer });
    child.stdin.write(`${JSON.stringify({ ...payload, id })}\n`);
  });
}

export async function GET() {
  if (hfApiUrl()) return NextResponse.json({ ready: true });
  const result = await sendRag({ warmup: true });
  if (result.error)
    return NextResponse.json({ error: result.error }, { status: 502 });
  return NextResponse.json({ ready: true });
}

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as {
    question?: unknown;
  } | null;
  const question =
    typeof body?.question === "string" ? body.question.trim() : "";
  if (!question)
    return NextResponse.json(
      { error: "Please enter a question." },
      { status: 400 },
    );
  if (question.length > 1000)
    return NextResponse.json(
      { error: "Question is too long." },
      { status: 400 },
    );
  const result = hfApiUrl()
    ? await sendRemote(question)
    : await sendRag({ question });
  if (result.error)
    return NextResponse.json({ error: result.error }, { status: 502 });
  return NextResponse.json({
    answer: result.answer,
    source: result.source,
    sources: result.sources ?? [],
  });
}
