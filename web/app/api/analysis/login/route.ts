import { NextRequest, NextResponse } from "next/server";
import { ANALYSIS_COOKIE, sessionToken, validPassword } from "../../../../lib/analysis-auth";

export async function POST(request: NextRequest) {
  const body = (await request.json().catch(() => null)) as { password?: unknown } | null;
  const password = typeof body?.password === "string" ? body.password : "";
  if (!validPassword(password)) {
    return NextResponse.json({ error: "Incorrect password." }, { status: 401 });
  }
  const response = NextResponse.json({ ok: true });
  response.cookies.set(ANALYSIS_COOKIE, sessionToken(), {
    httpOnly: true,
    sameSite: "strict",
    secure: request.nextUrl.protocol === "https:",
    path: "/",
    maxAge: 60 * 60 * 8,
  });
  return response;
}
