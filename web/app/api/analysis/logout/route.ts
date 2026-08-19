import { NextResponse } from "next/server";
import { ANALYSIS_COOKIE } from "../../../../lib/analysis-auth";

export async function POST() {
  const response = NextResponse.json({ ok: true });
  response.cookies.set(ANALYSIS_COOKIE, "", { path: "/", maxAge: 0 });
  return response;
}
