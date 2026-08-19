import { createHash, timingSafeEqual } from "node:crypto";

export const ANALYSIS_COOKIE = "analysis_session";

function password() {
  return process.env.ANALYSIS_PASSWORD ?? "";
}

export function analysisConfigured() {
  return password().length >= 8;
}

export function sessionToken() {
  return createHash("sha256")
    .update(`colorectal-analysis:${password()}`)
    .digest("hex");
}

export function validPassword(candidate: string) {
  const expected = Buffer.from(password());
  const received = Buffer.from(candidate);
  return (
    analysisConfigured() &&
    expected.length === received.length &&
    timingSafeEqual(expected, received)
  );
}

export function validSession(value?: string) {
  if (!value || !analysisConfigured()) return false;
  const expected = Buffer.from(sessionToken());
  const received = Buffer.from(value);
  return expected.length === received.length && timingSafeEqual(expected, received);
}
