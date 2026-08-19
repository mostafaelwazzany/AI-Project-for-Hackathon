import { cookies } from "next/headers";
import AnalysisDashboard from "../../components/dashboard";
import AnalysisLogin from "../../components/analysis-login";
import { ANALYSIS_COOKIE, analysisConfigured, validSession } from "../../lib/analysis-auth";

export default async function AnalysisPage() {
  const session = (await cookies()).get(ANALYSIS_COOKIE)?.value;
  if (!validSession(session)) {
    return <AnalysisLogin configured={analysisConfigured()} />;
  }
  return <AnalysisDashboard />;
}
