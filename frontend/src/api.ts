import type { RunInfo, RunSignals, WhatIf } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
}

// run_id RCAEval contient des "/": transmis tel quel (route FastAPI {run_id:path}).
export const fetchRuns = () => getJson<RunInfo[]>("/runs");
export const fetchSignals = (runId: string) => getJson<RunSignals>(`/causal/${runId}`);
export const fetchWhatIf = (runId: string, hypothesis: string, minScore: number) =>
  getJson<WhatIf>(`/causal/${runId}/whatif?hypothesis=${encodeURIComponent(hypothesis)}&min_score=${minScore}`);
export const reportUrl = (runId: string, hypothesis: string | null) =>
  `/causal/${runId}/report${hypothesis ? `?hypothesis=${encodeURIComponent(hypothesis)}` : ""}`;
export const reportPdfUrl = (runId: string, hypothesis: string | null) =>
  `/causal/${runId}/report.pdf${hypothesis ? `?hypothesis=${encodeURIComponent(hypothesis)}` : ""}`;
export const jsonldUrl = (runId: string) => `/causal/${runId}/jsonld`;
