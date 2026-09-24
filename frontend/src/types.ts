// Miroir du JSON de GET /causal/{run_id} (src/causal/signals.py).
export type Modality = "metrics" | "logs" | "traces";

export interface LogTemplate {
  template: string;
  beta: number;
  n_base: number;
  n_probe: number;
}

export interface ServiceNode {
  service: string;
  display_name: string;
  anomaly_score: number;
  modality_contributions: Partial<Record<Modality, number>>;
  n_modalities_deviating: number;
  onset_s: number | null;
  metrics: { score: number; top_metric: string | null; onset_s: number | null } | null;
  logs: { score: number; entropy_bits: number; top_templates: LogTemplate[]; onset_s: number | null } | null;
  traces: { score: number; reason: string; duration_ratio: number | null; onset_s: number | null } | null;
}

export interface CallEdge {
  caller: string;
  callee: string;
  n_calls: number;
  alpha_caller_to_callee?: number | null;
  alpha_callee_to_caller?: number | null;
}

export interface RunSignals {
  run_id: string;
  subset: string;
  window: { kind: string; probe_seconds: number; base_seconds: number };
  modalities_available: Record<Modality, boolean>;
  provenance: Record<string, string>;
  nodes: ServiceNode[];
  edges: CallEdge[];
  attention: { nodes: string[]; layer1: number[][]; layer2: number[][] } | null;
}

export interface WhatIf {
  hypothesis: string;
  hypothesis_anomaly_score: number | null;
  propagation_path: string[];
  explained_anomalous: string[];
  unexplained_anomalous: string[];
  coverage: number | null;
  caveat: string;
  paths: string[][];
}

export interface RunInfo {
  run_id: string;
  anormal: boolean;
  signaux_en_cache: boolean;
}
