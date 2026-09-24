import { useState } from "react";
import * as d3 from "d3";
import { scoreColor } from "../colors";
import type { ServiceNode } from "../types";

interface Props {
  nodes: ServiceNode[];
  selected: string | null;
  onSelect: (service: string) => void;
}

const AXES: { key: string; label: string; value: (n: ServiceNode) => number | null }[] = [
  { key: "logs", label: "logs", value: (n) => n.modality_contributions.logs ?? null },
  { key: "metrics", label: "métriques", value: (n) => n.modality_contributions.metrics ?? null },
  { key: "traces", label: "traces", value: (n) => n.modality_contributions.traces ?? null },
  { key: "a", label: "a_i", value: (n) => n.anomaly_score },
];
const W = 520;
const H = 340;
const M = { top: 30, right: 30, bottom: 20, left: 30 };

/**
 * Vue de corrélation modale (section 3.2.3) en coordonnées parallèles: un
 * trait par service, axes = contribution de chaque modalité. Ce ne sont PAS
 * des "poids d'attention intermodaux" (le détecteur retenu, fusion tardive,
 * n'en a pas): une modalité absente pour un service est tracée en pointillé
 * à 0 plutôt qu'inventée.
 */
export default function ModalParallel({ nodes, selected, onSelect }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);
  const x = d3.scalePoint(AXES.map((a) => a.key), [M.left, W - M.right]);
  const y = d3.scaleLinear([0, 1], [H - M.bottom, M.top]);
  const css = getComputedStyle(document.documentElement);
  const axisColor = css.getPropertyValue("--axis").trim();
  const text2 = css.getPropertyValue("--text-2").trim();
  const highlight = css.getPropertyValue("--highlight").trim();
  const focus = hovered ?? selected;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Contributions par modalité et par service">
        {AXES.map((a) => (
          <g key={a.key} transform={`translate(${x(a.key)},0)`}>
            <line y1={y(0)} y2={y(1)} stroke={axisColor} />
            <text y={M.top - 12} textAnchor="middle" fontSize={12} fill={text2}>{a.label}</text>
            {[0, 0.5, 1].map((t) => (
              <text key={t} x={-6} y={y(t) + 4} textAnchor="end" fontSize={10} fill={text2}>{t}</text>
            ))}
          </g>
        ))}
        {[...nodes].sort((a, b) => a.anomaly_score - b.anomaly_score).map((n) => {
          const points = AXES.map((a) => [x(a.key)!, y(a.value(n) ?? 0)] as [number, number]);
          const missing = AXES.some((a) => a.value(n) === null);
          const isFocus = n.service === focus;
          return (
            <path
              key={n.service}
              d={d3.line()(points)!}
              fill="none"
              stroke={isFocus ? highlight : scoreColor(n.anomaly_score)}
              strokeWidth={isFocus ? 3.5 : 2}
              strokeOpacity={focus && !isFocus ? 0.25 : 0.9}
              strokeDasharray={missing ? "4 3" : undefined}
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHovered(n.service)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onSelect(n.service)}
            >
              <title>{`${n.display_name} — a_i=${n.anomaly_score.toFixed(2)}${missing ? " (modalité absente)" : ""}`}</title>
            </path>
          );
        })}
      </svg>
      <div className="legend">
        <span>un trait = un service · pointillé = une modalité absente (tracée à 0)</span>
        <span>{focus ? `survol/sélection : ${nodes.find((n) => n.service === focus)?.display_name ?? focus}` : "survoler un trait"}</span>
      </div>
    </div>
  );
}
