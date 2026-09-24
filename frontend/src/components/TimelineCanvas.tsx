import { useEffect, useMemo, useRef, useState } from "react";
import { scoreColor } from "../colors";
import type { ServiceNode } from "../types";

interface Props {
  nodes: ServiceNode[];
  probeSeconds: number;
  selected: string | null;
  onSelect: (service: string) => void;
}

const ROW_H = 34;
const LABEL_W = 170;
const TOP = 26;

interface Glyph {
  node: ServiceNode;
  x: number;
  y: number;
  w: number;
  h: number;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Vue chronologique sur canevas défilable (section 3.2.1 / 4.2): une ligne par
 * service, glyphe à l'instant de première déviation. Encodage (Table 1):
 * couleur = a_i, hauteur = entropie des logs H(L_i), largeur = rapport de
 * durée médiane des spans, bordure pointillée = déviation métrique notable.
 * Canvas plutôt que SVG: le zoom horizontal peut rendre la zone très large.
 */
export default function TimelineCanvas({ nodes, probeSeconds, selected, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pxPerSecond, setPxPerSecond] = useState(1.5);
  const [hover, setHover] = useState<{ glyph: Glyph; x: number; y: number } | null>(null);

  const rows = useMemo(
    () => [...nodes].sort((a, b) => (a.onset_s ?? Infinity) - (b.onset_s ?? Infinity) || b.anomaly_score - a.anomaly_score),
    [nodes],
  );
  const noOnsetX = LABEL_W + probeSeconds * pxPerSecond + 90;
  const width = noOnsetX + 70;
  const height = TOP + rows.length * ROW_H + 10;
  const maxEntropy = Math.max(1e-6, ...rows.map((n) => n.logs?.entropy_bits ?? 0));

  const glyphs: Glyph[] = useMemo(
    () =>
      rows.map((node, i) => {
        const ratio = Math.min(Math.max(node.traces?.duration_ratio ?? 1, 0.5), 4);
        const w = 10 * ratio;
        const h = 8 + 18 * ((node.logs?.entropy_bits ?? 0) / maxEntropy);
        const x = node.onset_s === null ? noOnsetX : LABEL_W + node.onset_s * pxPerSecond;
        return { node, x, y: TOP + i * ROW_H + ROW_H / 2, w, h };
      }),
    [rows, maxEntropy, noOnsetX, pxPerSecond],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(dpr, dpr);
    const [text, text2, grid, axis, stroke, highlight] = ["--text", "--text-2", "--grid", "--axis", "--glyph-stroke", "--highlight"].map(cssVar);
    ctx.clearRect(0, 0, width, height);
    ctx.font = "12px system-ui, sans-serif";

    // Graduations toutes les 60 s
    ctx.textAlign = "center";
    for (let t = 0; t <= probeSeconds; t += 60) {
      const x = LABEL_W + t * pxPerSecond;
      ctx.strokeStyle = t === 0 ? axis : grid;
      ctx.setLineDash(t === 0 ? [4, 3] : []);
      ctx.beginPath();
      ctx.moveTo(x, TOP - 6);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.fillStyle = text2;
      ctx.fillText(`${t}s`, x, 14);
    }
    ctx.setLineDash([]);
    ctx.fillStyle = grid;
    ctx.globalAlpha = 0.45;
    ctx.fillRect(noOnsetX - 55, TOP - 6, 110, height);
    ctx.globalAlpha = 1;
    ctx.fillStyle = text2;
    ctx.fillText("aucune déviation", noOnsetX, 14);

    glyphs.forEach((g) => {
      const isSel = g.node.service === selected;
      ctx.textAlign = "right";
      ctx.fillStyle = isSel ? highlight : text;
      ctx.font = isSel ? "600 12px system-ui, sans-serif" : "12px system-ui, sans-serif";
      ctx.fillText(g.node.display_name, LABEL_W - 10, g.y + 4);
      ctx.fillStyle = scoreColor(g.node.anomaly_score);
      ctx.beginPath();
      ctx.roundRect(g.x - g.w / 2, g.y - g.h / 2, g.w, g.h, 3);
      ctx.fill();
      ctx.lineWidth = isSel ? 3 : 1.5;
      ctx.strokeStyle = isSel ? highlight : stroke;
      ctx.setLineDash((g.node.modality_contributions.metrics ?? 0) >= 0.5 ? [3, 2] : []);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  }, [glyphs, width, height, probeSeconds, pxPerSecond, noOnsetX, selected]);

  const hitTest = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Cible plus grande que la marque (8 px de marge, ligne entière en y)
    return glyphs.find((g) => Math.abs(x - g.x) <= g.w / 2 + 8 && Math.abs(y - g.y) <= ROW_H / 2) ?? null;
  };

  return (
    <div>
      <div className="controls">
        <label>
          Zoom temporel ({pxPerSecond.toFixed(1)} px/s — faire défiler horizontalement)
          <input type="range" min={0.5} max={8} step={0.25} value={pxPerSecond} onChange={(e) => setPxPerSecond(+e.target.value)} />
        </label>
      </div>
      <div className="timeline-scroll">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label="Chronologie des premières déviations par service"
          onMouseMove={(e) => {
            const g = hitTest(e);
            setHover(g ? { glyph: g, x: e.clientX, y: e.clientY } : null);
            e.currentTarget.style.cursor = g ? "pointer" : "default";
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const g = hitTest(e);
            if (g) onSelect(g.node.service);
          }}
        />
      </div>
      <div className="legend">
        <span><span className="swatch" /> couleur = a_i (0 → 1)</span>
        <span>hauteur = entropie des logs</span>
        <span>largeur = durée des spans (× référence)</span>
        <span>bordure pointillée = déviation métrique</span>
      </div>
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          <b>{hover.glyph.node.display_name}</b>
          <br />a_i = {hover.glyph.node.anomaly_score.toFixed(2)} · {hover.glyph.node.n_modalities_deviating} modalité(s) en déviation
          <br />début : {hover.glyph.node.onset_s === null ? "—" : `${hover.glyph.node.onset_s.toFixed(0)} s`}
          <br />entropie logs : {(hover.glyph.node.logs?.entropy_bits ?? 0).toFixed(2)} bits · durée ×
          {(hover.glyph.node.traces?.duration_ratio ?? 1).toFixed(2)}
        </div>
      )}
    </div>
  );
}
