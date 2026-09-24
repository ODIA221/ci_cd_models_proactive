import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { scoreColor } from "../colors";
import type { CallEdge, ServiceNode } from "../types";

interface Props {
  nodes: ServiceNode[];
  edges: CallEdge[];
  hasAttention: boolean;
  threshold: number;
  selected: string | null;
  path: string[] | null;
  onSelect: (service: string) => void;
}

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  label: string;
  score: number;
}

interface SimLink extends d3.SimulationLinkDatum<SimNode> {
  edge: CallEdge;
  onPath: boolean;
}

const WIDTH = 760;
const HEIGHT = 520;

/**
 * Vue de propagation (section 3.2.2), disposition dirigée par forces avec
 * animation physique (d3-force). Flèches dans le sens de propagation d'un
 * SYMPTÔME (appelé -> appelant). Épaisseur = alpha GAT de l'appelant vers
 * l'appelé si disponible, sinon volume d'appels. Les arêtes du chemin
 * sélectionné sont animées ("ondes de propagation"), sauf si l'utilisateur a
 * demandé la réduction des animations.
 */
export default function ForceGraph({ nodes, edges, hasAttention, threshold, selected, path, onSelect }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; title: string; lines: string[] } | null>(null);

  useEffect(() => {
    const svg = d3.select(svgRef.current!);
    svg.selectAll("*").remove();

    const scores = new Map(nodes.map((n) => [n.service, n]));
    const pathSet = new Set(path ?? []);
    const pathEdges = new Set((path ?? []).slice(1).map((caller, i) => `${path![i]}->${caller}`));
    const inGraph = new Set(edges.flatMap((e) => [e.caller, e.callee]));
    const keep = [...inGraph].filter(
      (s) => (scores.get(s)?.anomaly_score ?? 0) >= threshold || s === selected || pathSet.has(s),
    );
    const keepSet = new Set(keep);
    const simNodes: SimNode[] = keep.map((id) => ({
      id,
      label: scores.get(id)?.display_name ?? id,
      score: scores.get(id)?.anomaly_score ?? 0,
    }));
    const simLinks: SimLink[] = edges
      .filter((e) => keepSet.has(e.caller) && keepSet.has(e.callee))
      .map((e) => ({ source: e.callee, target: e.caller, edge: e, onPath: pathEdges.has(`${e.callee}->${e.caller}`) }));

    const css = getComputedStyle(document.documentElement);
    const axis = css.getPropertyValue("--axis").trim();
    const text = css.getPropertyValue("--text").trim();
    const highlight = css.getPropertyValue("--highlight").trim();
    const stroke = css.getPropertyValue("--glyph-stroke").trim();

    const defs = svg.append("defs");
    for (const [id, color] of [["arrow", axis], ["arrow-hl", highlight]] as const) {
      defs.append("marker").attr("id", id).attr("viewBox", "0 -5 10 10").attr("refX", 10).attr("markerWidth", 6)
        .attr("markerHeight", 6).attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("fill", color);
    }

    const width = (l: SimLink) =>
      hasAttention && l.edge.alpha_caller_to_callee != null ? 1 + 8 * l.edge.alpha_caller_to_callee : 1 + Math.log10(1 + l.edge.n_calls);
    const radius = (n: SimNode) => 8 + 16 * n.score;

    const link = svg.append("g").selectAll<SVGLineElement, SimLink>("line").data(simLinks).join("line")
      .attr("stroke", (l) => (l.onPath ? highlight : axis))
      .attr("stroke-opacity", (l) => (l.onPath || pathEdges.size === 0 ? 0.9 : 0.3))
      .attr("stroke-width", (l) => width(l) + (l.onPath ? 2 : 0))
      .attr("marker-end", (l) => `url(#${l.onPath ? "arrow-hl" : "arrow"})`)
      .classed("wave", (l) => l.onPath)
      .on("mousemove", (event: MouseEvent, l) =>
        setTooltip({
          x: event.clientX, y: event.clientY,
          title: `${l.edge.callee} → ${l.edge.caller} (propagation)`,
          lines: [`${l.edge.n_calls} appels`].concat(
            l.edge.alpha_caller_to_callee != null ? [`α appelant→appelé = ${l.edge.alpha_caller_to_callee.toFixed(3)}`] : []),
        }))
      .on("mouseleave", () => setTooltip(null));

    const node = svg.append("g").selectAll<SVGGElement, SimNode>("g").data(simNodes).join("g").style("cursor", "pointer")
      .on("click", (_, n) => onSelect(n.id))
      .on("mousemove", (event: MouseEvent, n) =>
        setTooltip({ x: event.clientX, y: event.clientY, title: n.label, lines: [`a_i = ${n.score.toFixed(2)}`] }))
      .on("mouseleave", () => setTooltip(null));
    node.append("circle")
      .attr("r", radius)
      .attr("fill", (n) => scoreColor(n.score))
      .attr("stroke", (n) => (n.id === selected || pathSet.has(n.id) ? highlight : stroke))
      .attr("stroke-width", (n) => (n.id === selected ? 3.5 : 1.5));
    node.append("text").text((n) => n.label).attr("x", 0).attr("y", (n) => radius(n) + 13)
      .attr("text-anchor", "middle").attr("font-size", 11.5).attr("fill", text);

    const simulation = d3.forceSimulation(simNodes)
      .force("link", d3.forceLink<SimNode, SimLink>(simLinks).id((n) => n.id).distance(150))
      .force("charge", d3.forceManyBody().strength(-700))
      .force("collide", d3.forceCollide<SimNode>().radius((n) => radius(n) + 18))
      .force("center", d3.forceCenter(WIDTH / 2, HEIGHT / 2));

    node.call(
      d3.drag<SVGGElement, SimNode>()
        .on("start", (event, n) => { if (!event.active) simulation.alphaTarget(0.3).restart(); n.fx = n.x; n.fy = n.y; })
        .on("drag", (event, n) => { n.fx = event.x; n.fy = event.y; })
        .on("end", (event, n) => { if (!event.active) simulation.alphaTarget(0); n.fx = null; n.fy = null; }),
    );

    simulation.on("tick", () => {
      link.each(function (l) {
        const s = l.source as SimNode;
        const t = l.target as SimNode;
        const dx = t.x! - s.x!;
        const dy = t.y! - s.y!;
        const dist = Math.hypot(dx, dy) || 1;
        // Flèche arrêtée au bord du cercle cible
        const r = radius(t) + 3;
        d3.select(this).attr("x1", s.x!).attr("y1", s.y!).attr("x2", t.x! - (dx / dist) * r).attr("y2", t.y! - (dy / dist) * r);
      });
      node.attr("transform", (n) => `translate(${Math.max(20, Math.min(WIDTH - 20, n.x!))},${Math.max(20, Math.min(HEIGHT - 30, n.y!))})`);
    });
    return () => { simulation.stop(); };
  }, [nodes, edges, hasAttention, threshold, selected, path, onSelect]);

  return (
    <div>
      <svg ref={svgRef} viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" role="img" aria-label="Graphe de propagation causale" />
      <div className="legend">
        <span>flèche = sens de propagation d'un symptôme (appelé → appelant)</span>
        <span>{hasAttention ? "épaisseur = attention GAT α" : "épaisseur = volume d'appels (attention indisponible)"}</span>
        <span>taille/couleur = a_i · nœuds déplaçables</span>
      </div>
      {tooltip && (
        <div className="tooltip" style={{ left: tooltip.x + 12, top: tooltip.y + 12 }}>
          <b>{tooltip.title}</b>
          {tooltip.lines.map((line) => <div key={line}>{line}</div>)}
        </div>
      )}
    </div>
  );
}
