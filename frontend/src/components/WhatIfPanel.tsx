import type { ServiceNode, WhatIf } from "../types";

interface Props {
  whatIf: WhatIf;
  node: ServiceNode | undefined;
  labels: Map<string, string>;
  pathIndex: number;
  onPathIndex: (i: number) => void;
}

export default function WhatIfPanel({ whatIf, node, labels, pathIndex, onPathIndex }: Props) {
  const name = (s: string) => labels.get(s) ?? s;
  return (
    <div className="panel">
      <h2>Et si la cause racine était « {name(whatIf.hypothesis)} » ?</h2>
      <p className="sub">{whatIf.caveat} Test structurel, pas contrefactuel.</p>
      <div className="metrics-row">
        <div className="metric"><div className="v">{whatIf.hypothesis_anomaly_score?.toFixed(2) ?? "—"}</div><div className="l">a_i de l'hypothèse</div></div>
        <div className="metric"><div className="v">{whatIf.explained_anomalous.length}</div><div className="l">services anormaux expliqués</div></div>
        <div className="metric"><div className="v">{whatIf.coverage === null ? "—" : `${Math.round(whatIf.coverage * 100)} %`}</div><div className="l">couverture</div></div>
      </div>
      {whatIf.unexplained_anomalous.length > 0 && (
        <div className="caveat">Non expliqués par cette hypothèse : {whatIf.unexplained_anomalous.map(name).join(", ")}</div>
      )}
      {whatIf.paths.length > 0 && (
        <label className="controls">
          <span>Chemin de propagation surligné dans le graphe (appelé → appelants)</span>
          <select value={pathIndex} onChange={(e) => onPathIndex(+e.target.value)}>
            {whatIf.paths.map((p, i) => <option key={i} value={i}>{p.map(name).join(" → ")}</option>)}
          </select>
        </label>
      )}
      {node?.logs && node.logs.top_templates.length > 0 && (
        <>
          <h2 style={{ marginTop: 12 }}>Templates de logs les plus modifiés</h2>
          <p className="sub">β = log2 du rapport des taux après/avant (substitut : le modèle n'a pas d'attention sur les logs)</p>
          <table>
            <thead><tr><th>template</th><th>β</th><th>avant</th><th>après</th></tr></thead>
            <tbody>
              {node.logs.top_templates.map((t) => (
                <tr key={t.template}><td>{t.template}</td><td>{t.beta.toFixed(2)}</td><td>{t.n_base}</td><td>{t.n_probe}</td></tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
