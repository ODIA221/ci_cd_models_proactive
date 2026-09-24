import { useCallback, useEffect, useMemo, useState } from "react";
import { fetchRuns, fetchSignals, fetchWhatIf, jsonldUrl, reportUrl } from "./api";
import ForceGraph from "./components/ForceGraph";
import ModalParallel from "./components/ModalParallel";
import TimelineCanvas from "./components/TimelineCanvas";
import WhatIfPanel from "./components/WhatIfPanel";
import type { RunInfo, RunSignals, WhatIf } from "./types";

// Lien direct (section 4.3, étape 2): /ui/?run=<run_id>&service=<service>
function readUrl() {
  const params = new URLSearchParams(window.location.search);
  return { run: params.get("run"), service: params.get("service") };
}

export default function App() {
  const initial = useMemo(readUrl, []);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [runId, setRunId] = useState<string | null>(initial.run);
  const [signals, setSignals] = useState<RunSignals | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threshold, setThreshold] = useState(0.3);
  const [selected, setSelected] = useState<string | null>(initial.service);
  const [whatIf, setWhatIf] = useState<WhatIf | null>(null);
  const [pathIndex, setPathIndex] = useState(0);

  useEffect(() => {
    fetchRuns().then(setRuns).catch((e: Error) => setError(`Liste des runs indisponible : ${e.message}`));
  }, []);

  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    setSignals(null);
    setWhatIf(null);
    fetchSignals(runId)
      .then(setSignals)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [runId]);

  useEffect(() => {
    const params = new URLSearchParams();
    if (runId) params.set("run", runId);
    if (selected) params.set("service", selected);
    window.history.replaceState(null, "", `${window.location.pathname}?${params}`);
  }, [runId, selected]);

  useEffect(() => {
    if (!runId || !selected || !signals) return;
    setPathIndex(0);
    fetchWhatIf(runId, selected, threshold).then(setWhatIf).catch(() => setWhatIf(null));
  }, [runId, selected, threshold, signals]);

  const onSelect = useCallback((service: string) => setSelected((cur) => (cur === service ? null : service)), []);

  const visibleNodes = useMemo(
    () => (signals ? signals.nodes.filter((n) => n.anomaly_score >= threshold || n.service === selected) : []),
    [signals, threshold, selected],
  );
  const labels = useMemo(() => new Map(signals?.nodes.map((n) => [n.service, n.display_name]) ?? []), [signals]);
  const path = whatIf?.paths[pathIndex] ?? null;
  const missing = signals ? Object.entries(signals.modalities_available).filter(([, ok]) => !ok).map(([m]) => m) : [];

  return (
    <>
      <header>
        <h1>LogPipeGuard v2 — exploration causale</h1>
        <p>Chronologie, propagation et corrélation modale des signaux d'un run RCAEval. Les nœuds sont des services, pas des étapes CI/CD.</p>
      </header>
      <main>
        <div className="controls">
          <label>
            Run
            <select value={runId ?? ""} onChange={(e) => { setRunId(e.target.value || null); setSelected(null); }}>
              <option value="">— choisir —</option>
              {runs.map((r) => (
                <option key={r.run_id} value={r.run_id}>
                  {r.anormal ? "⚠ " : ""}{r.run_id}{r.signaux_en_cache ? "" : " (calcul ~1-2 min)"}
                </option>
              ))}
            </select>
          </label>
          <label>
            Seuil a_i ≥ {threshold.toFixed(2)} (filtre les 3 vues)
            <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={(e) => setThreshold(+e.target.value)} />
          </label>
          <label>
            Service sélectionné (lié entre les vues)
            <select value={selected ?? ""} onChange={(e) => setSelected(e.target.value || null)} disabled={!signals}>
              <option value="">—</option>
              {signals?.nodes.map((n) => <option key={n.service} value={n.service}>{n.display_name}</option>)}
            </select>
          </label>
          {runId && signals && (
            <>
              <a className="btn" href={reportUrl(runId, selected)} download={`diagnostic_${runId.replaceAll("/", "_")}.md`}>Exporter le rapport (Markdown)</a>
              <a className="btn" href={jsonldUrl(runId)} download={`diagnostic_${runId.replaceAll("/", "_")}.jsonld`}>Exporter (JSON-LD)</a>
            </>
          )}
        </div>

        {error && <div className="caveat">Erreur : {error}</div>}
        {loading && <p>Calcul des signaux (relecture des logs/métriques/traces bruts au premier appel, puis cache)…</p>}
        {!runId && !error && <p>Choisir un run. Un run marqué ⚠ contient une faute injectée ; un run normal montre les fausses pistes que les vues peuvent suggérer.</p>}

        {signals && (
          <>
            <div className="caveat">
              <b>Avant d'interpréter :</b> a_i est une heuristique (écart robuste par modalité) ; l'attention GAT est
              {signals.attention ? " associative (auto-encodeur entraîné sur des graphes normaux), pas causale" : " indisponible pour ce run"} ; β est
              un rapport de taux de templates, pas une attention.
              {missing.length > 0 && <> Modalités absentes : <b>{missing.join(", ")}</b> (rien n'est extrapolé).</>}
            </div>

            <section className="panel">
              <h2>Chronologie — première déviation par service</h2>
              <p className="sub">Secondes après le début de la fenêtre analysée ({signals.window.kind === "post_injection" ? "injection de la faute" : "2e moitié d'une fenêtre normale"}). Cliquer un glyphe pour le sélectionner.</p>
              <TimelineCanvas nodes={visibleNodes} probeSeconds={signals.window.probe_seconds} selected={selected} onSelect={onSelect} />
            </section>

            <div className="grid2">
              <section className="panel">
                <h2>Propagation</h2>
                <p className="sub">Graphe d'appels observé dans les traces. Cliquer un nœud pour tester l'hypothèse « cause racine ».</p>
                {signals.edges.length === 0 ? (
                  <p>Pas de traces pour ce run (ex. Sock Shop dans RCAEval) : graphe indisponible.</p>
                ) : (
                  <ForceGraph nodes={signals.nodes} edges={signals.edges} hasAttention={signals.attention !== null}
                    threshold={threshold} selected={selected} path={path} onSelect={onSelect} />
                )}
              </section>
              <section className="panel">
                <h2>Corrélation modale</h2>
                <p className="sub">Contribution de chaque modalité à a_i, par service.</p>
                <ModalParallel nodes={visibleNodes} selected={selected} onSelect={onSelect} />
              </section>
            </div>

            {whatIf && selected && (
              <WhatIfPanel whatIf={whatIf} node={signals.nodes.find((n) => n.service === selected)} labels={labels}
                pathIndex={pathIndex} onPathIndex={setPathIndex} />
            )}

            <details className="panel">
              <summary>Provenance détaillée des signaux</summary>
              <ul>{Object.entries(signals.provenance).map(([k, v]) => <li key={k}><b>{k}</b> : {v}</li>)}</ul>
            </details>
          </>
        )}
      </main>
    </>
  );
}
