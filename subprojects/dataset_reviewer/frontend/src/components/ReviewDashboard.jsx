import { useEffect, useState } from "react";
import { api } from "../api";

function formatDuration(milliseconds) {
  if (!milliseconds) return "—";
  if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
  return `${(milliseconds / 1000).toFixed(1)} s`;
}

export default function ReviewDashboard({ dataset, revision, onOpenSample, onCleared }) {
  const [dashboard, setDashboard] = useState(null);
  const [exportResult, setExportResult] = useState(null);
  const [confirmation, setConfirmation] = useState("");
  const [dangerOpen, setDangerOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    api.dashboard().then((value) => alive && setDashboard(value)).catch((reason) => {
      if (alive) setError(reason.message);
    });
    return () => { alive = false; };
  }, [revision]);

  async function exportAll() {
    setBusy(true);
    setError("");
    try {
      setExportResult(await api.exports());
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  }

  async function clearAll() {
    setBusy(true);
    setError("");
    try {
      await api.clearReviews(confirmation);
      setDangerOpen(false);
      setConfirmation("");
      setExportResult(null);
      onCleared();
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
    }
  }

  if (!dashboard) {
    return <main className="screen-panel dashboard-screen"><div className="loading-block">CALCULATING REVIEW TELEMETRY…</div></main>;
  }

  const counts = dashboard.counts;
  const reviewed = Math.max(1, counts.reviewed);
  const cards = [
    ["GOOD", counts.good, "good"],
    ["BAD", counts.bad, "bad"],
    ["FIX", counts.fix, "fix"],
    ["UNSURE", counts.unsure, "unsure"],
  ];

  return (
    <main className="screen-panel dashboard-screen">
      <section className="dashboard-hero">
        <div>
          <span className="eyebrow">CURATION TELEMETRY</span>
          <h2>{counts.reviewed} of {counts.total} samples reviewed</h2>
          <p>{counts.remaining} remain · average decision {formatDuration(dashboard.average_review_ms)}</p>
        </div>
        <button className="primary-action" type="button" onClick={() => onOpenSample(null)}>
          RETURN TO REVIEW
        </button>
      </section>

      <section className="metric-grid">
        {cards.map(([label, value, tone]) => (
          <article className={`metric-card metric-${tone}`} key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
            <div className="metric-track"><i style={{ width: `${(value / reviewed) * 100}%` }} /></div>
            <small>{counts.reviewed ? ((value / counts.reviewed) * 100).toFixed(1) : "0.0"}% of reviewed</small>
          </article>
        ))}
      </section>

      <div className="dashboard-columns">
        <section className="lab-panel distribution-panel">
          <header><span className="eyebrow">VERDICT DISTRIBUTION</span><strong>{counts.percent.toFixed(1)}%</strong></header>
          <div className="distribution-bar" aria-label="Verdict distribution">
            {cards.map(([, value, tone]) => value > 0 && (
              <span key={tone} className={`dist-${tone}`} style={{ width: `${(value / reviewed) * 100}%` }} />
            ))}
          </div>
          <div className="legend-row">
            {cards.map(([label, value, tone]) => <span key={tone}><i className={`legend-${tone}`} />{label} {value}</span>)}
          </div>
          <div className="session-card">
            <span>THIS SESSION</span>
            <strong>{dashboard.session.decisions} decisions</strong>
            <small>{formatDuration(dashboard.session.review_duration_ms)} captured review time</small>
          </div>
        </section>

        <section className="lab-panel reasons-panel">
          <header><span className="eyebrow">MOST COMMON ISSUES</span></header>
          {dashboard.reason_counts.length ? dashboard.reason_counts.slice(0, 8).map((item) => (
            <div className="reason-row" key={item.reason}>
              <span>{item.reason}</span><strong>{item.count}</strong>
              <i style={{ width: `${(item.count / dashboard.reason_counts[0].count) * 100}%` }} />
            </div>
          )) : <p className="muted-copy">No issue categories recorded yet.</p>}
        </section>
      </div>

      <section className="lab-panel export-panel">
        <div>
          <span className="eyebrow">EXPORT BAY</span>
          <h3>Build safe, schema-preserving exports</h3>
          <p>Creates verdict JSONL files, a corrected training set, and a full audit report. Source remains untouched.</p>
        </div>
        <button className="primary-action" type="button" disabled={busy} onClick={exportAll}>
          {busy ? "WORKING…" : "GENERATE EXPORTS"}
        </button>
      </section>
      {exportResult && (
        <section className="export-results">
          {Object.entries(exportResult.files).map(([name, file]) => (
            <a key={name} href={file.url} download>
              <span>{name.toUpperCase()}</span><strong>{file.records} records</strong><small>{file.filename}</small>
            </a>
          ))}
        </section>
      )}

      <section className="danger-zone">
        {!dangerOpen ? (
          <button type="button" className="text-button danger-link" onClick={() => setDangerOpen(true)}>
            CLEAR REVIEW DATABASE…
          </button>
        ) : (
          <div className="danger-confirm">
            <p>Type dataset fingerprint <code>{dataset.clear_confirmation}</code> to clear decisions, drafts, and history.</p>
            <input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder={dataset.clear_confirmation} />
            <button type="button" disabled={busy || confirmation !== dataset.clear_confirmation} onClick={clearAll}>CONFIRM CLEAR</button>
            <button type="button" onClick={() => setDangerOpen(false)}>CANCEL</button>
          </div>
        )}
      </section>
      {error && <div className="inline-error">{error}</div>}
    </main>
  );
}
