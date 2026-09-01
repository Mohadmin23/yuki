import { useEffect, useState } from "react";
import { api } from "../api";

function relativeTime(timestamp) {
  const seconds = Math.max(0, Math.round(Date.now() / 1000 - timestamp));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(timestamp * 1000).toLocaleDateString();
}

export default function ReviewHistory({ revision, onOpenSample }) {
  const [events, setEvents] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    api.history().then((result) => alive && setEvents(result.events)).catch((reason) => {
      if (alive) setError(reason.message);
    });
    return () => { alive = false; };
  }, [revision]);

  return (
    <main className="screen-panel history-screen">
      <section className="dashboard-hero">
        <div><span className="eyebrow">AUDIT TRAIL</span><h2>Review history</h2><p>Every saved decision and revision, newest first.</p></div>
      </section>
      {error && <div className="inline-error">{error}</div>}
      {!events ? <div className="loading-block">READING EVENT LOG…</div> : events.length === 0 ? (
        <div className="empty-state compact"><span>NO SIGNAL</span><h3>No decisions yet</h3><p>History appears as soon as you review a sample.</p></div>
      ) : (
        <section className="history-list">
          {events.map((event) => (
            <button
              type="button"
              className={`history-row history-${event.review.verdict} ${event.undone ? "is-undone" : ""}`}
              key={event.event_id}
              onClick={() => onOpenSample(event.index)}
            >
              <span className="history-index">#{String(event.index + 1).padStart(4, "0")}</span>
              <strong>{event.review.verdict.toUpperCase()}</strong>
              <span className="history-reasons">{event.review.reasons.join(" · ") || "No reason tags"}</span>
              <span>rev {event.review.revision}</span>
              <time>{relativeTime(event.created_at)}</time>
              {event.undone && <em>UNDONE</em>}
            </button>
          ))}
        </section>
      )}
    </main>
  );
}
