function Stat({ label, value, tone }) {
  return (
    <div className={`header-stat ${tone || ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default function ProgressHeader({ dataset, sample, view, onView }) {
  const counts = dataset.counts;
  return (
    <header className="progress-header">
      <div className="brand-block">
        <div>
          <span className="eyebrow">Yuki Review</span>
          <h1>{dataset.name.replace(/\.jsonl$/i, "")}</h1>
        </div>
      </div>
      <nav className="view-tabs" aria-label="Application sections">
        {[
          ["review", "Review"],
          ["dashboard", "Stats"],
          ["history", "History"],
        ].map(([value, label]) => (
          <button
            type="button"
            key={value}
            className={view === value ? "active" : ""}
            onClick={() => onView(value)}
          >
            {label}
          </button>
        ))}
      </nav>
      <div className="header-stats">
        <Stat label="Sample" value={sample ? `${sample.number} / ${dataset.total}` : `— / ${dataset.total}`} />
        <Stat label="Reviewed" value={`${counts.reviewed}`} tone="tone-blue" />
        <Stat label="Progress" value={`${counts.percent.toFixed(1)}%`} tone="tone-red" />
      </div>
      <div className="progress-track" aria-label={`${counts.percent}% reviewed`}>
        <span style={{ width: `${counts.percent}%` }} />
      </div>
    </header>
  );
}
