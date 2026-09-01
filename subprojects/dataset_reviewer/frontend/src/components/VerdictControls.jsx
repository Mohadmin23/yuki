const verdicts = [
  { value: "bad", label: "BAD", icon: "×", key: "A" },
  { value: "unsure", label: "UNSURE", icon: "?", key: "W" },
  { value: "fix", label: "FIX", icon: "✎", key: "E" },
  { value: "good", label: "GOOD", icon: "✓", key: "D" },
];

export default function VerdictControls({ selected, disabled, onChoose }) {
  return (
    <div className="verdict-controls" aria-label="Review verdict">
      {verdicts.map((verdict) => (
        <button
          key={verdict.value}
          type="button"
          className={`verdict-button verdict-${verdict.value} ${selected === verdict.value ? "selected" : ""}`}
          disabled={disabled}
          onClick={() => onChoose(verdict.value)}
        >
          <span className="verdict-icon">{verdict.icon}</span>
          <span>{verdict.label}</span>
          <kbd>{verdict.key}</kbd>
        </button>
      ))}
    </div>
  );
}
