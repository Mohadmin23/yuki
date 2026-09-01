const shortcuts = [
  ["A / ←", "Mark Bad"],
  ["D / →", "Mark Good"],
  ["W / ↑", "Mark Unsure"],
  ["E", "Mark for Fix"],
  ["N", "Focus reviewer note"],
  ["Enter", "Save pending decision"],
  ["Z", "Undo last decision"],
  ["J / K", "Next / previous sample"],
  ["?", "Toggle this overlay"],
];

export function isTypingTarget(target) {
  return target instanceof HTMLElement && (
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable
  );
}

export default function KeyboardShortcuts({ open, onClose }) {
  if (!open) return null;
  return (
    <div className="overlay" role="presentation" onMouseDown={onClose}>
      <section className="shortcut-modal" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div><span className="eyebrow">COMMAND MAP</span><h2>Keyboard shortcuts</h2></div>
          <button type="button" onClick={onClose}>CLOSE</button>
        </header>
        <div className="shortcut-grid">
          {shortcuts.map(([key, action]) => (
            <div key={key}><kbd>{key}</kbd><span>{action}</span></div>
          ))}
        </div>
        <p>Shortcuts pause automatically while you type in search, notes, or corrected JSON.</p>
      </section>
    </div>
  );
}
