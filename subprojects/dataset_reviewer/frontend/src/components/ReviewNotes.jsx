import { forwardRef } from "react";

const ReviewNotes = forwardRef(function ReviewNotes({
  verdict,
  categories,
  selectedReasons,
  note,
  correctedText,
  autosaveStatus,
  disabled,
  error,
  onToggleReason,
  onNote,
  onCorrected,
  onUseOriginal,
  onClose,
  onSave,
}, noteRef) {
  return (
    <section className={`review-notes notes-${verdict}`}>
      <header className="notes-header">
        <div>
          <span className="eyebrow">{verdict === "fix" ? "Needs a fix" : "Mark as bad"}</span>
          <h2>{verdict === "fix" ? "What should be corrected?" : "Why is this sample bad?"}</h2>
        </div>
        <div className="notes-header-actions">
          <span className={`autosave-status status-${autosaveStatus}`}>{autosaveStatus}</span>
          <button className="close-notes" type="button" onClick={onClose} aria-label="Close notes">×</button>
        </div>
      </header>
      <div className="reason-grid">
        {categories.map((reason) => (
          <button
            type="button"
            key={reason}
            disabled={disabled}
            className={`reason-chip ${selectedReasons.includes(reason) ? "selected" : ""}`}
            onClick={() => onToggleReason(reason)}
          >
            {reason}
          </button>
        ))}
      </div>
      <label className="field-block">
        <span>Reviewer note <kbd>N</kbd></span>
        <textarea
          ref={noteRef}
          disabled={disabled}
          value={note}
          onChange={(event) => onNote(event.target.value)}
          placeholder="What failed, and what would make it useful?"
          rows={4}
        />
      </label>
      {verdict === "fix" && (
        <div className="field-block">
          <span className="field-heading">
            <label htmlFor="corrected-sample">Corrected sample · complete JSON object</label>
            <button className="text-button" type="button" disabled={disabled} onClick={onUseOriginal}>
              Start from original
            </button>
          </span>
          <textarea
            id="corrected-sample"
            disabled={disabled}
            className="code-editor"
            value={correctedText}
            onChange={(event) => onCorrected(event.target.value)}
            rows={13}
            spellCheck="false"
          />
        </div>
      )}
      {error && <div className="inline-error">{error}</div>}
      <div className="notes-footer">
        <span>Enter saves and moves on</span>
        <button className={`commit-button commit-${verdict}`} type="button" disabled={disabled} onClick={onSave}>
          Save {verdict} · Next
        </button>
      </div>
    </section>
  );
});

export default ReviewNotes;
