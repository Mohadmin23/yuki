import { useState } from "react";

export default function MetadataPanel({ sample }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  async function copyRaw(event) {
    event.stopPropagation();
    await navigator.clipboard.writeText(JSON.stringify(sample.raw, null, 2));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  }

  return (
    <div className="metadata-panel">
      <div className="metadata-summary">
        <button className="text-button" type="button" onClick={() => setOpen((value) => !value)}>
          {open ? "Hide raw data" : "View raw data"}
        </button>
        <span>{sample.row_hash.slice(0, 10)}</span>
        <span>{Object.keys(sample.metadata || {}).length} extra fields</span>
      </div>
      {open && (
        <div className="raw-wrap">
          <button className="copy-button" type="button" onClick={copyRaw}>
            {copied ? "Copied" : "Copy JSON"}
          </button>
          <pre className="raw-json">{JSON.stringify(sample.raw, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
