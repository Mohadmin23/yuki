import { useState } from "react";

function renderContent(content) {
  if (typeof content === "string") return content || "(empty content)";
  return JSON.stringify(content, null, 2);
}

export default function ConversationMessage({ message, turn }) {
  const [expandedExtra, setExpandedExtra] = useState(false);
  const role = String(message.role || "unknown").toLowerCase();
  const hasExtra = message.extra && Object.keys(message.extra).length > 0;

  return (
    <article className={`message message-${role}`} data-role={role}>
      <header className="message-header">
        <span className="role-label">{role.toUpperCase()}</span>
        <span className="turn-label">{turn + 1}</span>
        {hasExtra && (
          <button
            className="text-button"
            type="button"
            onClick={() => setExpandedExtra((value) => !value)}
          >
            {expandedExtra ? "Hide details" : "Call details"}
          </button>
        )}
      </header>
      <pre className="message-content">{renderContent(message.content)}</pre>
      {hasExtra && expandedExtra && (
        <pre className="message-extra">{JSON.stringify(message.extra, null, 2)}</pre>
      )}
    </article>
  );
}
