import { useRef, useState } from "react";
import ConversationMessage from "./ConversationMessage";
import MetadataPanel from "./MetadataPanel";

export default function DatasetCard({ sample, exitDirection, onSwipe }) {
  const origin = useRef(null);
  const [dragX, setDragX] = useState(0);
  const [dragging, setDragging] = useState(false);

  function pointerDown(event) {
    if (event.button !== 0) return;
    origin.current = { x: event.clientX, pointerId: event.pointerId };
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
  }

  function pointerMove(event) {
    if (!origin.current || origin.current.pointerId !== event.pointerId) return;
    setDragX(Math.max(-180, Math.min(180, event.clientX - origin.current.x)));
  }

  function pointerUp(event) {
    if (!origin.current || origin.current.pointerId !== event.pointerId) return;
    const completedX = dragX;
    origin.current = null;
    setDragging(false);
    if (completedX > 105) onSwipe("good");
    else if (completedX < -105) onSwipe("bad");
    setDragX(0);
  }

  const exitTransforms = {
    right: "translate3d(115vw, 20px, 0) rotate(12deg)",
    left: "translate3d(-115vw, 20px, 0) rotate(-12deg)",
    up: "translate3d(0, -115vh, 0) rotate(-2deg)",
    down: "translate3d(0, 115vh, 0) scale(.92)",
  };
  const transform = exitDirection
    ? exitTransforms[exitDirection]
    : `translate3d(${dragX}px, 0, 0) rotate(${dragX / 42}deg)`;

  return (
    <section
      className={`dataset-card ${dragging ? "is-dragging" : ""} ${exitDirection ? "is-exiting" : ""}`}
      style={{ transform }}
      onPointerDown={pointerDown}
      onPointerMove={pointerMove}
      onPointerUp={pointerUp}
      onPointerCancel={pointerUp}
      aria-label={`Dataset sample ${sample.number}`}
    >
      <div className={`swipe-stamp stamp-good ${dragX > 35 ? "visible" : ""}`}>GOOD</div>
      <div className={`swipe-stamp stamp-bad ${dragX < -35 ? "visible" : ""}`}>BAD</div>
      <div className="sample-id-row">
        <span>Sample {sample.number} of {sample.total}</span>
        <span>{sample.format}</span>
        <span>{sample.id.slice(0, 8)}</span>
      </div>
      <div className="conversation-stack">
        {sample.messages.length > 0 ? sample.messages.map((message, index) => (
          <ConversationMessage key={`${sample.id}-${index}`} message={message} turn={index} />
        )) : (
          <article className="message message-unknown">
            <header className="message-header"><span className="role-label">GENERIC RECORD</span></header>
            <pre className="message-content">{JSON.stringify(sample.raw, null, 2)}</pre>
          </article>
        )}
      </div>
      <MetadataPanel sample={sample} />
    </section>
  );
}
