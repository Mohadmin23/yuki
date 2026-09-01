import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import DatasetCard from "./components/DatasetCard";
import DatasetNavigator from "./components/DatasetNavigator";
import KeyboardShortcuts, { isTypingTarget } from "./components/KeyboardShortcuts";
import ProgressHeader from "./components/ProgressHeader";
import ReviewDashboard from "./components/ReviewDashboard";
import ReviewHistory from "./components/ReviewHistory";
import ReviewNotes from "./components/ReviewNotes";
import VerdictControls from "./components/VerdictControls";

const pause = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [sample, setSample] = useState(null);
  const [view, setView] = useState("review");
  const [filter, setFilter] = useState("unreviewed");
  const [search, setSearch] = useState("");
  const [appliedSearch, setAppliedSearch] = useState("");
  const [jump, setJump] = useState("");
  const [pendingVerdict, setPendingVerdict] = useState(null);
  const [editorOpen, setEditorOpen] = useState(false);
  const [selectedReasons, setSelectedReasons] = useState([]);
  const [note, setNote] = useState("");
  const [correctedText, setCorrectedText] = useState("");
  const [pendingChanges, setPendingChanges] = useState(false);
  const [autosaveStatus, setAutosaveStatus] = useState("ready");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [exitDirection, setExitDirection] = useState(null);
  const [error, setError] = useState("");
  const [fatalError, setFatalError] = useState("");
  const [toast, setToast] = useState("");
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [revision, setRevision] = useState(0);

  const noteRef = useRef(null);
  const loadedAt = useRef(performance.now());
  const draftNeedsSave = useRef(false);
  const actionLock = useRef(false);
  const stateRef = useRef({});
  const sampleCache = useRef(new Map());

  stateRef.current = {
    sample, pendingVerdict, selectedReasons, note, correctedText,
    pendingChanges, editorOpen, filter, appliedSearch, busy, view,
  };

  const flash = useCallback((message) => {
    setToast(message);
    window.setTimeout(() => setToast((current) => current === message ? "" : current), 1900);
  }, []);

  function markChanged() {
    draftNeedsSave.current = true;
    setPendingChanges(true);
    setAutosaveStatus("pending");
  }

  const prefetchAround = useCallback((nextSample) => {
    if (!nextSample) return;
    for (const index of [nextSample.index - 1, nextSample.index + 1]) {
      if (index < 0 || index >= nextSample.total || sampleCache.current.has(index)) continue;
      if (sampleCache.current.size >= 12) {
        sampleCache.current.delete(sampleCache.current.keys().next().value);
      }
      const request = api.sample(index).catch(() => null);
      sampleCache.current.set(index, request);
    }
  }, []);

  const displaySample = useCallback((nextSample) => {
    setSample(nextSample);
    setError("");
    setExitDirection(null);
    const draft = nextSample?.draft;
    const review = nextSample?.review;
    setPendingVerdict(draft?.verdict ?? review?.verdict ?? null);
    setEditorOpen(Boolean(draft));
    setSelectedReasons(draft?.reasons ?? review?.reasons ?? []);
    setNote(draft?.note ?? review?.note ?? "");
    setCorrectedText(
      draft?.corrected_text
      ?? (review?.corrected_content ? JSON.stringify(review.corrected_content, null, 2) : ""),
    );
    setPendingChanges(Boolean(draft));
    draftNeedsSave.current = false;
    setAutosaveStatus(draft ? "restored" : "ready");
    loadedAt.current = performance.now();
    if (nextSample) prefetchAround(nextSample);
  }, [prefetchAround]);

  const persistDraft = useCallback(async () => {
    const current = stateRef.current;
    if (!draftNeedsSave.current || !current.sample) return;
    setAutosaveStatus("saving");
    const sampleId = current.sample.id;
    await api.saveDraft({
      sample_id: sampleId,
      verdict: ["bad", "fix"].includes(current.pendingVerdict) ? current.pendingVerdict : null,
      reasons: current.selectedReasons,
      note: current.note,
      corrected_text: current.correctedText,
    });
    if (stateRef.current.sample?.id === sampleId) {
      draftNeedsSave.current = false;
      setAutosaveStatus("saved");
    }
  }, []);

  const findSample = useCallback(async ({
    current = stateRef.current.sample?.index ?? -1,
    direction = "next",
    verdict = stateRef.current.filter,
    query = stateRef.current.appliedSearch,
    saveDraft = true,
  } = {}) => {
    if (saveDraft) {
      try {
        await persistDraft();
      } catch (reason) {
        setError(`Draft could not be saved: ${reason.message}`);
        return;
      }
    }
    setLoading(true);
    try {
      const result = await api.navigate({ current, direction, verdict, search: query });
      displaySample(result.sample);
      if (result.wrapped && result.sample) flash("Wrapped around the matching set");
    } catch (reason) {
      setError(reason.message);
    } finally {
      setLoading(false);
    }
  }, [displaySample, flash, persistDraft]);

  const loadIndex = useCallback(async (index, { saveDraft = true } = {}) => {
    if (saveDraft) {
      try {
        await persistDraft();
      } catch (reason) {
        setError(`Draft could not be saved: ${reason.message}`);
        return;
      }
    }
    setLoading(true);
    try {
      let request = sampleCache.current.get(index);
      if (!request) request = api.sample(index);
      const nextSample = await request;
      if (!nextSample) throw new Error("That sample could not be loaded");
      sampleCache.current.delete(index);
      displaySample(nextSample);
    } catch (reason) {
      setError(reason.message);
    } finally {
      setLoading(false);
    }
  }, [displaySample, persistDraft]);

  useEffect(() => {
    let alive = true;
    api.dataset().then(async (value) => {
      if (!alive) return;
      setDataset(value);
      const result = await api.navigate({ current: -1, direction: "next", verdict: "unreviewed", search: "" });
      if (!alive) return;
      displaySample(result.sample);
      setLoading(false);
    }).catch((reason) => {
      if (alive) {
        setFatalError(reason.message);
        setLoading(false);
      }
    });
    return () => { alive = false; };
  }, [displaySample]);

  useEffect(() => {
    const timeout = window.setTimeout(() => setAppliedSearch(search.trim()), 320);
    return () => window.clearTimeout(timeout);
  }, [search]);

  const initialSearch = useRef(true);
  useEffect(() => {
    if (!dataset) return;
    if (initialSearch.current) {
      initialSearch.current = false;
      return;
    }
    findSample({ current: -1, direction: "next", query: appliedSearch });
  }, [appliedSearch, dataset, findSample]);

  useEffect(() => {
    if (!sample || !draftNeedsSave.current) return undefined;
    const sampleId = sample.id;
    const timeout = window.setTimeout(() => {
      if (stateRef.current.sample?.id !== sampleId) return;
      persistDraft().catch((reason) => setError(`Draft could not be saved: ${reason.message}`));
    }, 450);
    return () => window.clearTimeout(timeout);
  }, [sample, pendingVerdict, selectedReasons, note, correctedText, persistDraft]);

  const commitDecision = useCallback(async (verdict) => {
    const current = stateRef.current;
    if (!current.sample || current.busy || actionLock.current || !verdict) return;
    actionLock.current = true;
    setBusy(true);
    setError("");
    const keepsDetails = verdict === "bad" || verdict === "fix";
    try {
      const result = await api.saveReview({
        sample_id: current.sample.id,
        verdict,
        reasons: keepsDetails ? current.selectedReasons : [],
        note: keepsDetails ? current.note : "",
        corrected_text: verdict === "fix" ? current.correctedText : "",
        duration_ms: Math.round(performance.now() - loadedAt.current),
      });
      setDataset((value) => ({ ...value, counts: result.counts }));
      sampleCache.current.clear();
      draftNeedsSave.current = false;
      setPendingChanges(false);
      setEditorOpen(false);
      setAutosaveStatus("saved");
      setRevision((value) => value + 1);
      setExitDirection({ good: "right", bad: "left", unsure: "up", fix: "down" }[verdict]);
      await pause(220);
      await findSample({ current: current.sample.index, direction: "next", saveDraft: false });
    } catch (reason) {
      setExitDirection(null);
      setError(reason.message);
    } finally {
      setBusy(false);
      actionLock.current = false;
    }
  }, [findSample]);

  const chooseVerdict = useCallback((verdict) => {
    if (verdict === "good" || verdict === "unsure") {
      commitDecision(verdict);
      return;
    }
    const existingReview = stateRef.current.sample?.review;
    const isOpeningSavedReview = existingReview?.verdict === verdict
      && !stateRef.current.sample?.draft;
    setPendingVerdict(verdict);
    setEditorOpen(true);
    if (!isOpeningSavedReview) markChanged();
  }, [commitDecision]);

  const move = useCallback((direction) => {
    if (stateRef.current.busy || actionLock.current) return;
    findSample({ direction });
  }, [findSample]);

  const undo = useCallback(async () => {
    if (stateRef.current.busy || actionLock.current) return;
    actionLock.current = true;
    try {
      await persistDraft();
      setBusy(true);
      const result = await api.undo();
      setDataset((value) => ({ ...value, counts: result.counts }));
      setRevision((value) => value + 1);
      sampleCache.current.clear();
      if (result.undone) {
        await loadIndex(result.undone.index, { saveDraft: false });
        flash("Last decision undone");
      } else {
        flash("Nothing to undo");
      }
    } catch (reason) {
      setError(reason.message);
    } finally {
      setBusy(false);
      actionLock.current = false;
    }
  }, [flash, loadIndex, persistDraft]);

  const changeFilter = useCallback(async (nextFilter) => {
    setFilter(nextFilter);
    stateRef.current.filter = nextFilter;
    await findSample({ current: -1, verdict: nextFilter });
  }, [findSample]);

  const changeView = useCallback(async (nextView) => {
    if (actionLock.current) return;
    try {
      await persistDraft();
      setView(nextView);
    } catch (reason) {
      setError(`Draft could not be saved: ${reason.message}`);
    }
  }, [persistDraft]);

  const openFromSecondaryView = useCallback(async (index) => {
    setView("review");
    if (Number.isInteger(index)) await loadIndex(index);
  }, [loadIndex]);

  useEffect(() => {
    function onKeyDown(event) {
      if (isTypingTarget(event.target)) return;
      if (event.key === "?") {
        event.preventDefault();
        setShortcutsOpen((value) => !value);
        return;
      }
      if (shortcutsOpen && event.key === "Escape") {
        setShortcutsOpen(false);
        return;
      }
      if (stateRef.current.view !== "review" || stateRef.current.busy) return;
      const key = event.key.toLowerCase();
      if (["arrowleft", "arrowright", "arrowup", "a", "d", "w", "e", "n", "enter", "z", "j", "k"].includes(key)) {
        event.preventDefault();
      }
      if (key === "a" || key === "arrowleft") chooseVerdict("bad");
      else if (key === "d" || key === "arrowright") chooseVerdict("good");
      else if (key === "w" || key === "arrowup") chooseVerdict("unsure");
      else if (key === "e") chooseVerdict("fix");
      else if (key === "n") {
        if (!["bad", "fix"].includes(stateRef.current.pendingVerdict)) {
          setPendingVerdict("bad");
          markChanged();
        }
        setEditorOpen(true);
        window.setTimeout(() => noteRef.current?.focus(), 0);
      } else if (key === "enter") {
        if (stateRef.current.pendingChanges && stateRef.current.pendingVerdict) {
          commitDecision(stateRef.current.pendingVerdict);
        } else move("next");
      } else if (key === "z") undo();
      else if (key === "j") move("next");
      else if (key === "k") move("previous");
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [chooseVerdict, commitDecision, move, shortcutsOpen, undo]);

  function toggleReason(reason) {
    setSelectedReasons((values) => values.includes(reason)
      ? values.filter((value) => value !== reason)
      : [...values, reason]);
    markChanged();
  }

  async function jumpTo(event) {
    event.preventDefault();
    const number = Number(jump);
    if (!Number.isInteger(number) || number < 1 || number > dataset.total) {
      setError(`Enter a sample number from 1 to ${dataset.total}`);
      return;
    }
    await loadIndex(number - 1);
    setJump("");
  }

  async function refreshAfterClear() {
    const value = await api.dataset();
    setDataset(value);
    setRevision((current) => current + 1);
    sampleCache.current.clear();
    setFilter("unreviewed");
    stateRef.current.filter = "unreviewed";
    await findSample({ current: -1, verdict: "unreviewed", saveDraft: false });
    flash("Review database cleared");
  }

  if (fatalError) {
    return <div className="fatal-screen"><span>Yuki Review</span><h1>Could not open the dataset</h1><p>{fatalError}</p></div>;
  }
  if (!dataset) {
    return <div className="boot-screen"><div className="boot-mark">Y</div><span>Opening dataset…</span><i /></div>;
  }

  return (
    <div className="app-shell">
      <ProgressHeader dataset={dataset} sample={sample} view={view} onView={changeView} />

      {view === "review" && (
        <main className="review-screen">
          <DatasetNavigator
            filter={filter}
            search={search}
            jump={jump}
            disabled={busy || loading}
            onFilter={changeFilter}
            onSearch={setSearch}
            onJumpChange={setJump}
            onJump={jumpTo}
            onPrevious={() => move("previous")}
            onNext={() => move("next")}
            onUndo={undo}
            onShortcuts={() => setShortcutsOpen(true)}
          />

          <section className="review-workspace">
            {loading && !sample ? <div className="loading-block">FETCHING SAMPLE…</div> : sample ? (
              <>
                <DatasetCard key={sample.id} sample={sample} exitDirection={exitDirection} onSwipe={commitDecision} />
                {editorOpen && ["bad", "fix"].includes(pendingVerdict) && (
                  <ReviewNotes
                    ref={noteRef}
                    verdict={pendingVerdict}
                    categories={dataset.reason_categories}
                    selectedReasons={selectedReasons}
                    note={note}
                    correctedText={correctedText}
                    autosaveStatus={autosaveStatus}
                    disabled={busy}
                    error={error}
                    onToggleReason={toggleReason}
                    onNote={(value) => { setNote(value); markChanged(); }}
                    onCorrected={(value) => { setCorrectedText(value); markChanged(); }}
                    onUseOriginal={() => { setCorrectedText(JSON.stringify(sample.raw, null, 2)); markChanged(); }}
                    onClose={() => setEditorOpen(false)}
                    onSave={() => commitDecision(pendingVerdict)}
                  />
                )}
                {error && !(editorOpen && ["bad", "fix"].includes(pendingVerdict)) && <div className="inline-error floating-error">{error}</div>}
              </>
            ) : (
              <div className="empty-state">
                <span>QUEUE CLEAR</span>
                <h2>No samples match this view</h2>
                <p>Change the filter or search query to keep reviewing.</p>
                <button type="button" onClick={() => changeFilter("all")}>SHOW ALL SAMPLES</button>
              </div>
            )}
          </section>

          <footer className="review-footer">
            <VerdictControls selected={pendingVerdict} disabled={busy || loading || !sample} onChoose={chooseVerdict} />
          </footer>
        </main>
      )}

      {view === "dashboard" && (
        <ReviewDashboard dataset={dataset} revision={revision} onOpenSample={openFromSecondaryView} onCleared={refreshAfterClear} />
      )}
      {view === "history" && <ReviewHistory revision={revision} onOpenSample={openFromSecondaryView} />}

      <KeyboardShortcuts open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
