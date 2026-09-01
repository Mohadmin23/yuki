# Yuki Dataset Review Lab

A local, keyboard-first review application for curating Yuki training data one sample at a time. It reads the source JSONL by byte offset, stores every decision in a separate SQLite database, and never calls a model or a Yuki tool.

## Launch

From the repository root:

```bash
subprojects/dataset_reviewer/run.sh
```

Then open [http://127.0.0.1:7870](http://127.0.0.1:7870).

The checked-in production UI is already built. To rebuild it after frontend changes:

```bash
cd subprojects/dataset_reviewer/frontend
npm install
npm run build
```

## Defaults

- Source dataset: `subprojects/finetune/datasets/yuki_clean_v4.jsonl`
- Review database: `subprojects/dataset_reviewer/data/reviews.sqlite3`
- Exports: `subprojects/dataset_reviewer/exports/`
- Local port: `7870`

These can be overridden without changing code:

```bash
YUKI_REVIEW_DATASET=/absolute/path/to/data.jsonl \
YUKI_REVIEW_DB=/absolute/path/to/reviews.sqlite3 \
YUKI_REVIEW_EXPORTS=/absolute/path/to/exports \
YUKI_REVIEW_PORT=7871 \
subprojects/dataset_reviewer/run.sh
```

## Review workflow

- `A` / left: Bad
- `D` / right: Good
- `W` / up: Unsure
- `E`: Fix
- `N`: focus reviewer note
- `Enter`: save a pending Bad/Fix decision, otherwise move next
- `Z`: undo the last saved decision
- `J` / `K`: next / previous
- `?`: shortcut map

Global shortcuts pause while an input, selector, note, or correction editor has focus. Bad and Fix drafts autosave before navigation and restore after restart.

## Data integrity

The original JSONL is opened for indexed reads only. Its SHA-256 fingerprint identifies the dataset, and each reviewable row receives a deterministic ID derived from:

```text
dataset fingerprint + original row index + row hash
```

Including the row index is intentional: exact duplicate rows remain separate review tasks. Reviews, drafts, revisions, and the undo event log live in SQLite with WAL journaling and full synchronous commits. Export files are written to temporary files, flushed, and atomically replaced.

## Exports

`Generate exports` creates:

- accepted/good JSONL
- rejected/bad JSONL
- needs-fix JSONL containing the untouched originals
- unsure JSONL
- corrected JSONL containing Good originals plus approved Fix edits
- a complete JSON audit report with IDs, row hashes, verdicts, notes, reasons, timing, and revisions

All training-data exports preserve the source records as JSON objects and remain valid JSONL. A correction never overwrites its source row.

## Architecture

```text
JSONL source (read-only)
  → DatasetStore byte-offset index
  → FastAPI sample/navigation API
  → React one-card review loop
  → ReviewStore SQLite decisions + drafts + event log
  → atomic JSONL / JSON exports
```

The frontend renders ChatML, ShareGPT, prompt/completion, Alpaca, and generic records. Chat roles, arbitrary message content, tool/function-call fields, record metadata, and raw JSON remain inspectable.

## Tests

The isolated backend test suite uses temporary datasets, databases, and export directories:

```bash
uv run pytest subprojects/dataset_reviewer/tests -q
```
