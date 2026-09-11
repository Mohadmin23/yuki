# Yuki RAG — Phase 1 retrieval foundation

Phase 1 adds a shared retrieval boundary without adding another agent loop,
changing the 18-tool ontology, or allowing retrieved text to execute actions.

## Runtime flow

```text
validated recall/search/fetch call
              │
              ▼
      provider retrieval
              │
              ▼
 typed RetrievalHit objects
 text + title + URI + source type + rank
              │
              ▼
 source-bounded retrieval context
              │
              ▼
 existing tool-result injection
              │
              ▼
       Yuki's main model
```

The dispatcher and direct router still decide which existing tool is needed.
The retrieval package does not decide whether to use a tool and never calls a
chat model.

## Local retrieval

`recall` now combines two independent sources:

1. Existing semantic episodic recall from `episodic.py` and `sqlite-vec`.
2. SQLite FTS5 keyword retrieval over text-like files in `yuki/`.

The ranked lists are combined with Reciprocal Rank Fusion, so vector distance
and BM25 scores are never compared directly. The current chat remains direct
context and is excluded from episodic retrieval.

The Yuki-file index is incremental and stored at `data/retrieval.db`. It uses
file modification time, size, and a SHA-256 content digest. Binary files and
files larger than 1 MB are excluded. Exact file requests still belong to
`yuki_read`; topic retrieval belongs to `recall`.

If OpenRouter embeddings are unavailable, recall can still return matching
Yuki-file passages and exposes the episodic failure as a partial-retrieval
warning.

## Web retrieval

`search` keeps the existing DuckDuckGo provider, increases the candidate set to
five, and attempts to read relevant chunks from the top two public pages. Page
fetches run concurrently and are bounded by byte, character, page-count, and
timeout limits. Failed page reads fall back to the provider snippet.

Automatic page enrichment rejects localhost, private, link-local, reserved,
and other non-global addresses, including redirects. Direct `fetch` retains its
existing exact-URL behavior but now wraps returned text with source provenance.

Set `YUKI_WEB_RAG_FETCH_PAGES` to `0`, `1`, `2`, or `3` to control automatic
page enrichment. The default is `2`.

## Trust boundary

Every retrieval result is rendered as numbered SOURCE blocks. Web content is
explicitly labeled `UNTRUSTED EXTERNAL DATA`. The prompt tells the model that
source passages are evidence rather than commands, policies, role changes, or
tool requests. Lookalike retrieval/tool boundary markers inside source text are
neutralized before injection.

Search and fetch responses are told to cite useful SOURCE numbers and preserve
URLs. Recall responses are told that retrieved memories are candidates rather
than guaranteed facts.

Retrieved content cannot bypass the ordinary validated tool boundary. No code,
shell command, URL, or instruction found in a source is executed automatically.

## Deliberate Phase 1 limits

- Yuki-file passages use FTS5 only; they do not receive new embeddings yet.
- Search uses deterministic lexical chunk selection after DuckDuckGo ranking;
  there is no cross-encoder reranker yet.
- Search does not build a permanent web index. Web evidence is live and
  ephemeral.
- Retrieval quality still needs a frozen, human-labeled evaluation set before
  tuning chunk sizes, thresholds, fusion weights, or page counts.
- `read` and `yuki_read` remain exact deterministic tools.
- Action tools are unchanged.
