"""Source-aware live web retrieval via DuckDuckGo and selected public pages."""

from __future__ import annotations

import os


def tool_search(query):
    """Search and retrieve relevant web passages. Returns (context, links)."""
    query = (query or "").strip()
    if not query:
        return "Error: empty web search query", ""
    try:
        from retrieval import render_retrieval_context
        from retrieval.web import retrieve_web

        configured = os.environ.get("YUKI_WEB_RAG_FETCH_PAGES", "2").strip()
        fetch_pages = int(configured) if configured.isdigit() else 2
        hits, warnings = retrieve_web(
            query,
            max_results=5,
            fetch_pages=max(0, min(fetch_pages, 3)),
        )
    except ImportError:
        return "Web search unavailable. Run `llama setup` to install ddgs.", ""
    except Exception as exc:  # noqa: BLE001
        return f"Search error: {exc}", ""

    if not hits:
        return "No results found.", ""

    context = render_retrieval_context(
        query,
        hits,
        label="LIVE WEB",
        warnings=warnings,
    )
    # URLs are already carried inside every SOURCE block. Keep the historical
    # tuple shape for CLI/web compatibility without duplicating them.
    return context, ""
