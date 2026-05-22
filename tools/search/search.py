"""Web search via DuckDuckGo (ddgs/duckduckgo_search package)."""
from __future__ import annotations


def tool_search(query):
    """Search the web. Returns (summary, links)."""
    try:
        from ddgs import DDGS  # type: ignore
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore
        except ImportError:
            return "Web search unavailable. Install: pip install ddgs", ""
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No results found.", ""
        lines = []
        links = []
        for r in results:
            lines.append(f"- {r['title']}: {r['body']}")
            if r.get("href"):
                links.append(f"  🔗 {r['href']}")
        return "\n".join(lines), "\n".join(links)
    except Exception as e:
        return f"Search error: {e}", ""
