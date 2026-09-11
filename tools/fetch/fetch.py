"""Download a webpage and return its readable text.

Designed for "summarize this URL" use cases — search returns DDG snippets,
which don't contain page contents. This tool downloads the actual HTML and
extracts a plain-text version the model can summarize from."""

from __future__ import annotations

import re
from urllib.request import Request, urlopen

MAX_BYTES = 400_000  # hard cap on bytes pulled from the wire
MAX_RETURN_CHARS = 8000  # what we hand back to the model
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Apple Silicon) "
        "llama-voice-assist/1.0 (+https://github.com/)"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}

# Strip <script> / <style> / <noscript> bodies whole (case-insensitive,
# dotall so newlines inside the body don't break the match).
_BLOCK_TAGS_RE = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _looks_like_url(s: str) -> bool:
    return s.lower().startswith(("http://", "https://"))


def tool_fetch(url):
    """Fetch URL, strip HTML, return readable plain text (max 8000 chars)."""
    url = (url or "").strip()
    if not url:
        return "Error: empty URL"
    if not _looks_like_url(url):
        return f"Error: not a URL (need http:// or https://): {url[:80]}"

    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=10) as resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            raw = resp.read(MAX_BYTES)
    except Exception as e:  # noqa: BLE001
        return f"Fetch error: {e}"

    # Decode. Use the charset hint if present, otherwise be forgiving.
    encoding = "utf-8"
    m = re.search(r"charset=([\w\-]+)", ctype)
    if m:
        encoding = m.group(1)
    try:
        body = raw.decode(encoding, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")

    # Already plain text? Just clean whitespace.
    if "text/plain" in ctype or "json" in ctype or "markdown" in ctype:
        cleaned = _WS_RE.sub(" ", body).strip()
        text = cleaned[:MAX_RETURN_CHARS] if cleaned else "(empty page)"
        return _render_page(url, text)

    # HTML path: strip script/style blocks, then tags, then collapse whitespace.
    cleaned = _BLOCK_TAGS_RE.sub(" ", body)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return _render_page(url, "(page had no readable text after stripping HTML)")
    return _render_page(url, cleaned[:MAX_RETURN_CHARS])


def _render_page(url: str, text: str) -> str:
    """Put exact-page retrieval behind the same provenance/safety boundary."""
    from retrieval import RetrievalHit, render_retrieval_context

    hit = RetrievalHit(
        id=url,
        source_type="web_page",
        title=url,
        text=text,
        uri=url,
        rank=1,
        untrusted=True,
    )
    return render_retrieval_context(url, [hit], label="EXACT WEB PAGE")
