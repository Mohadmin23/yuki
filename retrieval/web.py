"""Live web retrieval with typed sources and bounded public-page enrichment."""

from __future__ import annotations

import html
import ipaddress
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .core import RetrievalHit, chunk_text, lexical_score

MAX_BYTES = 400_000
MAX_PAGE_CHARS = 80_000
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Apple Silicon) "
        "llama-voice-assist/1.0 (+https://github.com/)"
    ),
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
}


class _ReadableHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_parts: list[str] = []
        self._parts: list[str] = []
        self._ignored_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth += 1
        if lowered == "title" and not self._ignored_depth:
            self._in_title = True
        if lowered in {"p", "div", "article", "section", "br", "li", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"script", "style", "noscript", "svg", "template"}:
            self._ignored_depth = max(0, self._ignored_depth - 1)
        if lowered == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        self._parts.append(data)

    def finish(self) -> tuple[str, str]:
        title = re.sub(r"\s+", " ", " ".join(self._title_parts)).strip()
        text = re.sub(r"[ \t\f\v]+", " ", " ".join(self._parts))
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
        return title, html.unescape(text)


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only full http:// or https:// URLs are allowed")
    host = parsed.hostname.casefold().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("local addresses are not eligible for automatic web retrieval")
    try:
        default_port = 443 if parsed.scheme == "https" else 80
        addresses = {
            item[4][0] for item in socket.getaddrinfo(host, parsed.port or default_port)
        }
    except OSError as exc:
        raise ValueError(f"could not resolve host: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(
                "private or non-global addresses are not eligible for web retrieval"
            )


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        _validate_public_url(target)
        return super().redirect_request(req, fp, code, msg, headers, target)


def fetch_public_page(url: str, *, timeout: float = 8.0) -> tuple[str, str]:
    """Fetch one public text page, validating the initial URL and redirects."""
    _validate_public_url(url)
    request = Request(url, headers=HEADERS)
    opener = build_opener(_PublicRedirects())
    with opener.open(request, timeout=timeout) as response:
        content_type = (response.headers.get("Content-Type") or "").casefold()
        if not any(
            kind in content_type for kind in ("text/", "json", "markdown", "html")
        ):
            raise ValueError(f"unsupported content type: {content_type or 'unknown'}")
        encoding = response.headers.get_content_charset() or "utf-8"
        raw = response.read(MAX_BYTES)
    try:
        body = raw.decode(encoding, errors="replace")
    except LookupError:
        body = raw.decode("utf-8", errors="replace")

    if "html" not in content_type:
        return "", re.sub(r"\s+", " ", body).strip()[:MAX_PAGE_CHARS]
    parser = _ReadableHTML()
    parser.feed(body)
    title, text = parser.finish()
    return title, text[:MAX_PAGE_CHARS]


def search_web(query: str, *, max_results: int = 5, client=None) -> list[RetrievalHit]:
    """Search DuckDuckGo and normalize its results into retrieval hits."""
    if client is None:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        client = DDGS()
    rows = list(client.text(query, max_results=max_results))
    hits: list[RetrievalHit] = []
    for index, row in enumerate(rows, start=1):
        uri = str(row.get("href") or "").strip()
        title = str(row.get("title") or uri or f"Search result {index}").strip()
        passage = str(row.get("body") or "").strip()
        if not passage and not uri:
            continue
        hits.append(
            RetrievalHit(
                id=f"web-search://{index}:{uri}",
                source_type="web_search_result",
                title=title,
                text=passage,
                uri=uri,
                rank=index,
                metadata={"provider": "DuckDuckGo", "provider_rank": index},
                untrusted=True,
            )
        )
    return hits


def _page_hits(
    query: str,
    search_hit: RetrievalHit,
    page_loader,
    *,
    chunks_per_page: int = 2,
) -> list[RetrievalHit]:
    title, text = page_loader(search_hit.uri)
    if not text:
        return []
    scored = sorted(
        (
            (lexical_score(query, passage), index, passage)
            for index, passage in enumerate(chunk_text(text))
        ),
        key=lambda item: (-item[0], item[1]),
    )[:chunks_per_page]
    page_title = title or search_hit.title
    return [
        RetrievalHit(
            id=f"{search_hit.uri}#chunk-{chunk_index}",
            source_type="web_page",
            title=page_title,
            text=passage,
            uri=search_hit.uri,
            rank=search_hit.rank,
            score=score,
            metadata={
                "provider": "DuckDuckGo + page fetch",
                "provider_rank": search_hit.rank,
                "chunk_index": chunk_index,
            },
            untrusted=True,
        )
        for score, chunk_index, passage in scored
    ]


def retrieve_web(
    query: str,
    *,
    max_results: int = 5,
    fetch_pages: int = 2,
    search_client=None,
    page_loader=fetch_public_page,
) -> tuple[list[RetrievalHit], list[str]]:
    """Search the web and enrich top public results with relevant page passages."""
    search_hits = search_web(query, max_results=max_results, client=search_client)
    if not search_hits or fetch_pages <= 0:
        return search_hits, []

    candidates = [hit for hit in search_hits[:fetch_pages] if hit.uri]
    page_hits: dict[str, list[RetrievalHit]] = {}
    warnings: list[str] = []
    with ThreadPoolExecutor(max_workers=min(3, len(candidates) or 1)) as executor:
        futures = {
            executor.submit(_page_hits, query, hit, page_loader): hit
            for hit in candidates
        }
        for future in as_completed(futures):
            hit = futures[future]
            try:
                page_hits[hit.id] = future.result()
            except Exception as exc:  # noqa: BLE001 - snippets remain useful
                warnings.append(f"could not read {hit.uri}: {exc}")

    combined: list[RetrievalHit] = []
    for hit in search_hits:
        enriched = page_hits.get(hit.id) or []
        combined.extend(enriched if enriched else [hit])
    return combined, warnings
