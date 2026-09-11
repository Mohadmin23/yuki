"""OpenRouter endpoint discovery and explicit provider pinning."""

from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import Request, urlopen


def provider_request_body(extra: dict | None, provider: str | None) -> dict | None:
    body = dict(extra or {})
    if isinstance(provider, str) and provider:
        policy = dict(body.get("provider") or {})
        policy.update({"only": [provider], "order": [provider], "allow_fallbacks": False})
        body["provider"] = policy
    return body or None


def fetch_model_providers(model_id: str) -> list[dict]:
    model = model_id.removeprefix("openrouter/")
    if "/" not in model:
        raise ValueError("An OpenRouter model id must include its author.")
    url = f"https://openrouter.ai/api/v1/models/{quote(model, safe='/')}/endpoints"
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=15) as response:
        payload = json.load(response)
    rows = payload.get("data", {}).get("endpoints")
    if not isinstance(rows, list):
        raise TypeError("OpenRouter returned no endpoint list for this model.")
    providers = {}
    for row in rows:
        tag = row.get("tag")
        if not isinstance(tag, str) or not tag:
            continue
        providers[tag] = {
            "id": tag,
            "name": row.get("provider_name") or tag,
            "context_length": row.get("context_length"),
            "pricing": row.get("pricing") or {},
            "supported_parameters": row.get("supported_parameters") or [],
        }
    return list(providers.values())


def provider_label(provider: dict) -> str:
    def price(value):
        try:
            return f"${float(value) * 1_000_000:g}"
        except (TypeError, ValueError):
            return "unknown"
    pricing = provider["pricing"]
    tools = "tools" if "tools" in provider["supported_parameters"] else "tools unlisted"
    context = provider.get("context_length")
    return (
        f"{provider['name']} [{provider['id']}] · {tools}\n"
        f"{price(pricing.get('prompt'))} in / {price(pricing.get('completion'))} out per 1M tokens"
        + (f" · context {context:,}" if isinstance(context, int) else "")
    )
