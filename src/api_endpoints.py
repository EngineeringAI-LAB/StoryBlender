"""Normalize OpenAI-compatible API roots without duplicating version paths."""

from urllib.parse import urlsplit


def is_openrouter(api_base):
    return urlsplit(api_base or "").hostname in {"openrouter.ai", "api.openrouter.ai"}


def normalize_openai_base(api_base):
    if not api_base:
        return None
    base = api_base.strip().rstrip("/")
    if is_openrouter(base):
        return "https://openrouter.ai/api/v1"
    return base if base.endswith("/v1") else base + "/v1"
