"""Per-client routing; never change process-wide proxy environment variables."""

from contextvars import ContextVar
from functools import wraps
import requests as _requests

try:
    from .proxy_config import get_proxy_url
except ImportError:
    from proxy_config import get_proxy_url


_category = ContextVar("storyblender_proxy_category", default=None)


def proxy_category(category):
    """Scope nested downloads to their task. Decorate thread workers explicitly."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            selected = category(*args, **kwargs) if callable(category) else category
            token = _category.set(selected)
            try:
                return function(*args, **kwargs)
            finally:
                _category.reset(token)
        return wrapped
    return decorate


class RoutingSession(_requests.Session):
    def __init__(self, category):
        super().__init__()
        self.category = category
        self.trust_env = False

    def _proxies_for(self, url):
        proxy = get_proxy_url(_category.get() or self.category, url)
        return {"http": proxy, "https": proxy, "all": proxy}

    def request(self, method, url, **kwargs):
        # Explicit direct routing also overrides any ambient OS/shell proxy.
        kwargs["proxies"] = self._proxies_for(url)
        return super().request(method, url, **kwargs)

    def rebuild_proxies(self, prepared_request, proxies):
        # Re-evaluate redirects, especially redirects back to local services.
        return super().rebuild_proxies(prepared_request, self._proxies_for(prepared_request.url))


class _CategoryRequests:
    """Module-local requests interface, preserving exceptions/adapters/type hints."""
    def __init__(self, category):
        class Session(RoutingSession):
            def __init__(self):
                super().__init__(category)
        self.Session = Session

    def __getattr__(self, name):
        return getattr(_requests, name)

    def request(self, method, url, **kwargs):
        with self.Session() as session:
            return session.request(method, url, **kwargs)

    def get(self, url, **kwargs):
        kwargs.setdefault("allow_redirects", True)
        return self.request("GET", url, **kwargs)

    def head(self, url, **kwargs):
        kwargs.setdefault("allow_redirects", False)
        return self.request("HEAD", url, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        return self.request("POST", url, data=data, json=json, **kwargs)


def requests_for(category):
    return _CategoryRequests(category)


def genai_http_options(category, base_url=None, existing=None):
    """Use HTTPX for both Gemini clients, including SOCKS and explicit direct mode."""
    import httpx
    options = dict(existing or {})
    if base_url:
        options["base_url"] = base_url
    proxy = get_proxy_url(category, options.get("base_url") or "https://generativelanguage.googleapis.com")
    options["client_args"] = {**(options.get("client_args") or {}), "proxy": proxy, "trust_env": False}
    # A transport makes google-genai use HTTPX instead of its optional aiohttp path.
    options["async_client_args"] = {
        **(options.get("async_client_args") or {}),
        "transport": httpx.AsyncHTTPTransport(proxy=proxy, trust_env=False),
        "trust_env": False,
    }
    return options


def llm_client_args(provider_key, api_base, existing=None):
    """Return provider kwargs and any HTTP clients owned by the caller."""
    import httpx
    from any_llm import AnyLLM
    from any_llm.providers.openai.base import BaseOpenAIProvider

    args = dict(existing or {})
    name = str(getattr(provider_key, "value", provider_key)).lower()
    if name in {"gemini", "vertexai"}:
        args["http_options"] = genai_http_options("llm", api_base, args.get("http_options"))
        return args, [args["http_options"]["async_client_args"]["transport"]]
    proxy = get_proxy_url("llm", api_base)
    cls = AnyLLM.get_provider_class(provider_key)
    if issubclass(cls, BaseOpenAIProvider) or name in {
        "openai", "azure", "azureopenai", "anthropic", "azureanthropic", "vertexaianthropic"
    }:
        client = httpx.AsyncClient(proxy=proxy, trust_env=False)
        args["http_client"] = client
        return args, [client]
    if name == "ollama":
        args.update(proxy=proxy, trust_env=False)
    elif proxy:
        raise ValueError(f"Proxy routing is not supported for the {name} SDK. Use an OpenAI-compatible provider endpoint.")
    return args, []
