"""Validated, process-local proxy preferences shared by Gradio and Blender."""

from dataclasses import dataclass, field
import ipaddress
import threading
from urllib.parse import quote, urlsplit

try:
    from .env_config import get_env
except ImportError:
    from env_config import get_env


PROXY_CATEGORIES = {
    "llm": "LLM (Reasoning & Vision)",
    "image_generation": "Image Generation",
    "asset_retrieval": "3D Model Retrieval",
    "model_generation": "3D Model Generation",
    "animation": "Animation & Rigging",
    "environment_assets": "HDRI & Textures",
}
PROXY_PROTOCOLS = ("HTTP", "HTTPS", "SOCKS5")


@dataclass(frozen=True)
class ProxySettings:
    host: str = "127.0.0.1"
    protocol: str = "HTTP"
    port: int = 7897
    username: str = field(default="", repr=False)
    password: str = field(default="", repr=False)
    categories: frozenset = field(default_factory=frozenset)

    def __post_init__(self):
        host = str(self.host or "").strip()
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        if not host or any(c.isspace() or c in "/\\@?#" for c in host):
            raise ValueError("Server Address must be a hostname or IP address without a protocol or path.")
        if ":" in host:
            try:
                ipaddress.IPv6Address(host)
            except ValueError:
                raise ValueError("Server Address must not include a protocol or port.") from None
        protocol = str(self.protocol).upper()
        if protocol not in PROXY_PROTOCOLS:
            raise ValueError("Choose HTTP, HTTPS, or SOCKS5 as the Proxy Protocol.")
        try:
            port = int(self.port)
            if isinstance(self.port, bool) or float(self.port) != port or not 1 <= port <= 65535:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Proxy Port must be a whole number between 1 and 65535.") from None
        categories = frozenset(self.categories or ())
        if categories - PROXY_CATEGORIES.keys():
            raise ValueError("Unknown proxy service category.")
        username, password = self.username or "", self.password or ""
        if password and not username:
            raise ValueError("Enter a Username when using a proxy Password.")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "port", port)
        object.__setattr__(self, "username", username)
        object.__setattr__(self, "password", password)
        object.__setattr__(self, "categories", categories)

    @property
    def url(self):
        scheme = "socks5h" if self.protocol == "SOCKS5" else self.protocol.lower()
        auth = ""
        if self.username:
            auth = f"{quote(self.username, safe='')}:{quote(self.password, safe='')}@"
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{scheme}://{auth}{host}:{self.port}"

    def to_dict(self):
        """For the local Blender bridge only; never write this to project files."""
        return dict(host=self.host, protocol=self.protocol, port=self.port,
                    username=self.username, password=self.password,
                    categories=sorted(self.categories))

    def summary(self):
        if not self.categories:
            return "Proxy: Off — all services connect directly."
        labels = ", ".join(label for key, label in PROXY_CATEGORIES.items() if key in self.categories)
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"Proxy: {self.protocol} · {host}:{self.port} · {labels}"


_lock = threading.Lock()
_settings = None


def get_proxy_settings():
    global _settings
    with _lock:
        if _settings is None:
            _settings = ProxySettings(
                host=get_env("STORYBLENDER_PROXY_HOST", "127.0.0.1"),
                protocol=get_env("STORYBLENDER_PROXY_PROTOCOL", "HTTP"),
                port=get_env("STORYBLENDER_PROXY_PORT", "7897"),
                username=get_env("STORYBLENDER_PROXY_USERNAME", ""),
                password=get_env("STORYBLENDER_PROXY_PASSWORD", ""),
                categories=[s.strip() for s in get_env("STORYBLENDER_PROXY_CATEGORIES", "").split(",") if s.strip()],
            )
        return _settings


def configure_proxy(settings):
    global _settings
    validated = settings if isinstance(settings, ProxySettings) else ProxySettings(**settings)
    with _lock:
        _settings = validated
    return validated


def is_loopback_url(url):
    host = (urlsplit(str(url)).hostname or "").lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def get_proxy_url(category, url=None):
    settings = get_proxy_settings()
    if category not in settings.categories or (url and is_loopback_url(url)):
        return None
    return settings.url
