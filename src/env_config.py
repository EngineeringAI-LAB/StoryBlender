"""Environment-backed configuration for StoryBlender."""

from __future__ import annotations

import os
from pathlib import Path


_ENV_LOADED = False


def _default_env_path() -> Path:
    """Return the add-on/repo root .env path."""
    return Path(__file__).resolve().parent.parent / ".env"


def load_storyblender_env() -> None:
    """Load StoryBlender's local .env file without overriding real env vars."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_file = os.getenv("STORYBLENDER_ENV_FILE")
    env_path = Path(env_file).expanduser() if env_file else _default_env_path()

    try:
        from dotenv import load_dotenv
    except Exception:
        _ENV_LOADED = True
        return

    load_dotenv(env_path, override=False)
    _ENV_LOADED = True


def get_env(name: str, default: str = "") -> str:
    """Read an env var after loading StoryBlender's local .env."""
    load_storyblender_env()
    return os.getenv(name, default)


def get_int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back when it is missing or invalid."""
    value = get_env(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default
