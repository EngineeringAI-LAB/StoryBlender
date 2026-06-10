"""Drop-in replacement for any_llm.completion that properly closes the underlying client session.

The any_llm.api.completion() function creates a provider (e.g. GeminiProvider with a genai.Client)
internally but never closes it, leaving aiohttp sessions unclosed. This wrapper creates the provider
explicitly, calls completion, and closes the client in a finally block.

Usage: replace `from any_llm import completion` with
       `from operators.llm_completion import completion`
"""

import asyncio
import logging

from any_llm import AnyLLM
from any_llm.constants import LLMProvider


class _ThinkingPartFilter(logging.Filter):
    """Suppress the 'non-text parts in the response' warning emitted by
    google.genai.types when Gemini's thinking mode produces thought_signature
    parts alongside regular text.  The text content is still returned correctly."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "non-text parts in the response" not in record.getMessage()


class _AFCFilter(logging.Filter):
    """Suppress the 'AFC is enabled with max remote calls' INFO message emitted by
    google_genai.models when Automatic Function Calling is enabled by default."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "AFC is enabled" not in record.getMessage()


logging.getLogger("google_genai.types").addFilter(_ThinkingPartFilter())
logging.getLogger("google_genai.models").addFilter(_AFCFilter())


def _close_client(llm):
    """Close both sync and async sessions on the provider's client."""
    client = getattr(llm, 'client', None)
    if client is None:
        return

    # Close sync httpx session (may actually be an async coroutine)
    if hasattr(client, 'close'):
        try:
            result = client.close()
            if asyncio.iscoroutine(result):
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(result)
                finally:
                    loop.close()
        except Exception:
            pass

    # Close async aiohttp session (the actual source of the warning)
    if hasattr(client, 'aio') and hasattr(client.aio, 'aclose'):
        _aclose = client.aio.aclose
    elif hasattr(client, '_api_client') and hasattr(client._api_client, 'aclose'):
        _aclose = client._api_client.aclose
    else:
        return

    # THREAD-SAFETY: Never use create_task() on a potentially running loop
    # (e.g. Gradio's asyncio loop inside Blender).  That would schedule the
    # close on another thread, leaving dangling references that cause GC
    # crashes during Blender's OpenGL operations.  Always run the close
    # synchronously in a fresh, short-lived event loop.
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_aclose())
        finally:
            loop.close()
    except Exception:
        pass


def _normalize_api_base(api_base, provider_key):
    """Ensure api_base ends with /v1 for OpenAI-compatible providers.

    The any_llm SDK passes api_base directly to AsyncOpenAI(base_url=...),
    which expects the version path (e.g. https://host/v1).  Many proxy
    services only store the bare domain in config.
    """
    if api_base is None:
        return None
    try:
        from any_llm.providers.openai.base import BaseOpenAIProvider
        provider_cls = AnyLLM.get_provider_class(provider_key)
        if issubclass(provider_cls, BaseOpenAIProvider):
            api_base = api_base.rstrip('/')
            if not api_base.endswith('/v1'):
                api_base += '/v1'
    except Exception:
        pass
    return api_base


def completion(model, messages, *, provider=None, api_key=None, api_base=None, client_args=None, **kwargs):
    """Drop-in replacement for any_llm.completion with proper session cleanup."""
    if provider is None:
        provider_key, model_id = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_id = model

    api_base = _normalize_api_base(api_base, provider_key)
    llm = AnyLLM.create(provider_key, api_key=api_key, api_base=api_base, **(client_args or {}))
    try:
        return llm.completion(model=model_id, messages=messages, **kwargs)
    finally:
        _close_client(llm)


if __name__ == "__main__":
    from operators.api_keys import anyllm_api_key, anyllm_api_base

    response = completion(
        model="gpt-5.4-high",
        provider="openai",
        messages=[{"role": "user", "content": "Who are you?"}],
        api_key=anyllm_api_key,
        api_base=anyllm_api_base,
    )
    print("Response:", response)
    print("Response type:", type(response))
    print("Response content:", response.choices[0].message.content)
