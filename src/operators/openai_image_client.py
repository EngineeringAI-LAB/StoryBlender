"""OpenAI-compatible images and OpenRouter's dedicated Images API."""

import base64
from io import BytesIO
import mimetypes
import os
from pathlib import Path

from PIL import Image

try:
    from ..api_endpoints import is_openrouter, normalize_openai_base
    from ..proxy_transport import requests_for
except ImportError:
    from api_endpoints import is_openrouter, normalize_openai_base
    from proxy_transport import requests_for

requests = requests_for("image_generation")


def _image_response(response):
    try:
        data = response.json()
    except ValueError:
        response.raise_for_status()
        raise RuntimeError("Image service returned a non-JSON response.") from None
    error = data.get("error")
    if not response.ok or error:
        message = error.get("message", "Image request failed") if isinstance(error, dict) else str(error or response.reason)
        raise RuntimeError(f"Image service error (HTTP {response.status_code}): {message[:600]}")
    images = data.get("data") or []
    if not images:
        raise RuntimeError("Image service returned no images.")
    item = images[0]
    if item.get("b64_json"):
        contents = base64.b64decode(item["b64_json"], validate=True)
    elif item.get("url"):
        url = item["url"]
        if url.startswith("data:"):
            contents = base64.b64decode(url.split(",", 1)[1], validate=True)
        else:
            downloaded = requests.get(url, timeout=(30, 300))
            downloaded.raise_for_status()
            contents = downloaded.content
    else:
        raise RuntimeError("Image service returned neither image bytes nor a URL.")
    with Image.open(BytesIO(contents)) as image:
        image.load()
        return image.copy()


def generate_openai_image(api_key, api_base, model, prompt, image_path=None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    api_base = (api_base or os.getenv("OPENAI_API_BASE") or "").strip()
    if not api_key:
        raise RuntimeError("Missing OpenAI API key. Fill it in Configuration or OPENAI_API_KEY.")
    if not api_base or api_base.lower() == "none":
        raise RuntimeError("Missing OpenAI API base. Fill it in Configuration or OPENAI_API_BASE.")
    if not api_base.startswith(("http://", "https://")):
        raise RuntimeError("OpenAI API base must start with http:// or https://.")
    base = normalize_openai_base(api_base)
    headers = {"Authorization": f"Bearer {api_key}"}
    if is_openrouter(base):
        payload = {"model": model, "prompt": prompt, "n": 1, "aspect_ratio": "1:1"}
        if image_path:
            path = Path(image_path)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            payload["input_references"] = [{
                "type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }]
        response = requests.post(f"{base}/images", headers=headers, json=payload, timeout=(30, 300))
    elif image_path:
        path = Path(image_path)
        with path.open("rb") as image_file:
            response = requests.post(
                f"{base}/images/edits", headers=headers,
                files={"image": (path.name, image_file, mimetypes.guess_type(path.name)[0] or "image/png")},
                data={"model": model, "prompt": prompt, "n": "1"}, timeout=(30, 300),
            )
    else:
        response = requests.post(
            f"{base}/images/generations", headers=headers,
            json={"model": model, "prompt": prompt, "size": "1024x1024"}, timeout=(30, 300),
        )
    return _image_response(response)
