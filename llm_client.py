"""Client for an Ollama-compatible LLM endpoint.

Owns two responsibilities:
  * Page rendering -- turn a PDF (or a PyMuPDF page) into base64-encoded
    PNG strings ready to send as multimodal input.
  * The actual /api/generate POST -- with text + images, structured-JSON
    output, thinking mode disabled, and the runtime context overrides
    that Ollama's tiny defaults make mandatory.
"""

import sys
import json
import base64
import urllib.request
import urllib.error
from typing import Any

try:
    import pymupdf  # PyMuPDF
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        print("Error: PyMuPDF is not installed. Install it with: pip install pymupdf")
        sys.exit(1)


def render_page_to_base64_png(page: Any, dpi: int = 200) -> str:
    """Rasterize a single PDF page to a base64-encoded PNG string. 200 DPI is
    a good balance for vision models -- detailed enough to read small dimension
    text, not so large that we balloon the request payload."""
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("ascii")


def render_pdf_pages_to_base64(filepath: str, dpi: int = 200) -> list:
    """Open a PDF and rasterize every page to a base64-encoded PNG. Returns
    a list of strings, one per page, ready to drop into an LLM `images` field."""
    doc = pymupdf.open(str(filepath))
    try:
        return [render_page_to_base64_png(page, dpi=dpi) for page in doc]
    finally:
        doc.close()


def call_ollama(config: dict, prompt: str, images_b64: list, timeout: int = 300) -> dict:
    """POST to an Ollama-compatible /api/generate endpoint with text + images
    and return the parsed JSON response. Uses `format: json` so the model is
    constrained to valid JSON output.

    We use /api/generate (not /api/chat) because:
      * Single-shot extraction doesn't need multi-turn chat semantics.
      * The payload is simpler: a `prompt` string + `system` string + `images`.
      * The response field is just `response`, not nested under `message.content`.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    payload = {
        "model": config["model"],
        "system": (
            "You are an expert at parsing mechanical engineering drawings "
            "into structured JSON. You are precise, faithful to the drawing, "
            "and never invent values that are not visible."
        ),
        "prompt": prompt,
        "images": images_b64,
        "stream": False,
        "format": "json",
        # Disable thinking mode for thinking-capable models (Qwen3, etc.).
        # When thinking is on AND format=json is set, these models tend to
        # emit their JSON answer through the `thinking` channel instead of
        # `response`, leaving the response field empty. We don't need
        # exposed reasoning for structured extraction anyway.
        "think": False,
        # Ollama-specific runtime options. Without these the request is
        # silently truncated to the model's tiny default context (2048).
        "options": {
            "num_ctx": config["num_ctx"],
            "num_predict": config["num_predict"],
        },
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        config["endpoint"], data=body, headers=headers, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM endpoint returned HTTP {e.code}: {e.reason}\nBody: {err_body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach LLM endpoint {config['endpoint']}: {e.reason}") from e

    response = json.loads(raw)
    # /api/generate returns the model output in the top-level `response` field.
    # Fallback: some thinking-mode models route their JSON output through the
    # `thinking` field even when format=json is set. We try response first,
    # then thinking, so we're robust to either behavior.
    content = response.get("response") or response.get("thinking") or ""
    if not content:
        # Surface Ollama's done_reason (e.g. 'load', 'stop', 'length') -- it's
        # the most useful clue when something goes wrong upstream.
        done_reason = response.get("done_reason", "<unknown>")
        raise RuntimeError(
            f"Empty content in LLM response (done_reason={done_reason}). "
            f"Full response: {response}"
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned non-JSON content despite format=json. Error: {e}\n"
            f"First 500 chars of content:\n{content[:500]}"
        ) from e
