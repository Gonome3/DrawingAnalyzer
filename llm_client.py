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


def render_page_to_base64_png(page: Any, dpi: int = 200, grayscale: bool = True) -> str:
    """Rasterize a single PDF page to a base64-encoded PNG string. 200 DPI is
    a good balance for vision models -- detailed enough to read small dimension
    text, not so large that we balloon the request payload.

    Defaults to grayscale rendering because engineering drawings are
    essentially black ink on white paper -- the color channels carry no
    information gain but roughly triple the encoded byte count. Combined
    with PNG's good compression on line-art content, grayscale typically
    cuts the encoded payload by 2-3x with zero quality loss for our use
    case. Pass `grayscale=False` if a drawing uses color meaningfully
    (rare for production engineering drawings).
    """
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    colorspace = pymupdf.csGRAY if grayscale else pymupdf.csRGB
    pix = page.get_pixmap(matrix=matrix, alpha=False, colorspace=colorspace)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("ascii")


def render_pdf_pages_to_base64(filepath: str, dpi: int = 200, grayscale: bool = True) -> list:
    """Open a PDF and rasterize every page to a base64-encoded PNG. Returns
    a list of strings, one per page, ready to drop into an LLM `images` field."""
    doc = pymupdf.open(str(filepath))
    try:
        return [render_page_to_base64_png(page, dpi=dpi, grayscale=grayscale) for page in doc]
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
        # Streaming is required when the endpoint sits behind a proxy with a
        # read timeout (very common for k8s ingresses and stock nginx). A
        # non-streamed call holds the connection idle for the entire duration
        # of inference, which can run 30-120 seconds for a vision model on a
        # busy drawing -- enough to trigger a 504 Gateway Timeout. With
        # streaming, Ollama emits NDJSON chunks as tokens are produced,
        # keeping data flowing through the proxy and preventing idle-timeout
        # disconnections.
        "stream": True,
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

    # Read the streamed NDJSON response line-by-line, accumulating the
    # `response` (and fallback `thinking`) text fields across all chunks
    # until we see `done: true`.
    content_parts: list = []
    thinking_parts: list = []
    last_chunk: dict = {}

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                last_chunk = chunk
                if chunk.get("response"):
                    content_parts.append(chunk["response"])
                if chunk.get("thinking"):
                    thinking_parts.append(chunk["thinking"])
                if chunk.get("done"):
                    break
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"LLM endpoint returned HTTP {e.code}: {e.reason}\nBody: {err_body[:500]}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach LLM endpoint {config['endpoint']}: {e.reason}") from e

    # Same response/thinking fallback logic as before, applied to the
    # accumulated streamed text.
    content = "".join(content_parts) or "".join(thinking_parts)
    if not content:
        # Surface Ollama's done_reason (e.g. 'load', 'stop', 'length') -- it's
        # the most useful clue when something goes wrong upstream.
        done_reason = last_chunk.get("done_reason", "<unknown>")
        raise RuntimeError(
            f"Empty content in LLM response (done_reason={done_reason}). "
            f"Last chunk: {last_chunk}"
        )

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"LLM returned non-JSON content despite format=json. Error: {e}\n"
            f"First 500 chars of content:\n{content[:500]}"
        ) from e
