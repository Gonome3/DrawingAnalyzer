"""
PDF Reader - Reads a PDF file using PyMuPDF and prints its contents to the console.

Usage:
    python pdf_reader.py <filepath>
"""

import os
import sys
import json
import math
import base64
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Optional

try:
    import pymupdf  # PyMuPDF
except ImportError:
    try:
        import fitz as pymupdf  # Older import name for PyMuPDF
    except ImportError:
        print("Error: PyMuPDF is not installed. Install it with: pip install pymupdf")
        sys.exit(1)


def read_pdf(filepath: str) -> None:
    """Open a PDF file and print all its text contents to the console."""
    pdf_path = Path(filepath)

    if not pdf_path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    if not pdf_path.is_file():
        print(f"Error: Not a file: {filepath}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"Warning: File does not have a .pdf extension: {filepath}")

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    print(f"=== PDF: {pdf_path.name} ===")
    print(f"Pages: {doc.page_count}")
    print("=" * 60)

    for page_num, page in enumerate(doc, start=1):
        print(f"\n--- Page {page_num} ---")
        text = page.get_text()
        print(text if text.strip() else "[No extractable text on this page]")

    doc.close()
    print("\n" + "=" * 60)
    print("End of document")


def _merge_colinear_spans(spans: list) -> list:
    """Merge horizontally-adjacent spans that share the same baseline,
    rotation, and font size. PyMuPDF often splits a single visual phrase
    into multiple spans (e.g. "Ø10.00" arrives as "Ø" + "10.00"); this
    collapses them so downstream consumers see one annotation per entry.
    """
    if not spans:
        return []

    # Merging only makes sense within a single rotation group.
    by_rotation: dict = {}
    for s in spans:
        by_rotation.setdefault(s["rotation"], []).append(s)

    merged: list = []
    for rotation, group in by_rotation.items():
        # Sort in reading order along the rotation's primary axis.
        if rotation in (0, 180):
            group.sort(key=lambda s: (round(s["bbox"][1]), s["bbox"][0]))
        else:  # 90 / 270 -- text runs along the y-axis
            group.sort(key=lambda s: (round(s["bbox"][0]), s["bbox"][1]))

        current = None
        for s in group:
            if current is None:
                current = dict(s)
                continue

            same_size = abs(current["size"] - s["size"]) < 0.5
            font_h = max(current["size"], s["size"])

            if rotation in (0, 180):
                same_line = abs(current["bbox"][1] - s["bbox"][1]) < 1.0
                gap = s["bbox"][0] - current["bbox"][2]
            else:
                same_line = abs(current["bbox"][0] - s["bbox"][0]) < 1.0
                gap = s["bbox"][1] - current["bbox"][3]

            # Allow small overlaps and gaps up to ~one space-character wide.
            adjacent = -1.0 <= gap <= font_h * 0.4

            if same_size and same_line and adjacent:
                sep = " " if gap > font_h * 0.15 else ""
                current["text"] = current["text"] + sep + s["text"]
                current["bbox"] = [
                    min(current["bbox"][0], s["bbox"][0]),
                    min(current["bbox"][1], s["bbox"][1]),
                    max(current["bbox"][2], s["bbox"][2]),
                    max(current["bbox"][3], s["bbox"][3]),
                ]
            else:
                merged.append(current)
                current = dict(s)
        if current is not None:
            merged.append(current)

    # Final ordering: top-to-bottom, then left-to-right (typical reading order).
    merged.sort(key=lambda s: (round(s["bbox"][1]), s["bbox"][0]))
    return merged


def _angle_from_dir(direction: Any) -> int:
    """Convert PyMuPDF's line direction tuple (cos θ, sin θ) into degrees,
    snapped to the nearest 90° (0 / 90 / 180 / 270). Engineering drawings
    almost exclusively use orthogonal text rotation, so snapping keeps
    downstream reasoning simple."""
    if not direction or len(direction) < 2:
        return 0
    cos_t, sin_t = direction[0], direction[1]
    angle_deg = math.degrees(math.atan2(sin_t, cos_t))
    snapped = int(round(angle_deg / 90.0)) * 90
    return snapped % 360


def extract_text_with_coordinates(filepath: str, compact: bool = False) -> dict:
    """
    Extract every text span from a PDF along with its position, font, size,
    rotation, and color. Returns a JSON-serializable dict that can be passed
    to a multimodal LLM (alongside a rendered page image) so it can reason
    about spatial relationships -- e.g. linking a tolerance value to the
    nominal dimension it modifies, or grouping a feature control frame.

    PDF coordinate notes:
      * Coordinates are in PDF points (1/72 inch).
      * The origin (0, 0) is the TOP-LEFT of the page in PyMuPDF's `Rect`
        convention used here. Y increases downward.
      * `bbox` is [x0, y0, x1, y1] -- top-left and bottom-right corners.

    Output schema:
    {
      "document": "<filename>",
      "page_count": N,
      "pages": [
        {
          "page_number": 1,
          "width":  float,    # page width  in points
          "height": float,    # page height in points
          "spans": [
            {
              "text":     str,
              "bbox":     [x0, y0, x1, y1],
              "font":     str,
              "size":     float,    # font size in points
              "rotation": int,      # 0 / 90 / 180 / 270
              "color":    int,      # packed sRGB integer
              "flags":    int,      # PyMuPDF font flags (bold/italic/etc.)
              "origin":   [x, y]    # baseline-left anchor of the span
            },
            ...
          ]
        },
        ...
      ]
    }
    """
    pdf_path = Path(filepath)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {filepath}")

    doc = pymupdf.open(str(pdf_path))
    result: dict = {
        "document": pdf_path.name,
        "page_count": doc.page_count,
        "pages": [],
    }

    for page_num, page in enumerate(doc, start=1):
        page_data: dict = {
            "page_number": page_num,
            "width": round(page.rect.width, 3),
            "height": round(page.rect.height, 3),
            "spans": [],
        }

        text_dict = page.get_text("dict")
        for block in text_dict.get("blocks", []):
            # block type: 0 = text, 1 = image. We only want text here.
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                rotation = _angle_from_dir(line.get("dir"))
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    bbox = span.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    origin = span.get("origin", [bbox[0], bbox[3]])
                    page_data["spans"].append({
                        "text": text,
                        "bbox": [round(float(c), 3) for c in bbox],
                        "font": span.get("font", ""),
                        "size": round(float(span.get("size", 0.0)), 2),
                        "rotation": rotation,
                        "color": int(span.get("color", 0)),
                        "flags": int(span.get("flags", 0)),
                        "origin": [round(float(c), 3) for c in origin],
                    })

        result["pages"].append(page_data)

    doc.close()

    if compact:
        # Trim every page: merge co-linear spans, drop fields the LLM
        # almost never needs (font name, color, font flags, baseline origin),
        # and round coordinates to integers. Typical reduction: 3-5x fewer
        # tokens vs the verbose form, with no loss of spatial information.
        for page in result["pages"]:
            page["width"] = int(round(page["width"]))
            page["height"] = int(round(page["height"]))
            merged = _merge_colinear_spans(page["spans"])
            slim: list = []
            for s in merged:
                entry: dict = {
                    "text": s["text"],
                    "bbox": [int(round(c)) for c in s["bbox"]],
                    "size": round(s["size"], 1),
                }
                if s["rotation"]:
                    # Only emit rotation when it's non-zero (the common case
                    # is horizontal text -- saves a field on most spans).
                    entry["rotation"] = s["rotation"]
                slim.append(entry)
            page["spans"] = slim

    return result


def format_as_compact_text(data: dict) -> str:
    """Render the (compact) extraction dict as one span per line. This is
    the most token-efficient representation for an LLM: roughly a quarter
    the tokens of equivalent JSON.

    Per-span format:
        [x0,y0,x1,y1] s<size> [r<rot>] "text"

    Rotation is omitted when 0. Quotes inside the text are escaped.
    The header lines give the document name and per-page dimensions so
    the LLM knows the coordinate range it's working in.
    """
    lines: list = []
    lines.append(f'# document: {data.get("document", "")}  pages: {data.get("page_count", 0)}')
    for page in data.get("pages", []):
        n = len(page.get("spans", []))
        lines.append(
            f'## PAGE {page["page_number"]}  '
            f'size={page.get("width")}x{page.get("height")}  spans={n}'
        )
        for s in page.get("spans", []):
            bbox = s["bbox"]
            text = s["text"].replace("\\", "\\\\").replace('"', '\\"')
            rot = s.get("rotation", 0)
            rot_str = f" r{rot}" if rot else ""
            lines.append(
                f'[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}] '
                f's{s["size"]}{rot_str} "{text}"'
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM extraction pipeline
# ---------------------------------------------------------------------------

DRAWING_SCHEMA_DESCRIPTION = """\
You are extracting structured data from an engineering drawing. Return a SINGLE
JSON object matching the schema below. Use both the rendered page image(s) and
the extracted text spans (each span gives the literal text and its bounding box
in PDF points -- top-left origin, Y increases downward).

SCHEMA (all fields nullable if absent in the drawing; arrays may be empty):
{
  "drawing": {
    "drawing_number": str, "revision": str, "title": str, "size": str,
    "scale": str, "units": "in"|"mm", "weight": {"value": float, "unit": str}|null,
    "material": str, "finish": str, "projection": "third_angle"|"first_angle"|null,
    "sheet": {"current": int, "total": int}, "company": str
  },
  "default_tolerances": {
    "linear_2_place": {"plus": float, "minus": float, "unit": "in"|"mm"},
    "linear_3_place": {"plus": float, "minus": float, "unit": "in"|"mm"},
    "angular":        {"plus": float, "minus": float, "unit": "deg"}
  },
  "approvals":  [{"role": "drawn"|"reviewed"|"approved", "name": str, "date": "YYYY-MM-DD"}],
  "revisions":  [{"rev": str, "description": str, "date": "YYYY-MM-DD", "approved_by": str}],
  "notes":      [{"number": int, "text": str}],
  "dimensions": [
    {
      "id": "d1", "kind": "linear"|"radius"|"diameter"|"angle",
      "nominal": float, "unit": "in"|"mm"|"deg",
      "tolerance": {
        "source": "explicit"|"default_2_place"|"default_3_place"|"default_angular",
        "plus": float, "minus": float
      },
      "semantic_role": "overall_length"|"overall_width"|"overall_height"|
                       "edge_to_feature_distance"|"hole_diameter"|"hole_depth"|
                       "fillet_radius"|"thread_depth"|"other",
      "view": "front"|"top"|"right"|"left"|"isometric"|"section"|"detail"|null,
      "axis": "x"|"y"|"z"|null,
      "feature_ref": "f1"|null,
      "modifier": "TYP"|"REF"|"BASIC"|null,
      "source_text": str
    }
  ],
  "features": [
    {
      "id": "f1",
      "kind": "through_hole"|"blind_hole"|"threaded_hole"|"counterbore"|
              "countersink"|"fillet"|"chamfer"|"slot"|"pocket"|"other",
      "diameter":    {"value": float, "unit": "in"|"mm"}|null,
      "depth":       {"value": float, "unit": "in"|"mm"}|null,
      "radius":      {"value": float, "unit": "in"|"mm"}|null,
      "thread_spec": str|null,
      "modifier":    "THRU"|null,
      "applies_to":  "all_corners"|"all_edges"|null,
      "view":        "front"|"top"|"right"|"left"|"isometric"|"section"|"detail"|null,
      "source_text": str
    }
  ],
  "surface_finish": {"minimum_ra": int, "applies_unless_otherwise_specified": bool}|null
}

RULES:
1. For every dimension WITHOUT an explicit ± tolerance on the drawing, fill in
   the inherited default from `default_tolerances` and set `tolerance.source`
   accordingly (e.g. "4.00" with no ± uses default_2_place).
2. ALWAYS include `source_text` -- the exact string as it appears on the drawing.
3. Use ids "d1", "d2", ... for dimensions and "f1", "f2", ... for features.
4. When a dimension describes a specific feature (e.g. a hole diameter, a
   fillet radius, an edge-to-hole distance), set `feature_ref` to that feature's id.
5. Determine `view` from which view the annotation sits inside.
6. Output ONLY the JSON object -- no commentary, no markdown fences."""


def load_config(path: Optional[str] = None) -> dict:
    """Load the LLM endpoint config from a JSON file. Search order:
       1. The path passed in on the command line (--config)
       2. The PDF_EXTRACTOR_CONFIG environment variable
       3. config.json next to this script

    Expected file shape:
       {
         "api_key":     "...",
         "endpoint":    "https://your-ollama-host.example.com/api/chat",
         "model":       "qwen3-vl",
         "num_ctx":     65536,    # context window (override Ollama's 2048 default)
         "num_predict": 8192      # max output tokens (override Ollama's 128 default)
       }
    """
    candidates: list = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("PDF_EXTRACTOR_CONFIG")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).parent / "config.json")

    for p in candidates:
        if p.exists() and p.is_file():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            if not cfg.get("api_key"):
                raise ValueError(f"Config file {p} is missing required field 'api_key'")
            return {
                "api_key": cfg["api_key"],
                # Placeholder endpoint -- replace in your config.json
                "endpoint": cfg.get("endpoint", "https://ollama.example.com/api/chat"),
                "model": cfg.get("model", "llava"),
                # Ollama's defaults (num_ctx=2048, num_predict=128) are far too
                # small for this pipeline -- our prompt + image alone exceeds 2K
                # tokens. 64K context / 8K output is a safe starting point.
                "num_ctx": int(cfg.get("num_ctx", 65536)),
                "num_predict": int(cfg.get("num_predict", 8192)),
                "_source": str(p),
            }

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "No config file found. Create config.json (see config.example.json) with at "
        "minimum an 'api_key' field, or set the PDF_EXTRACTOR_CONFIG env var.\n"
        f"Searched:\n  {tried}"
    )


def render_page_to_base64_png(page: Any, dpi: int = 200) -> str:
    """Rasterize a PDF page to a base64-encoded PNG string. 200 DPI is a
    good balance for vision models -- detailed enough to read small dimension
    text, not so large that we balloon the request payload."""
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("ascii")


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


def extract_drawing(filepath: str, config: dict, dpi: int = 200) -> dict:
    """End-to-end: read the PDF, render every page to an image, extract a
    compact text-with-coordinates payload, and ask the configured LLM to
    populate the drawing schema. Returns the structured dict."""
    pdf_path = Path(filepath)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {filepath}")

    # Compact text payload (already token-efficient).
    text_data = extract_text_with_coordinates(str(pdf_path), compact=True)
    text_payload = format_as_compact_text(text_data)

    # Rasterize every page for the multimodal LLM.
    doc = pymupdf.open(str(pdf_path))
    images_b64 = [render_page_to_base64_png(page, dpi=dpi) for page in doc]
    doc.close()

    prompt = (
        f"{DRAWING_SCHEMA_DESCRIPTION}\n\n"
        f"=== EXTRACTED TEXT SPANS (compact) ===\n"
        f"{text_payload}\n\n"
        f"Now produce the JSON object."
    )

    return call_ollama(config, prompt, images_b64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a PDF file and print its contents to the console."
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default=None,
        help="Path to the PDF file to read (optional; will prompt if omitted)",
    )
    parser.add_argument(
        "--coords",
        action="store_true",
        help="Extract text with coordinates as structured JSON (instead of plain text dump)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="With --coords, merge co-linear spans and emit a token-efficient text format",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run the full extraction pipeline: render pages + send to LLM, output structured JSON",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the JSON config file containing api_key/endpoint/model (defaults: $PDF_EXTRACTOR_CONFIG, then config.json beside this script)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI for page rasterization sent to the LLM (default: 200)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="With --coords or --extract, write the output to this file instead of stdout",
    )
    args = parser.parse_args()

    filepath = args.filepath
    if not filepath:
        try:
            filepath = input("Enter the path to the PDF file: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo input provided. Exiting.")
            sys.exit(1)

        # Strip surrounding quotes that often come from drag-and-drop paths
        if len(filepath) >= 2 and filepath[0] == filepath[-1] and filepath[0] in ('"', "'"):
            filepath = filepath[1:-1]

        if not filepath:
            print("Error: No filepath provided.")
            sys.exit(1)

    if args.extract:
        try:
            config = load_config(args.config)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Config error: {e}")
            sys.exit(1)

        print(f"Using config from: {config.get('_source')}")
        print(f"Endpoint: {config['endpoint']}  |  Model: {config['model']}")
        print(f"num_ctx: {config['num_ctx']}  |  num_predict: {config['num_predict']}")
        print(f"Extracting drawing from {filepath} ...")

        try:
            structured = extract_drawing(filepath, config, dpi=args.dpi)
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Extraction error: {e}")
            sys.exit(1)

        output_text = json.dumps(structured, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            print(f"Wrote structured drawing JSON to {args.output}")
        else:
            print(output_text)

    elif args.coords:
        try:
            data = extract_text_with_coordinates(filepath, compact=args.compact)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

        if args.compact:
            output_text = format_as_compact_text(data)
        else:
            output_text = json.dumps(data, indent=2, ensure_ascii=False)

        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            total_spans = sum(len(p["spans"]) for p in data["pages"])
            fmt = "compact text" if args.compact else "JSON"
            print(
                f"Wrote {total_spans} text spans across {data['page_count']} page(s) "
                f"to {args.output} ({fmt})"
            )
        else:
            print(output_text)
    else:
        read_pdf(filepath)


if __name__ == "__main__":
    main()
