"""PDF text extraction utilities.

Handles three responsibilities:
  * `read_pdf`                       -- simple text dump for debugging / quick look
  * `extract_text_with_coordinates`  -- structured per-span extraction with bbox,
                                        font, size, and rotation; supports a
                                        compact mode that merges co-linear spans
                                        and drops fields the LLM doesn't need
  * `format_as_compact_text`         -- token-efficient one-span-per-line text
                                        representation of the compact dict
"""

import sys
import math
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF



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
