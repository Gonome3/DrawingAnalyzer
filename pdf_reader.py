"""
PDF Reader - Reads a PDF file using PyMuPDF and prints its contents to the console.

Usage:
    python pdf_reader.py <filepath>
"""

import sys
import json
import math
import argparse
from pathlib import Path
from typing import Any

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


def extract_text_with_coordinates(filepath: str) -> dict:
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
    return result


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
        "-o", "--output",
        default=None,
        help="When used with --coords, write JSON to this file instead of stdout",
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

    if args.coords:
        try:
            data = extract_text_with_coordinates(filepath)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

        json_text = json.dumps(data, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(json_text, encoding="utf-8")
            total_spans = sum(len(p["spans"]) for p in data["pages"])
            print(f"Wrote {total_spans} text spans across {data['page_count']} page(s) to {args.output}")
        else:
            print(json_text)
    else:
        read_pdf(filepath)


if __name__ == "__main__":
    main()
