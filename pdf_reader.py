"""PDF Reader CLI -- entry point for engineering drawing extraction.

This file is the thin CLI shell. The real work lives in:
  * config.py       -- load_config()
  * pdf_text.py     -- text extraction with coordinates, compact formatting
  * llm_client.py   -- Ollama API client + page rendering to base64 PNG
  * extraction.py   -- end-to-end orchestration + drawing schema

Usage:
    python pdf_reader.py <filepath>             # plain text dump
    python pdf_reader.py <filepath> --compact   # compact text spans (what the LLM sees)
    python pdf_reader.py <filepath> --extract   # full LLM extraction pipeline
"""

import sys
import json
import argparse
from pathlib import Path

from config import load_config
from pdf_text import (
    read_pdf,
    extract_text_with_coordinates,
    format_as_compact_text,
)
from extraction import extract_drawing


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
        "--compact",
        action="store_true",
        help="Output the compact text-with-coordinates payload that the LLM pipeline consumes (one span per line, useful for debugging what the model sees)",
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
        "-v", "--verbose",
        action="store_true",
        help="With --extract, print the assembled LLM prompt and context-usage stats before the call (debugging aid for gauging context window usage)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="With --compact or --extract, write the output to this file instead of stdout",
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
            structured = extract_drawing(
                filepath, config, dpi=args.dpi, verbose=args.verbose
            )
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Extraction error: {e}")
            sys.exit(1)

        output_text = json.dumps(structured, indent=2, ensure_ascii=False)
        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            print(f"Wrote structured drawing JSON to {args.output}")
        else:
            print(output_text)

    elif args.compact:
        try:
            data = extract_text_with_coordinates(filepath, compact=True)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")
            sys.exit(1)

        output_text = format_as_compact_text(data)
        if args.output:
            Path(args.output).write_text(output_text, encoding="utf-8")
            total_spans = sum(len(p["spans"]) for p in data["pages"])
            print(
                f"Wrote {total_spans} text spans across {data['page_count']} page(s) "
                f"to {args.output}"
            )
        else:
            print(output_text)
    else:
        read_pdf(filepath)


if __name__ == "__main__":
    main()
