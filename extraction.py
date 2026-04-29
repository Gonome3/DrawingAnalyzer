"""End-to-end orchestration: PDF -> structured drawing JSON via LLM.

This is the layer that wires text extraction (`pdf_text`) and the LLM
client (`llm_client`) together with the drawing schema. It's also where
future preprocessing layers (region detection, annotation clustering,
post-extraction validation) will plug in.
"""

from pathlib import Path

from pdf_text import extract_text_with_coordinates, format_as_compact_text
from llm_client import render_pdf_pages_to_base64, call_ollama


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


def extract_drawing(
    filepath: str,
    config: dict,
    dpi: int = 200,
    verbose: bool = False,
) -> dict:
    """End-to-end: read the PDF, render every page to an image, extract a
    compact text-with-coordinates payload, and ask the configured LLM to
    populate the drawing schema. Returns the structured dict.

    When `verbose=True`, prints the assembled LLM prompt and a small stats
    footer (character count, rough text-token estimate, image count, and
    configured num_ctx) so you can gauge how much of the context window
    is being consumed.
    """
    pdf_path = Path(filepath)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {filepath}")

    # Compact text payload (already token-efficient).
    text_data = extract_text_with_coordinates(str(pdf_path), compact=True)
    text_payload = format_as_compact_text(text_data)

    # Rasterize every page for the multimodal LLM.
    images_b64 = render_pdf_pages_to_base64(str(pdf_path), dpi=dpi)

    prompt = (
        f"{DRAWING_SCHEMA_DESCRIPTION}\n\n"
        f"=== EXTRACTED TEXT SPANS (compact) ===\n"
        f"{text_payload}\n\n"
        f"Now produce the JSON object."
    )

    if verbose:
        bar = "=" * 70
        print(bar)
        print("LLM PROMPT (text portion):")
        print(bar)
        print(prompt)
        print(bar)
        # Rough tokenization heuristic: ~4 chars/token for English-like text.
        # Real model tokenizers vary, but this gives a useful order-of-magnitude.
        approx_tokens = len(prompt) // 4
        print(
            f"Prompt size: {len(prompt):,} chars  "
            f"(~{approx_tokens:,} text tokens) + {len(images_b64)} image(s)"
        )
        print(
            f"Configured num_ctx: {config['num_ctx']:,}  |  "
            f"num_predict: {config['num_predict']:,}"
        )
        print(
            "Note: image token cost is model-specific and not included in the "
            "char/token count above."
        )
        print(bar)

    return call_ollama(config, prompt, images_b64)
