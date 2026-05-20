"""End-to-end orchestration: PDF -> structured drawing JSON via LLM.

This is the layer that wires text extraction (`pdf_text`) and the LLM
client (`llm_client`) together with the drawing schema. It's also where
future preprocessing layers (region detection, annotation clustering,
post-extraction validation) will plug in.
"""

import re
import unicodedata
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
  "drawing_summary": {
    "description":      str,     // 1-3 sentence factual description of the part
    "part_type":        str|null, // short lowercase noun, e.g. "plate", "bracket"
    "primary_function": str|null  // only if explicitly indicated on the drawing
  }|null,
  "default_tolerances": {
    "linear_2_place": {"plus": float, "minus": float, "unit": "in"|"mm"},
    "linear_3_place": {"plus": float, "minus": float, "unit": "in"|"mm"},
    "angular":        {"plus": float, "minus": float, "unit": "deg"}
  },
  "approvals":  [{"role": "drawn"|"reviewed"|"approved", "name": str, "date": "YYYY-MM-DD"}],
  "revisions":  [{"rev": str, "description": str, "date": "YYYY-MM-DD", "approved_by": str}],
  "notes":      [{"number": int, "text": str}],
  "views": [
    {
      "kind":  "front"|"top"|"right"|"left"|"bottom"|"back"|
               "isometric"|"section"|"detail"|"auxiliary"|"other",
      "label": str|null,    // e.g. "SECTION A-A", "DETAIL B"
      "scale": str|null     // only if explicitly different from main scale
    }
  ],
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
                       "fillet_radius"|"thread_depth"|"other"|null,
      "view": "front"|"top"|"right"|"left"|"bottom"|"back"|"isometric"|"section"|"detail"|"auxiliary"|"other"|null,
      "view_label": str|null,    // copy of the parent view's label (e.g. "SECTION A-A") for disambiguation
      "axis": "x"|"y"|"z"|null,
      "modifier": "TYP"|"REF"|"BASIC"|null,
      "source_text": str
    }
  ],
  "surface_finish": {"minimum_ra": int, "applies_unless_otherwise_specified": bool}|null,
  "uncovered_annotations": [str]   // brief descriptions of visible annotations that don't fit other schema fields (GD&T frames, datum refs, component data tables, etc.)
}

RULES:
1. NEVER invent values. Every value you emit must be visible on the drawing
   or be an inherited default explicitly defined in `default_tolerances`. If
   a value is not present on the drawing, omit the field or use null. Do
   NOT guess plausible-looking numbers just because the schema has a slot
   for them.
2. For tolerances:
   a. Use `tolerance.source: "explicit"` ONLY when an actual ± value is
      visible next to the dimension on the drawing. The `source_text` for
      that dimension MUST contain the ± portion verbatim
      (e.g. "2.500±.002", not just "2.500").
   b. For dimensions WITHOUT an explicit ± tolerance, inherit from
      `default_tolerances` and set `tolerance.source` to "default_2_place",
      "default_3_place", or "default_angular" as appropriate
      (e.g. "4.00" with no ± uses default_2_place).
   c. Never invent a tolerance value just because the field exists. If you
      did not see an explicit ± on the drawing, the source MUST be one of
      the default_* values, not "explicit".
3. ALWAYS include `source_text` -- the EXACT string as it appears on the
   drawing. The verifier will check that this string is present in the
   extracted text spans; fabricated source_text will be flagged.
4. Use ids "d1", "d2", ... for dimensions.
5. For `semantic_role`, apply these decision rules in order:
   a. If the dimension spans the full outer extent of the part profile
      along an axis -> `overall_length`, `overall_width`, or
      `overall_height` based on which axis the dimension is on.
   b. If the dimension's leader line points clearly to a hole ->
      `hole_diameter` or `hole_depth`.
   c. If the dimension is a radius callout on a fillet ->
      `fillet_radius`.
   d. If the dimension's leader line clearly connects an edge to a
      specific feature on the part -> `edge_to_feature_distance`.
   e. Otherwise -> `null`. Do NOT default to "edge_to_feature_distance"
      or "other" when uncertain. A null semantic_role is more useful
      downstream than a wrong label.
6. For `drawing_summary`: provide a brief factual description of the
   part based on what is visible on the drawing (1-3 sentences).
   Describe shape, prominent features, and orientation, not purpose
   unless purpose is explicit. `part_type` should be a short lowercase
   noun (e.g. "plate", "bracket", "shaft", "housing"). Set
   `primary_function` only if explicitly indicated by the drawing or
   its title; otherwise null. If the part cannot be characterised
   meaningfully, set `drawing_summary` to null.
7. For views:
   a. In the top-level `views` array, enumerate every distinct view
      present on the drawing (e.g. front, isometric, section,
      detail). Section and detail views MUST include their label
      exactly as shown on the drawing (e.g. "SECTION A-A",
      "DETAIL B"). Only set `scale` when a view is explicitly
      labelled with a scale that differs from the main drawing scale.
   b. For each dimension's `view` field, identify which view from the
      `views` array the annotation sits inside. The per-annotation
      `view` value must be consistent with the enumerated views.
   c. If the parent view in the `views` array has a `label`
      (e.g. "SECTION A-A", "DETAIL B"), copy that label verbatim
      into the dimension's `view_label` field so that annotations
      belonging to different sections or details can be
      distinguished. For views without a label (front, top, iso,
      etc.), set `view_label` to null.
8. For `uncovered_annotations`: list any annotation visible on the
   drawing that does NOT fit into any other schema field. Examples
   of things that belong here:
     - Hole, thread, chamfer, fillet, and other feature callouts
       (the schema captures their dimensions but not the feature
       descriptor itself; record e.g. "M6 threaded hole callout"
       or "1x45° chamfer applied to all edges")
     - GD&T feature control frames (perpendicularity, parallelism,
       position tolerances, often with datum references like ⊥ 0.5 B)
     - Datum reference labels (the boxed A, B, C, D, E, F symbols)
     - Component-specific data tables (gear tooth specifications,
       thread tables, hole charts)
     - Conditional tolerance tables (e.g. "measurements before
       heat treatment")
     - Per-feature surface finish callouts (Ra symbols pointing to
       a specific surface rather than the whole part)
     - External document references (e.g. "see measuring
       instructions KM 0022")
     - Process specification notes (heat treatment, coating,
       surface treatment per a referenced specification)
   Each entry is a brief descriptive string explaining what the
   annotation is and what it relates to. Use an empty list if no
   such annotations exist. Do NOT use this field to hide ambiguity
   for annotations that DO fit other schema fields -- it is for
   genuinely unsupported annotation types only. Err on the side of
   including more rather than fewer: this list is used to measure
   schema coverage gaps, so missing entries are worse than extra ones.
9. Output ONLY the JSON object -- no commentary, no markdown fences."""


def _normalize_for_text_match(text: str) -> str:
    """Normalize text for source_text verification:
      * NFC unicode normalization (collapses different Ø encodings, etc.)
      * Lowercase (forgives capitalization differences)
      * Collapse whitespace
      * Map common ± substitutions ("+/-" and "+-" both become "±")
    Aggressive on purpose -- we're checking 'did the model invent this
    entirely?' not 'does it match exactly character-for-character'."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.replace("+/-", "±").replace("+-", "±")
    return text


def verify_extraction_against_text(structured: dict, text_data: dict) -> dict:
    """Walk every dimension in the structured output and confirm that the
    `source_text` field actually appears in the extracted PDF text spans.
    Catches hallucinations where the model invented values that don't
    exist on the drawing.

    Mutates the structured dict in two ways:
      * Adds a `_verification` summary at the top level (issue_count + list
        of issues).
      * Demotes `tolerance.source` from "explicit" to "unverified" for any
        dimension whose source_text is not found in the drawing's text spans.
        Downstream consumers can then choose to ignore unverified tolerances
        or treat them with reduced confidence.

    The verification report is also useful raw material for thesis evaluation
    -- it gives a per-extraction count of suspected hallucinations that can
    be aggregated across a dataset.

    Returns the same dict (mutated) for convenient chaining.
    """
    # Build a normalized 'haystack' from every text span on every page.
    haystack_parts: list = []
    for page in text_data.get("pages", []) or []:
        for span in page.get("spans", []) or []:
            haystack_parts.append(span.get("text", ""))
    haystack = _normalize_for_text_match(" ".join(haystack_parts))

    issues: list = []

    def _check(item: dict, item_type: str) -> bool:
        """Return True iff `source_text` is found in the haystack. Records
        an entry in `issues` and returns False otherwise."""
        source_text = item.get("source_text") or ""
        if not source_text.strip():
            issues.append({
                "kind": "missing_source_text",
                "id": item.get("id"),
                "type": item_type,
            })
            return False
        if _normalize_for_text_match(source_text) not in haystack:
            issues.append({
                "kind": "source_text_not_found",
                "id": item.get("id"),
                "type": item_type,
                "source_text": source_text,
            })
            return False
        return True

    for dim in structured.get("dimensions") or []:
        if not _check(dim, "dimension"):
            # If the model claimed an explicit tolerance for a dimension
            # whose source_text we can't verify, demote that tolerance.
            tol = dim.get("tolerance") or {}
            if tol.get("source") == "explicit":
                tol["source"] = "unverified"
                dim["tolerance"] = tol

    structured["_verification"] = {
        "checked": True,
        "issue_count": len(issues),
        "issues": issues,
    }
    return structured


# Canonical reading order for views, used when sorting dimensions so the
# output JSON groups annotations by which view they belong to. Front
# comes first (primary view), orthographic projections next, then
# isometric, then callout-style views (section, detail), with
# auxiliary/other last. Unknown or null views sort to the end of the list.
_VIEW_ORDER = [
    "front", "top", "right", "left", "bottom", "back",
    "isometric", "section", "detail", "auxiliary", "other",
]


def _view_sort_key(item: dict) -> tuple:
    """Sort key that groups dimensions by view in the canonical reading
    order defined by `_VIEW_ORDER`, then sub-groups by `view_label` so
    multiple sections or details cluster cleanly (all SECTION A-A
    entries before all SECTION B-B entries, etc.). `source_text`
    provides a final tiebreaker so output is deterministic across
    runs (modulo LLM stochasticity in the upstream extraction).
    """
    view = item.get("view")
    try:
        view_index = _VIEW_ORDER.index(view) if view else len(_VIEW_ORDER)
    except ValueError:
        # View kind not in our canonical list -- sort to the end.
        view_index = len(_VIEW_ORDER)
    return (
        view_index,
        item.get("view_label") or "",
        item.get("source_text") or "",
    )


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
            f"(~{approx_tokens:,} text tokens)"
        )

        # Image payload size: this is what dominates the HTTP request body.
        # Reverse proxies (nginx, k8s ingress) often cap body size at 1-2 MB
        # by default, so it's worth surfacing the actual number.
        image_byte_sizes = [len(img) for img in images_b64]
        total_image_bytes = sum(image_byte_sizes)
        mb = 1024 * 1024
        if len(images_b64) == 1:
            print(
                f"Image payload: {total_image_bytes / mb:.2f} MB "
                f"(1 page, base64-encoded)"
            )
        else:
            avg = total_image_bytes / len(images_b64)
            print(
                f"Image payload: {total_image_bytes / mb:.2f} MB total "
                f"({len(images_b64)} pages, avg {avg / mb:.2f} MB each, base64-encoded)"
            )

        # Approximate total HTTP request body (text + images + JSON envelope
        # overhead is small enough to ignore at this resolution).
        total_body_bytes = len(prompt) + total_image_bytes
        print(f"Approx total request body: {total_body_bytes / mb:.2f} MB")

        # Soft warning if we're near a typical proxy default (1 MB on stock
        # nginx). The user may need to raise the proxy limit, lower the DPI,
        # or accept the failure -- this lets them see the cliff before they
        # fall off it.
        if total_image_bytes > mb:
            print(
                "Note: image payload exceeds 1 MB. If your endpoint is behind "
                "a default-config nginx/ingress, this may hit a 413 (Request "
                "Too Large) limit. Consider lowering --dpi or raising the "
                "proxy's client_max_body_size."
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

    structured = call_ollama(config, prompt, images_b64)

    # Post-extraction verification: catch hallucinations where the model
    # claimed source_text values that don't actually appear on the drawing.
    # This adds a `_verification` block to the output and demotes claimed-
    # explicit tolerances to "unverified" when their source_text is missing.
    structured = verify_extraction_against_text(structured, text_data)

    # Group output by view for readability. The LLM emits dimensions in
    # roughly the order it processed them, which often jumps between
    # views. Sorting here is post-processing only -- the data is
    # unchanged, just reordered, so downstream consumers that read by ID
    # (d1, d2, ...) are unaffected.
    if structured.get("dimensions"):
        structured["dimensions"].sort(key=_view_sort_key)

    n_issues = (structured.get("_verification") or {}).get("issue_count", 0)
    if n_issues:
        # Always surface this -- it's the user's signal that the model
        # may have made things up. Details are in `_verification.issues`
        # in the output JSON.
        print(
            f"Verification: flagged {n_issues} item(s) where source_text was "
            f"not found on the drawing. See `_verification` in the output JSON."
        )
    elif verbose:
        print("Verification: all source_text values matched the drawing.")

    return structured
