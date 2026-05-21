"""Evaluation script for the drawing extraction pipeline.

Compares extraction outputs against a ground truth set and produces
per-category precision/recall/F1 metrics. Operates purely on JSON files
and never modifies its inputs.

Matching strategy
-----------------
  * Dimensions: primary key is the normalized `source_text`. Fallback is
    the (nominal, kind, unit) tuple, used only when source_text fails
    and exactly one unmatched ground-truth candidate remains.
  * Approvals: primary key is (name, date) with name-only as fallback.
    The `role` label is excluded from the identity because GT and
    extraction may legitimately use different terms for the same
    approval (e.g. Swedish "Konstruerad" vs schema enum "drawn"); role
    is scored as a field on matched approvals but not as part of the key.
  * Notes: matched by `number`. Revisions by `rev`. Views by (kind, label).

Tolerant comparison for semantic_role
-------------------------------------
The schema's `semantic_role` enum does not cover every feature kind
that appears on real drawings (spherical features, for instance), so
both GT and extraction occasionally use ad-hoc values outside the
enum. To avoid punishing near-equivalent labels, `semantic_role` uses
token-containment equality: `spherical_diameter` and
`spherical_end_diameter` are treated as equivalent because their
underscore-split tokens have a subset relationship. Other fields
keep strict equality. This is a deliberate, scoped loosening rather
than a general fuzziness in the matcher.

Numerical comparison
--------------------
Absolute tolerance of 1e-6 throughout. For `tolerance.plus` and
`tolerance.minus` (both in per-dimension tolerances and in
default_tolerances), comparison is on absolute values rather than signed
values, because the extraction model is inconsistent about whether
those fields carry sign or magnitude. Additionally, the value 0 and
null are treated as equivalent for these tolerance fields, because
both encode "no usable tolerance information" for downstream consumers
and the extraction model frequently writes 0 where the schema would
expect null.

Scoring layers
--------------
  * Discovery (list-level): TP/FP/FN of matched/unmatched items.
  * Field accuracy (per matched item): for each tracked field, did GT
    and extraction agree? Null-on-both-sides counts as correct here but
    does not contribute to discovery rates.
  * All-or-nothing (per matched item): summary boolean "every tracked
    field was correct on this item."

Outputs (per run, in --output dir)
----------------------------------
  * report_<tag>_<timestamp>.md       aggregate metrics + per-category breakdown
  * per_drawing_<tag>_<timestamp>.csv one row per drawing for downstream analysis

CLI
---
    python evaluate.py <ground_truth_dir> <extractions_dir>
        [--model MODEL_TAG]      filename suffix matched in extractions dir
        [--output OUTPUT_DIR]    default ./eval_results/
        [-v | --verbose]         echo per-item FP/FN to console
                                 (debug only; reveals GT contents)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

FLOAT_EPS = 1e-6


def normalize_text(text: Any) -> str:
    """Mirror extraction.py's source_text normalizer so the eval script
    matches dimensions the same way the inline verifier does."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.replace("+/-", "±").replace("+-", "±")
    return s


def floats_equal(a: Any, b: Any) -> bool:
    """Numerical equality with FLOAT_EPS tolerance. Null == null."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= FLOAT_EPS
    except (TypeError, ValueError):
        return False


def floats_equal_magnitude(a: Any, b: Any) -> bool:
    """Like floats_equal but on absolute values. Used for tolerance
    plus/minus fields where the model is inconsistent about sign."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return abs(abs(float(a)) - abs(float(b))) <= FLOAT_EPS
    except (TypeError, ValueError):
        return False


def tolerance_value_equal(a: Any, b: Any) -> bool:
    """Equality for tolerance plus/minus fields (per-dimension and
    default_tolerances). Treats null and 0 (within FLOAT_EPS) as
    equivalent, because both encode "no usable tolerance information"
    for any downstream consumer, and the extraction model frequently
    writes 0 where the schema would expect null. Outside that special
    case, falls back to absolute-value comparison so the sign-convention
    inconsistency on plus/minus doesn't cause spurious mismatches."""
    def _is_absent(v: Any) -> bool:
        if v is None:
            return True
        try:
            return abs(float(v)) <= FLOAT_EPS
        except (TypeError, ValueError):
            return False
    a_absent = _is_absent(a)
    b_absent = _is_absent(b)
    if a_absent and b_absent:
        return True
    if a_absent or b_absent:
        return False
    return floats_equal_magnitude(a, b)


def strs_equal(a: Any, b: Any) -> bool:
    """Case- and whitespace-insensitive string equality. Null == null."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return normalize_text(a) == normalize_text(b)


def values_equal(a: Any, b: Any) -> bool:
    """Type-aware scalar equality dispatcher. Used for fields whose type
    isn't fixed by the schema (mixes of int, float, str, bool, None)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return floats_equal(a, b)
    return strs_equal(a, b)


def semantic_roles_equivalent(a: Any, b: Any) -> bool:
    """Tolerant equality for the `semantic_role` field only.

    The schema's semantic_role enum doesn't cover every kind of feature
    that appears on real drawings (spherical features, for instance),
    so both ground truth and extraction occasionally use ad-hoc values
    outside the enum. Strict equality is overly punitive when the two
    sides differ only by an extra qualifier
    (e.g. `spherical_diameter` vs `spherical_end_diameter`).

    This helper accepts a match when either:
      * strict (normalized) string equality holds, or
      * the underscore-separated tokens of one side are a subset of
        the other's (so `spherical_diameter` matches
        `spherical_end_diameter`, but `hole_diameter` does NOT match
        `hole_depth`).

    Used exclusively for semantic_role -- other fields keep strict
    equality. The loosening is a documented evaluation choice rather
    than a property of the schema.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if strs_equal(a, b):
        return True
    a_tokens = {t for t in normalize_text(a).split("_") if t}
    b_tokens = {t for t in normalize_text(b).split("_") if t}
    if not a_tokens or not b_tokens:
        return False
    return a_tokens.issubset(b_tokens) or b_tokens.issubset(a_tokens)


# ---------------------------------------------------------------------------
# Match-key functions per list category
# ---------------------------------------------------------------------------

def dim_primary_key(d: dict) -> str:
    return normalize_text(d.get("source_text"))


def dim_fallback_key(d: dict) -> tuple:
    nom = d.get("nominal")
    return (
        round(float(nom), 6) if nom is not None else None,
        (d.get("kind") or "").lower(),
        (d.get("unit") or "").lower(),
    )


def note_key(n: dict) -> Any:
    return n.get("number")


def approval_key(a: dict) -> tuple:
    """Match approvals on (name, date). The role label is intentionally
    excluded from the identity because it varies across drawings: ground
    truth often preserves the raw Swedish CAD text (e.g.
    "Konstruerad/Designed") while the extraction maps to the schema's
    English enum ("drawn"). Role is still compared as a scored field on
    matched approvals; just not used as part of the match key.
    """
    return (normalize_text(a.get("name")), (a.get("date") or "").lower())


def approval_fallback_key(a: dict) -> str:
    """Fallback to name-only matching when date is missing or differs
    between sides (rare but possible on drawings where the model failed
    to parse the date)."""
    return normalize_text(a.get("name"))


def revision_key(r: dict) -> str:
    return (r.get("rev") or "").lower()


def view_key(v: dict) -> tuple:
    return ((v.get("kind") or "").lower(), normalize_text(v.get("label")))


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    matched: list = field(default_factory=list)   # list[tuple[dict, dict]]
    fps: list = field(default_factory=list)       # extraction-only (false positives)
    fns: list = field(default_factory=list)       # GT-only (false negatives)


def _empty_key(k: Any) -> bool:
    return k is None or k == "" or k == () or k == (None,)


def match_list(
    gt_items,
    ext_items,
    primary_key: Callable[[dict], Any],
    fallback_key: Callable[[dict], Any] = None,
) -> MatchResult:
    """Greedy 1-to-1 matching between GT and extraction lists.

    Primary key is tried first; the first unmatched GT candidate with a
    matching key is paired with each extraction item. Fallback key is
    then tried on still-unmatched items, but only matches when exactly
    one unmatched GT candidate has the same fallback key (to avoid
    silently picking a wrong pairing when multiple share a nominal).
    """
    gt = list(gt_items or [])
    ex = list(ext_items or [])
    gt_used = [False] * len(gt)
    ex_used = [False] * len(ex)
    matched = []

    # --- primary pass
    gt_by_pkey: dict = {}
    for i, g in enumerate(gt):
        k = primary_key(g)
        if _empty_key(k):
            continue
        gt_by_pkey.setdefault(k, []).append(i)

    for j, e in enumerate(ex):
        k = primary_key(e)
        if _empty_key(k):
            continue
        for i in gt_by_pkey.get(k, []):
            if not gt_used[i]:
                matched.append((gt[i], e))
                gt_used[i] = True
                ex_used[j] = True
                break  # take first unused GT candidate

    # --- fallback pass (only on still-unmatched items)
    if fallback_key is not None:
        for j, e in enumerate(ex):
            if ex_used[j]:
                continue
            k = fallback_key(e)
            if _empty_key(k):
                continue
            candidates = [
                i for i, g in enumerate(gt)
                if not gt_used[i] and fallback_key(g) == k
            ]
            if len(candidates) == 1:
                i = candidates[0]
                matched.append((gt[i], e))
                gt_used[i] = True
                ex_used[j] = True

    fns = [gt[i] for i, u in enumerate(gt_used) if not u]
    fps = [ex[j] for j, u in enumerate(ex_used) if not u]
    return MatchResult(matched=matched, fps=fps, fns=fns)


# ---------------------------------------------------------------------------
# Per-field scoring containers
# ---------------------------------------------------------------------------

@dataclass
class FieldScore:
    correct: int = 0
    total: int = 0


@dataclass
class CategoryMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    fields: dict = field(default_factory=dict)  # dict[str, FieldScore]
    perfect_items: int = 0  # matched items where every tracked field is correct

    def field_check(self, name: str, ok: bool) -> None:
        fs = self.fields.setdefault(name, FieldScore())
        fs.total += 1
        if ok:
            fs.correct += 1

    def add(self, other: "CategoryMetrics") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.perfect_items += other.perfect_items
        for name, fs in other.fields.items():
            mine = self.fields.setdefault(name, FieldScore())
            mine.correct += fs.correct
            mine.total += fs.total


def pr_f1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# Per-matched-item scoring
# ---------------------------------------------------------------------------

# Field lists for the various list categories. Tolerance fields are
# handled separately because plus/minus use magnitude-aware equality.
DIMENSION_SIMPLE_FIELDS: list = [
    "kind", "nominal", "unit",
    "semantic_role", "view", "view_label", "axis", "modifier",
]
DIMENSION_NUMERIC_FIELDS = {"nominal"}


def score_dimension(gt: dict, ext: dict, m: CategoryMetrics) -> None:
    all_ok = True

    for name in DIMENSION_SIMPLE_FIELDS:
        if name in DIMENSION_NUMERIC_FIELDS:
            ok = floats_equal(gt.get(name), ext.get(name))
        elif name == "semantic_role":
            ok = semantic_roles_equivalent(gt.get(name), ext.get(name))
        else:
            ok = strs_equal(gt.get(name), ext.get(name))
        m.field_check(name, ok)
        if not ok:
            all_ok = False

    gt_tol = gt.get("tolerance") or {}
    ext_tol = ext.get("tolerance") or {}
    for sub_name, cmp in (
        ("source", strs_equal),
        ("plus", tolerance_value_equal),
        ("minus", tolerance_value_equal),
    ):
        ok = cmp(gt_tol.get(sub_name), ext_tol.get(sub_name))
        m.field_check(f"tolerance.{sub_name}", ok)
        if not ok:
            all_ok = False

    if all_ok:
        m.perfect_items += 1


def score_simple(
    gt: dict, ext: dict, fields: list, m: CategoryMetrics,
) -> None:
    """Per-field accuracy for list items whose fields are all plain scalars."""
    all_ok = True
    for fn in fields:
        ok = values_equal(gt.get(fn), ext.get(fn))
        m.field_check(fn, ok)
        if not ok:
            all_ok = False
    if all_ok:
        m.perfect_items += 1


# ---------------------------------------------------------------------------
# Sub-object scoring (returns correct/total counts; no discovery component)
# ---------------------------------------------------------------------------

def compare_fields(
    gt, ext, fields: list,
    magnitude_fields=None,
):
    """Compare a list of fields on two dict-like objects.

    Returns (correct_count, total_count). When both objects are absent
    (None), all fields trivially agree on absence and the totals still
    contribute -- callers that want to skip absent-on-both can check
    truthiness of `gt` and `ext` before invoking.
    """
    gt = gt or {}
    ext = ext or {}
    mag = magnitude_fields or set()
    correct = 0
    for fn in fields:
        gt_v = gt.get(fn)
        ext_v = ext.get(fn)
        if fn in mag:
            # Magnitude fields are used for tolerance plus/minus only,
            # where 0 and null both mean "no usable tolerance info".
            ok = tolerance_value_equal(gt_v, ext_v)
        else:
            ok = values_equal(gt_v, ext_v)
        if ok:
            correct += 1
    return correct, len(fields)


# ---------------------------------------------------------------------------
# Per-drawing orchestration
# ---------------------------------------------------------------------------

METADATA_FIELDS = [
    "drawing_number", "revision", "title", "size", "scale", "units",
    "material", "finish", "projection", "company",
]

DEFAULT_TOL_GROUPS = ["linear_2_place", "linear_3_place", "angular"]
DEFAULT_TOL_FIELDS = ["plus", "minus", "unit"]

SURFACE_FINISH_FIELDS = ["minimum_ra", "applies_unless_otherwise_specified"]

DRAWING_SUMMARY_FIELDS = ["part_type", "primary_function"]

APPROVAL_FIELDS = ["role", "name", "date"]
REVISION_FIELDS = ["rev", "description", "date", "approved_by"]
NOTE_FIELDS = ["number", "text"]
VIEW_FIELDS = ["kind", "label", "scale"]


@dataclass
class DrawingResult:
    name: str
    # list-based categories (discovery + field accuracy)
    approvals: CategoryMetrics = field(default_factory=CategoryMetrics)
    revisions: CategoryMetrics = field(default_factory=CategoryMetrics)
    notes: CategoryMetrics = field(default_factory=CategoryMetrics)
    views: CategoryMetrics = field(default_factory=CategoryMetrics)
    dimensions: CategoryMetrics = field(default_factory=CategoryMetrics)
    # flat field-accuracy categories (no discovery component)
    metadata_correct: int = 0
    metadata_total: int = 0
    default_tol_correct: int = 0
    default_tol_total: int = 0
    surface_finish_correct: int = 0
    surface_finish_total: int = 0
    summary_correct: int = 0
    summary_total: int = 0
    # diagnostics
    uncovered_gt: int = 0
    uncovered_ext: int = 0
    hallucinations: int = 0
    # verbose-mode payload
    fps: dict = field(default_factory=dict)  # category -> list of items
    fns: dict = field(default_factory=dict)


def _score_list(
    gt_items, ext_items,
    primary_key, fallback_key,
    scorer: Callable[[dict, dict, CategoryMetrics], None],
    fps_out: list, fns_out: list,
) -> CategoryMetrics:
    m = CategoryMetrics()
    mr = match_list(gt_items, ext_items, primary_key, fallback_key)
    m.tp = len(mr.matched)
    m.fp = len(mr.fps)
    m.fn = len(mr.fns)
    for g, e in mr.matched:
        scorer(g, e, m)
    fps_out.extend(mr.fps)
    fns_out.extend(mr.fns)
    return m


def score_drawing(gt: dict, ext: dict, name: str) -> DrawingResult:
    res = DrawingResult(name=name)

    # --- drawing metadata (flat + nested weight + nested sheet)
    gt_d = gt.get("drawing") or {}
    ext_d = ext.get("drawing") or {}
    c, t = compare_fields(gt_d, ext_d, METADATA_FIELDS)
    res.metadata_correct += c
    res.metadata_total += t
    c, t = compare_fields(gt_d.get("weight"), ext_d.get("weight"), ["value", "unit"])
    res.metadata_correct += c
    res.metadata_total += t
    c, t = compare_fields(gt_d.get("sheet"), ext_d.get("sheet"), ["current", "total"])
    res.metadata_correct += c
    res.metadata_total += t

    # --- default tolerances (3 groups, plus/minus magnitude-aware)
    gt_dt = gt.get("default_tolerances") or {}
    ext_dt = ext.get("default_tolerances") or {}
    for grp in DEFAULT_TOL_GROUPS:
        c, t = compare_fields(
            gt_dt.get(grp), ext_dt.get(grp), DEFAULT_TOL_FIELDS,
            magnitude_fields={"plus", "minus"},
        )
        res.default_tol_correct += c
        res.default_tol_total += t

    # --- surface finish
    c, t = compare_fields(
        gt.get("surface_finish"), ext.get("surface_finish"),
        SURFACE_FINISH_FIELDS,
    )
    res.surface_finish_correct += c
    res.surface_finish_total += t

    # --- drawing_summary (description excluded; only categorical fields)
    c, t = compare_fields(
        gt.get("drawing_summary"), ext.get("drawing_summary"),
        DRAWING_SUMMARY_FIELDS,
    )
    res.summary_correct += c
    res.summary_total += t

    # --- list categories
    fps, fns = [], []
    res.approvals = _score_list(
        gt.get("approvals"), ext.get("approvals"),
        approval_key, approval_fallback_key,
        lambda g, e, m: score_simple(g, e, APPROVAL_FIELDS, m),
        fps, fns,
    )
    res.fps["approvals"] = fps
    res.fns["approvals"] = fns

    fps, fns = [], []
    res.revisions = _score_list(
        gt.get("revisions"), ext.get("revisions"),
        revision_key, None,
        lambda g, e, m: score_simple(g, e, REVISION_FIELDS, m),
        fps, fns,
    )
    res.fps["revisions"] = fps
    res.fns["revisions"] = fns

    fps, fns = [], []
    res.notes = _score_list(
        gt.get("notes"), ext.get("notes"),
        note_key, None,
        lambda g, e, m: score_simple(g, e, NOTE_FIELDS, m),
        fps, fns,
    )
    res.fps["notes"] = fps
    res.fns["notes"] = fns

    fps, fns = [], []
    res.views = _score_list(
        gt.get("views"), ext.get("views"),
        view_key, None,
        lambda g, e, m: score_simple(g, e, VIEW_FIELDS, m),
        fps, fns,
    )
    res.fps["views"] = fps
    res.fns["views"] = fns

    fps, fns = [], []
    res.dimensions = _score_list(
        gt.get("dimensions"), ext.get("dimensions"),
        dim_primary_key, dim_fallback_key,
        score_dimension,
        fps, fns,
    )
    res.fps["dimensions"] = fps
    res.fns["dimensions"] = fns

    # --- diagnostics
    res.uncovered_gt = len(gt.get("uncovered_annotations") or [])
    res.uncovered_ext = len(ext.get("uncovered_annotations") or [])
    res.hallucinations = int(
        (ext.get("_verification") or {}).get("issue_count", 0)
    )

    return res


# ---------------------------------------------------------------------------
# Aggregation across drawings
# ---------------------------------------------------------------------------

@dataclass
class AggregateResult:
    drawings: int = 0
    approvals: CategoryMetrics = field(default_factory=CategoryMetrics)
    revisions: CategoryMetrics = field(default_factory=CategoryMetrics)
    notes: CategoryMetrics = field(default_factory=CategoryMetrics)
    views: CategoryMetrics = field(default_factory=CategoryMetrics)
    dimensions: CategoryMetrics = field(default_factory=CategoryMetrics)
    metadata_correct: int = 0
    metadata_total: int = 0
    default_tol_correct: int = 0
    default_tol_total: int = 0
    surface_finish_correct: int = 0
    surface_finish_total: int = 0
    summary_correct: int = 0
    summary_total: int = 0
    uncovered_gt: int = 0
    uncovered_ext: int = 0
    hallucinations: int = 0


def aggregate(per_drawing: list) -> AggregateResult:
    agg = AggregateResult(drawings=len(per_drawing))
    for d in per_drawing:
        agg.approvals.add(d.approvals)
        agg.revisions.add(d.revisions)
        agg.notes.add(d.notes)
        agg.views.add(d.views)
        agg.dimensions.add(d.dimensions)
        agg.metadata_correct += d.metadata_correct
        agg.metadata_total += d.metadata_total
        agg.default_tol_correct += d.default_tol_correct
        agg.default_tol_total += d.default_tol_total
        agg.surface_finish_correct += d.surface_finish_correct
        agg.surface_finish_total += d.surface_finish_total
        agg.summary_correct += d.summary_correct
        agg.summary_total += d.summary_total
        agg.uncovered_gt += d.uncovered_gt
        agg.uncovered_ext += d.uncovered_ext
        agg.hallucinations += d.hallucinations
    return agg


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _pct(correct: int, total: int) -> str:
    if total == 0:
        return "n/a"
    return f"{100.0 * correct / total:.1f}% ({correct}/{total})"


def _prf_row(name: str, m: CategoryMetrics) -> str:
    p, r, f = pr_f1(m.tp, m.fp, m.fn)
    perfect = f"{m.perfect_items}/{m.tp}" if m.tp else "0/0"
    return (
        f"| {name} | {m.tp} | {m.fp} | {m.fn} | "
        f"{p:.3f} | {r:.3f} | {f:.3f} | {perfect} |"
    )


def build_markdown_report(agg: AggregateResult, model_tag: str) -> str:
    lines = []
    lines.append("# Evaluation report")
    lines.append("")
    lines.append(f"Model tag: `{model_tag or '(unspecified)'}`")
    lines.append(f"Drawings evaluated: {agg.drawings}")
    lines.append("")

    lines.append("## List-level metrics (discovery)")
    lines.append("")
    lines.append("| Category | TP | FP | FN | Precision | Recall | F1 | Perfect items |")
    lines.append("|----------|----|----|----|-----------|--------|----|---------------|")
    lines.append(_prf_row("Dimensions", agg.dimensions))
    lines.append(_prf_row("Notes",      agg.notes))
    lines.append(_prf_row("Views",      agg.views))
    lines.append(_prf_row("Approvals",  agg.approvals))
    lines.append(_prf_row("Revisions",  agg.revisions))
    lines.append("")

    lines.append("## Field-level accuracy on flat categories")
    lines.append("")
    lines.append("| Category | Correct |")
    lines.append("|----------|---------|")
    lines.append(f"| Drawing metadata    | {_pct(agg.metadata_correct, agg.metadata_total)} |")
    lines.append(f"| Default tolerances  | {_pct(agg.default_tol_correct, agg.default_tol_total)} |")
    lines.append(f"| Surface finish      | {_pct(agg.surface_finish_correct, agg.surface_finish_total)} |")
    lines.append(f"| Drawing summary     | {_pct(agg.summary_correct, agg.summary_total)} |")
    lines.append("")

    lines.append("## Per-field accuracy on matched dimensions")
    lines.append("")
    lines.append("| Field | Correct |")
    lines.append("|-------|---------|")
    for name in sorted(agg.dimensions.fields.keys()):
        fs = agg.dimensions.fields[name]
        lines.append(f"| {name} | {_pct(fs.correct, fs.total)} |")
    lines.append("")

    lines.append("## Diagnostics")
    lines.append("")
    lines.append(
        f"Hallucinations flagged by extraction's verifier: "
        f"**{agg.hallucinations}** across {agg.drawings} drawing(s)"
    )
    lines.append(
        f"Uncovered annotations: GT total={agg.uncovered_gt}, "
        f"extraction total={agg.uncovered_ext}"
    )
    lines.append("")
    return "\n".join(lines)


CSV_COLUMNS = [
    "drawing",
    "dim_tp", "dim_fp", "dim_fn",
    "dim_precision", "dim_recall", "dim_f1", "dim_perfect",
    "notes_tp", "notes_fp", "notes_fn",
    "notes_precision", "notes_recall", "notes_f1",
    "views_tp", "views_fp", "views_fn",
    "views_precision", "views_recall", "views_f1",
    "approvals_tp", "approvals_fp", "approvals_fn",
    "approvals_precision", "approvals_recall", "approvals_f1",
    "revisions_tp", "revisions_fp", "revisions_fn",
    "revisions_precision", "revisions_recall", "revisions_f1",
    "metadata_correct", "metadata_total",
    "default_tol_correct", "default_tol_total",
    "surface_finish_correct", "surface_finish_total",
    "summary_correct", "summary_total",
    "hallucinations",
    "uncovered_gt", "uncovered_ext",
]


def write_per_drawing_csv(path: Path, per_drawing: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for d in per_drawing:
            dp, dr, df = pr_f1(d.dimensions.tp, d.dimensions.fp, d.dimensions.fn)
            np_, nr, nf = pr_f1(d.notes.tp, d.notes.fp, d.notes.fn)
            vp, vr, vf = pr_f1(d.views.tp, d.views.fp, d.views.fn)
            ap, ar, af = pr_f1(d.approvals.tp, d.approvals.fp, d.approvals.fn)
            rp, rr, rf = pr_f1(d.revisions.tp, d.revisions.fp, d.revisions.fn)
            w.writerow({
                "drawing": d.name,
                "dim_tp": d.dimensions.tp,
                "dim_fp": d.dimensions.fp,
                "dim_fn": d.dimensions.fn,
                "dim_precision": f"{dp:.3f}",
                "dim_recall": f"{dr:.3f}",
                "dim_f1": f"{df:.3f}",
                "dim_perfect": d.dimensions.perfect_items,
                "notes_tp": d.notes.tp,
                "notes_fp": d.notes.fp,
                "notes_fn": d.notes.fn,
                "notes_precision": f"{np_:.3f}",
                "notes_recall": f"{nr:.3f}",
                "notes_f1": f"{nf:.3f}",
                "views_tp": d.views.tp,
                "views_fp": d.views.fp,
                "views_fn": d.views.fn,
                "views_precision": f"{vp:.3f}",
                "views_recall": f"{vr:.3f}",
                "views_f1": f"{vf:.3f}",
                "approvals_tp": d.approvals.tp,
                "approvals_fp": d.approvals.fp,
                "approvals_fn": d.approvals.fn,
                "approvals_precision": f"{ap:.3f}",
                "approvals_recall": f"{ar:.3f}",
                "approvals_f1": f"{af:.3f}",
                "revisions_tp": d.revisions.tp,
                "revisions_fp": d.revisions.fp,
                "revisions_fn": d.revisions.fn,
                "revisions_precision": f"{rp:.3f}",
                "revisions_recall": f"{rr:.3f}",
                "revisions_f1": f"{rf:.3f}",
                "metadata_correct": d.metadata_correct,
                "metadata_total": d.metadata_total,
                "default_tol_correct": d.default_tol_correct,
                "default_tol_total": d.default_tol_total,
                "surface_finish_correct": d.surface_finish_correct,
                "surface_finish_total": d.surface_finish_total,
                "summary_correct": d.summary_correct,
                "summary_total": d.summary_total,
                "hallucinations": d.hallucinations,
                "uncovered_gt": d.uncovered_gt,
                "uncovered_ext": d.uncovered_ext,
            })


# ---------------------------------------------------------------------------
# File pairing + CLI
# ---------------------------------------------------------------------------

def find_pairs(
    gt_dir: Path, ext_dir: Path, model_tag,
) -> list:
    """Pair each GT JSON with its corresponding extraction JSON.

    A GT file `<base>.json` is paired with either:
      * `<base>.<model_tag>.json` if model_tag is given, or
      * `<base>.json` otherwise.
    Returns list of (drawing_name, gt_path, ext_path) tuples for pairs
    where the extraction file exists. Missing extractions are logged
    to stderr.
    """
    pairs = []
    gt_files = sorted(gt_dir.glob("*.json"))
    for gt_path in gt_files:
        base = gt_path.stem
        if model_tag:
            ext_path = ext_dir / f"{base}.{model_tag}.json"
        else:
            ext_path = ext_dir / f"{base}.json"
        if not ext_path.exists():
            print(f"missing extraction for {base}: {ext_path}", file=sys.stderr)
            continue
        pairs.append((base, gt_path, ext_path))
    return pairs


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run(args: argparse.Namespace) -> int:
    gt_dir = Path(args.ground_truth_dir)
    ext_dir = Path(args.extractions_dir)
    out_dir = Path(args.output)

    if not gt_dir.is_dir():
        print(f"ground_truth_dir is not a directory: {gt_dir}", file=sys.stderr)
    if not ext_dir.is_dir():
        print(f"extractions_dir is not a directory: {ext_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(gt_dir, ext_dir, args.model)
    if not pairs:
        print("No drawings to evaluate (no matching files found).",
              file=sys.stderr)
        return 1

    per_drawing = []
    for name, gt_path, ext_path in pairs:
        try:
            gt = load_json(gt_path)
            ext = load_json(ext_path)
        except Exception as e:
            print(f"skipping {name}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            continue
        res = score_drawing(gt, ext, name)
        per_drawing.append(res)

        if args.verbose:
            print(f"--- {name} ---")
            for cat in ("dimensions", "notes", "views",
                        "approvals", "revisions"):
                fps = res.fps.get(cat, [])
                fns = res.fns.get(cat, [])
                if fps:
                    print(f"  [{cat}] FPs ({len(fps)}):")
                    for item in fps:
                        print(f"    + {json.dumps(item, ensure_ascii=False)}")
                if fns:
                    print(f"  [{cat}] FNs ({len(fns)}):")
                    for item in fns:
                        print(f"    - {json.dumps(item, ensure_ascii=False)}")

    if not per_drawing:
        print("All file pairs failed to load.", file=sys.stderr)
        return 1

    agg = aggregate(per_drawing)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.model or "untagged"
    report_path = out_dir / f"report_{tag}_{timestamp}.md"
    csv_path = out_dir / f"per_drawing_{tag}_{timestamp}.csv"

    report = build_markdown_report(agg, args.model or "")
    report_path.write_text(report, encoding="utf-8")
    write_per_drawing_csv(csv_path, per_drawing)

    print(f"Wrote report: {report_path}")
    print(f"Wrote per-drawing CSV: {csv_path}")
    print(f"\nDrawings evaluated: {agg.drawings}")
    p, r, f = pr_f1(agg.dimensions.tp, agg.dimensions.fp, agg.dimensions.fn)
    print(
        f"Dimensions: TP={agg.dimensions.tp} FP={agg.dimensions.fp} "
        f"FN={agg.dimensions.fn}  P={p:.3f} R={r:.3f} F1={f:.3f}"
    )
    print(f"Hallucinations flagged: {agg.hallucinations}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate drawing extraction outputs against ground truth.",
    )
    parser.add_argument("ground_truth_dir")
    parser.add_argument("extractions_dir")
    parser.add_argument(
        "--model",
        default=None,
        help="Model tag suffix in extraction filenames (e.g. 'qwen3-vl').",
    )
    parser.add_argument(
        "--output",
        default="./eval_results",
        help="Output directory for the report and CSV (default: ./eval_results).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Echo per-item FP/FN to console (debug; reveals GT contents).",
    )
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
