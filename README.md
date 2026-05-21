# DrawingAnalyzer

A pipeline for extracting structured data from engineering drawing PDFs
using a multimodal language model running on a local Ollama instance.
Designed as the artifact for my bachelor's thesis.

The pipeline reads a PDF, extracts text-with-coordinates from it
deterministically, renders each page as a PNG image, and asks an Ollama
model to populate a fixed JSON schema describing the drawing. A
verification step then checks that every dimension's quoted source text
actually appears on the drawing, flagging likely hallucinations.

## Requirements

- Python 3.10+
- [PyMuPDF](https://pymupdf.readthedocs.io/) (`pip install pymupdf`)
- An [Ollama](https://ollama.com/) instance reachable over HTTP, with a
  multimodal model loaded. Tested with `qwen3.5`, `qwen3-vl`, and
  `gemma3`.

## Setup

Clone the repository and copy the example config:

```
git clone https://github.com/Gonome3/DrawingAnalyzer.git
cd DrawingAnalyzer
cp config.example.json config.json
```

Edit `config.json` with your Ollama endpoint, API key (if your endpoint
needs one), and model name. The `api_key` is never read from anywhere
other than the config file, so it does not get committed to source.

The config file's `num_ctx` and `num_predict` override Ollama's tiny
defaults (2048 / 128) which are far too small for this pipeline. The
defaults of 65536 / 8192 are fine for one- to three-sheet drawings; bump
`num_ctx` to 131072 for large multi-sheet assemblies.

A short note about HTTPS timeouts: large models can take 30-120 seconds
to respond on the first call after a cold load. The pipeline uses
streaming responses to keep proxies happy, but if you sit behind a
strict reverse proxy you may still need to raise its timeout.

### Multiple model configurations

You can keep several config files side by side and switch between them
with `--config`. The repository includes a few:

- `config-qwen3.5-32b.json`
- `config-qwen3-vl-32b.json`
- `config-qwen3-vl-32b-thinking-q8.json`
- `config-gemma4.json`

For Qwen3 models you may set `"think": false` to suppress thinking mode
(combining thinking with `format: json` routes the JSON answer into the
thinking channel and leaves the response empty). For other models
(Gemma, Llama) just omit the `think` field; the client only sends it
when the config explicitly sets it.

## Usage

### Extracting a drawing

The main entry point is `pdf_reader.py`. It has three modes depending
on which flags you pass.

Plain text dump from the PDF (no LLM call, useful for sanity checks):

```
python pdf_reader.py drawings/21097-C-Dummy.pdf
```

Compact text-with-coordinates payload — what the LLM actually consumes
as the text portion of its prompt:

```
python pdf_reader.py drawings/21097-C-Dummy.pdf --compact
```

Full extraction pipeline (LLM call + verification + structured JSON
output):

```
python pdf_reader.py drawings/21097-C-Dummy.pdf --extract --config config-qwen3.5-32b.json
```

By default the result prints to stdout. Use `-o` to write to a file:

```
python pdf_reader.py drawings/21097-C-Dummy.pdf --extract --config config-qwen3.5-32b.json -o extractions/21097-C-Dummy.qwen3.5.json
```

Or `-o` with no value to auto-generate `output/<stem>.json`:

```
python pdf_reader.py drawings/21097-C-Dummy.pdf --extract --config config-qwen3.5-32b.json -o
```

Use `-v` to see the assembled LLM prompt and context-usage stats before
the call. Useful when you're trying to figure out how close to the
context window you are:

```
python pdf_reader.py drawings/21097-C-Dummy.pdf --extract --config config-qwen3.5-32b.json -v
```

### Flag reference for `pdf_reader.py`

| Flag | Description |
|------|-------------|
| `<filepath>` | Path to the PDF file. Optional — if omitted, the program prompts for it. |
| `--compact` | Output the compact text-with-coordinates payload instead of plain text. |
| `--extract` | Run the full extraction pipeline (LLM + verification). |
| `--config PATH` | Path to a JSON config file. Defaults: `$PDF_EXTRACTOR_CONFIG` env var, then `config.json` next to the script. |
| `--dpi N` | DPI for page rasterization sent to the LLM. Default 200. |
| `-v`, `--verbose` | Print the assembled prompt and context-usage stats before the LLM call. |
| `-o`, `--output [PATH]` | Write output to a file. With no value, auto-generates `output/<stem>.{json,txt}`. |

### Evaluating extractions against ground truth

`evaluate.py` compares one or more extraction outputs against a ground
truth set and produces aggregate precision/recall/F1 metrics plus a
per-drawing CSV.

Basic invocation — pairs each `<base>.json` in `ground_truth/` with
`<base>.json` in `extractions/`:

```
python evaluate.py ground_truth extractions
```

When extractions are tagged with a model name in the filename (e.g.
`21097-C-Dummy.qwen3.5.json`), pass the tag with `--model`:

```
python evaluate.py ground_truth extractions --model qwen3.5
```

By default the report and CSV land in `./eval_results/`. Override with
`--output`:

```
python evaluate.py ground_truth extractions --model qwen3.5 --output reports/2026-05-21
```

Use `-v` to echo per-item false-positive and false-negative listings to
stdout. Useful when debugging matches; note that it prints the actual
ground-truth content, so don't redirect verbose output into shared
files:

```
python evaluate.py ground_truth extractions --model qwen3.5 --verbose
```

### Flag reference for `evaluate.py`

| Flag | Description |
|------|-------------|
| `ground_truth_dir` | Directory containing manually-created ground truth JSON files. |
| `extractions_dir` | Directory containing extraction outputs to evaluate. |
| `--model TAG` | Model tag suffix in extraction filenames (e.g. `qwen3.5` matches `<base>.qwen3.5.json`). |
| `--output DIR` | Output directory for the report and CSV. Default `./eval_results/`. |
| `-v`, `--verbose` | Echo per-item FP/FN to stdout (debug; reveals GT contents). |

### Migrating older ground truth files

If you have ground truth files created against an earlier version of
the schema (which included a separate `features` array and a
`feature_ref` field on dimensions), `migrate_ground_truth.py` strips
both in place:

```
python migrate_ground_truth.py ground_truth
```

It prints aggregate counts only and never echoes file contents.

## End-to-end example

A typical workflow for running a comparison across one model on a set
of drawings:

```
# 1. Extract every PDF in drawings/ using the Qwen3.5 config.
for pdf in drawings/*.pdf; do
    base=$(basename "$pdf" .pdf)
    python pdf_reader.py "$pdf" \
        --extract \
        --config config-qwen3.5-32b.json \
        -o "extractions/${base}.qwen3.5.json"
done

# 2. Compare against the ground truth and produce metrics.
python evaluate.py ground_truth extractions --model qwen3.5
```

After that, `eval_results/report_qwen3.5_<timestamp>.md` contains the
aggregate metrics and `eval_results/per_drawing_qwen3.5_<timestamp>.csv`
contains one row per drawing.

To compare two models, run the loop twice with different `--config`
values and different output filename suffixes, then run `evaluate.py`
twice with the matching `--model` tags:

```
python evaluate.py ground_truth extractions --model qwen3.5
python evaluate.py ground_truth extractions --model gemma4
```

## Output schema

The extraction output is a JSON object covering: title-block metadata,
default tolerances, approvals, revisions, notes, views, dimensions
(with per-dimension tolerances, semantic role, view assignment), and a
free-form `uncovered_annotations` list for things that don't fit the
schema. See the schema definition at the top of `extraction.py` for
the full structure.

A `_verification` block is added to each extraction by the inline
verifier; it lists dimensions whose claimed `source_text` could not be
found in the drawing's actual text spans. Note that the verifier has
known false-positive cases (when the model joins separate text spans
into a single source_text string) and false-negative cases (it does not
cross-check claimed default tolerances against the drawing's default
block) — see the thesis discussion chapter for details.

## Repository layout

```
pdf_reader.py             CLI entry point
extraction.py             schema + end-to-end orchestration
llm_client.py             Ollama API client + PDF page rendering
pdf_text.py               PyMuPDF text extraction with coordinates
config.py                 config file loading
evaluate.py               compare extractions against ground truth
migrate_ground_truth.py   strip features array from old GT files
config.example.json       template config (copy to config.json)
config-*.json             per-model config files
ground_truth/             manually-created reference outputs (gitignored)
extractions/              produced extraction outputs (gitignored)
eval_results/             reports from evaluate.py (gitignored)
report/                   LaTeX source for the thesis report
```

## License

See license included

