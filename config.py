"""Configuration loading for the PDF extractor.

The LLM endpoint, API key, model name, and runtime options live in a JSON
config file that is searched at startup. The api_key is never hardcoded
in source -- it must come from the config file or environment.
"""

import os
import json
from pathlib import Path
from typing import Optional


def load_config(path: Optional[str] = None) -> dict:
    """Load the LLM endpoint config from a JSON file. Search order:
       1. The path passed in on the command line (--config)
       2. The PDF_EXTRACTOR_CONFIG environment variable
       3. config.json next to the entry-point script

    Expected file shape:
       {
         "api_key":     "...",
         "endpoint":    "https://your-ollama-host.example.com/api/generate",
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
    # Default location: config.json next to the directory containing this file.
    candidates.append(Path(__file__).parent / "config.json")

    for p in candidates:
        if p.exists() and p.is_file():
            cfg = json.loads(p.read_text(encoding="utf-8"))
            if not cfg.get("api_key"):
                raise ValueError(f"Config file {p} is missing required field 'api_key'")
            return {
                "api_key": cfg["api_key"],
                # Placeholder endpoint -- replace in your config.json
                "endpoint": cfg.get("endpoint", "https://ollama.example.com/api/generate"),
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
