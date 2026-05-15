"""
utils/parsing.py
================
Shared JSON-parsing and text-normalization helpers used across all switch pipelines.
"""

import re
import json
from typing import Any, Dict, List


def _clip01(x: float) -> float:
    """Clip a float to [0, 1]."""
    return float(max(0.0, min(1.0, x)))


def extract_json_object(raw: str) -> Dict[str, Any]:
    """
    Robustly extract the first JSON object from a (possibly messy) string.
    Tries: direct parse → regex-slice → trailing-comma strip.
    Returns a dict with _parse_error=True on total failure.
    """
    raw = (raw or "").strip()
    try:
        out = json.loads(raw)
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {"_parse_error": True, "_raw": raw}
    block = m.group(0)
    try:
        return json.loads(block)
    except Exception:
        block2 = re.sub(r",\s*([\]}])", r"\1", block)
        try:
            return json.loads(block2)
        except Exception:
            return {"_parse_error": True, "_raw": raw, "_json_block": block}


def split_sentences(text: str) -> List[str]:
    """Split text on sentence-ending punctuation followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _norm_for_match(s: str) -> str:
    """Normalise a sentence for fuzzy support-sentence matching."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\.\!\?]", "", s)
    return s.strip()
