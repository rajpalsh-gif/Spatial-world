"""
utils/switch_helpers.py
=======================
Shared helper functions used by both stepgame/switch.py and spartun/switch.py.

Centralising functions that were copy-pasted between the two modules.

Public API
----------
    _submit(fn, *args, **kwargs) -> Future
        Submit a timed call to the shared thread pool.

    _bool_to_float(x) -> float
        Convert Optional[bool] to 0.0 / 1.0.

    _safe_mean(xs) -> float
        NaN-safe mean of a float list.

    relations_to_sentences_verbatim(relations, normalize_fn) -> str
        Format a relation list as "- head is rel of tail" bullet lines.

    match_support_sentences_to_story(story, support_sentences, story_sentences) -> List[str]
        Fuzzy-match support sentences back to verbatim story sentences.

    build_ablated_story_remove_sentences(story, support_sentences) -> Tuple[str, Dict]
        Return story text with the given support sentences removed.
"""

import re
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from utils.timing import _timed_call
from utils.parsing import split_sentences, _norm_for_match

# ── Shared thread pool ────────────────────────────────────────────────────────
_pool = ThreadPoolExecutor(max_workers=8)


def _submit(fn: Callable, *args, **kwargs) -> Future:
    """Submit *fn* to the shared pool via _timed_call; result is (output, dt_sec)."""
    return _pool.submit(_timed_call, fn, *args, **kwargs)


# ── Scalar utilities ──────────────────────────────────────────────────────────
def _bool_to_float(x: Optional[bool]) -> float:
    if x is True:
        return 1.0
    if x is False:
        return 0.0
    return 0.0


def _safe_mean(xs: List[float]) -> float:
    import numpy as np
    cleaned = [float(v) for v in xs if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(sum(cleaned) / max(1, len(cleaned)))


# ── Relation formatting ───────────────────────────────────────────────────────
def relations_to_sentences_verbatim(
    relations: List[Dict[str, Any]],
    normalize_fn: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Format a list of relation dicts as bullet lines.

    Each dict must have keys ``head``, ``tail``, ``relation``.
    *normalize_fn* is applied to the relation string; defaults to .lower().strip().
    """
    if normalize_fn is None:
        def normalize_fn(s: str) -> str:  # type: ignore[misc]
            return re.sub(r"\s+", " ", (s or "").strip().lower())

    lines = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        h = str(r.get("head", "")).strip()
        t = str(r.get("tail", "")).strip()
        rel = normalize_fn(str(r.get("relation", "")))
        if h and t and rel:
            lines.append(f"- {h} is {rel} of {t}")
    return "\n".join(lines) if lines else "- (none)"


# ── Support-sentence matching ─────────────────────────────────────────────────
def match_support_sentences_to_story(
    story: str,
    support_sentences: List[str],
    story_sentences: Optional[List[str]] = None,
) -> List[str]:
    """
    Match *support_sentences* (which may be paraphrased or slightly off) back to
    verbatim sentences from *story*.

    If *story_sentences* is already split (e.g. from dataset metadata), pass it
    directly to avoid re-splitting.
    """
    story_sents = (
        story_sentences
        if (isinstance(story_sentences, list) and story_sentences)
        else split_sentences(story)
    )
    story_norm = [_norm_for_match(x) for x in story_sents]

    matched: List[str] = []
    for ss in support_sentences or []:
        ssn = _norm_for_match(ss)
        if not ssn:
            continue
        if ssn in story_norm:
            matched.append(story_sents[story_norm.index(ssn)])
            continue
        for i, sn in enumerate(story_norm):
            if ssn in sn or sn in ssn:
                matched.append(story_sents[i])
                break

    out: List[str] = []
    seen: set = set()
    for s in matched:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


# ── Ablation ─────────────────────────────────────────────────────────────────
def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def build_ablated_story_remove_sentences(
    story: str,
    support_sentences: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Return the story with *support_sentences* removed, plus a metadata dict.

    Matching is fuzzy (lower-cased, punctuation-stripped) so minor formatting
    differences between the story and extracted support sentences don't break it.
    """
    sents = split_sentences(story)
    support_set_norm = {
        _norm_text(ss)
        for ss in (support_sentences or [])
        if isinstance(ss, str) and ss.strip()
    }

    if not support_set_norm:
        return story, {"removed_any": False, "removed_indices": [], "removed_sentences": []}

    removed_indices: List[int] = []
    removed_sentences: List[str] = []
    keep: List[str] = []

    for i, s in enumerate(sents):
        if _norm_text(s) in support_set_norm:
            removed_indices.append(i)
            removed_sentences.append(s)
        else:
            keep.append(s)

    return " ".join(keep).strip(), {
        "removed_any": bool(removed_indices),
        "removed_indices": removed_indices,
        "removed_sentences": removed_sentences,
    }
