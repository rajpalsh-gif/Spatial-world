"""
utils/llm.py
============
Shared LLM token-info helper used across switch pipelines.
Each experiment file manages its own client/model instances; only the
common empty-info sentinel is centralised here.
"""

from typing import Any, Dict


def _empty_token_info() -> Dict[str, Any]:
    """Return a zeroed-out token-info dict (used as a fallback on errors)."""
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_duration_ns": 0,
        "prompt_eval_duration_ns": 0,
        "eval_duration_ns": 0,
    }
