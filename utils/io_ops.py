"""
utils/io_ops.py
===============
File I/O helpers shared across experiments (JSONL append, path utilities).
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append a JSON-serialisable object as one line to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def file_is_empty(path: Path) -> bool:
    """Return True if the file does not exist or has zero bytes."""
    return (not path.exists()) or (path.stat().st_size == 0)


def stable_prompt_id(text: str) -> str:
    """Return a 16-char hex SHA-256 prefix as a stable prompt identifier."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:16]


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a file as JSON array, JSON object, or JSONL.
    Returns a list of dicts in all cases.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
    except Exception:
        pass
    items, ok = [], True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            ok = False
            break
    if ok and items:
        return items
    # last resort: blank-line-separated JSON blocks
    chunks = [ch.strip() for ch in text.split("\n\n") if ch.strip()]
    return [json.loads(ch) for ch in chunks]
