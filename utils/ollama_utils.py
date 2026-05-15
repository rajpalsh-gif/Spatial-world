"""
utils/ollama_utils.py
=====================
Helpers for managing Ollama models (pull, list, check availability).

Usage from main.py / CLI:
    from utils.ollama_utils import pull_model, list_models, model_is_available

Or directly:
    python -c "from utils.ollama_utils import pull_model; pull_model('qwen3:14b')"
"""

import subprocess
import sys
from typing import List, Optional


def pull_model(model_name: str, *, verbose: bool = True) -> bool:
    """
    Pull an Ollama model using the `ollama pull` CLI command.

    Parameters
    ----------
    model_name : str
        The model to pull, e.g. "qwen3:14b" or "llama3.1:70b".
    verbose : bool
        If True, stream ollama's output to stdout in real time.

    Returns
    -------
    bool
        True if the pull succeeded (exit code 0), False otherwise.
    """
    if verbose:
        print(f"[ollama] Pulling model: {model_name} …")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            stdout=None if verbose else subprocess.PIPE,
            stderr=None if verbose else subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            if verbose:
                print(f"[ollama] Successfully pulled: {model_name}")
            return True
        else:
            print(
                f"[ollama] Pull failed for '{model_name}' "
                f"(exit code {result.returncode})"
            )
            if result.stderr:
                print(result.stderr.strip())
            return False
    except FileNotFoundError:
        print(
            "[ollama] ERROR: 'ollama' command not found. "
            "Install Ollama from https://ollama.com/download"
        )
        return False
    except Exception as exc:
        print(f"[ollama] Unexpected error pulling '{model_name}': {exc}")
        return False


def list_models() -> List[str]:
    """
    Return a list of Ollama model names currently available locally.

    Returns an empty list if Ollama is not installed or no models exist.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().splitlines()
        models: List[str] = []
        for line in lines[1:]:  # skip header row
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models
    except FileNotFoundError:
        return []
    except Exception:
        return []


def model_is_available(model_name: str) -> bool:
    """Return True if *model_name* is already pulled locally."""
    available = list_models()
    # Normalise: strip the ":latest" suffix that ollama sometimes appends
    def _base(name: str) -> str:
        return name.split(":")[0] if ":" not in name else name

    target = model_name.lower()
    return any(
        m.lower() == target or _base(m.lower()) == _base(target)
        for m in available
    )


def ensure_model(model_name: str, *, auto_pull: bool = True) -> bool:
    """
    Ensure *model_name* is available locally, optionally pulling it first.

    Parameters
    ----------
    model_name : str
        Ollama model tag, e.g. "qwen3:14b".
    auto_pull : bool
        If True, pull the model automatically when it is not found locally.

    Returns
    -------
    bool
        True if the model is (or becomes) available.
    """
    if model_is_available(model_name):
        print(f"[ollama] Model already available: {model_name}")
        return True
    if auto_pull:
        return pull_model(model_name)
    print(
        f"[ollama] Model '{model_name}' not found locally. "
        "Run with --pull to download it."
    )
    return False
