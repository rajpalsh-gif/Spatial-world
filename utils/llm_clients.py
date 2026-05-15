"""
utils/llm_clients.py
====================
Single place for every LLM client and call helper used across the repository.

All other modules should import from here instead of defining their own clients.

Public API
----------
    get_openai_client()                          -> OpenAI
    get_ollama_lm(model, temperature, seed, ...)  -> OllamaLLM

    call_gpt(prompt, model, retries)             -> str
        Simple Responses-API call; no system prompt.

    call_gpt_nlp(prompt, model)                  -> str
        GPT call with NLP-assistant system prompt (paraphrasing / negation /
        keyword extraction).  Used by both switch files.

    call_gpt_reasoning(prompt, model, effort)    -> Tuple[str, Dict]
        GPT call with reasoning effort set.  Returns (text, token_info_dict).
        Used as the stepgame "Ollama-llama" replacement that actually calls
        GPT-5.1 with medium reasoning.

    call_ollama(prompt, model, temperature, seed, num_predict) -> str
        Thin Ollama wrapper (no logging).  Used by spartun/pipeline.py.

    call_ollama_logged(prompt, model, temperature, num_predict) -> str
        Ollama call that also logs token counts and elapsed time.
        Used by spartun/switch.py.

    empty_token_info()                           -> Dict
        Zero-valued token-info dict (returned on failures).
"""

import os
import time
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI

from config import OPENAI_API_KEY


def _effective_temperature(requested: float) -> float:
    """Return the temperature to actually use.

    If the caller passed ``OVERRIDE_TEMPERATURE`` via the environment
    (set by ``main.py --temperature``), that value takes precedence.
    """
    override = os.environ.get("OVERRIDE_TEMPERATURE", "").strip()
    if override:
        try:
            return float(override)
        except ValueError:
            pass
    return requested

# ─────────────────────────────────────────────────────────────────────────────
# Internal token-info sentinel (re-exported for callers that need it)
# ─────────────────────────────────────────────────────────────────────────────

def empty_token_info() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_duration_ns": 0,
        "prompt_eval_duration_ns": 0,
        "eval_duration_ns": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client (cached singleton)
# ─────────────────────────────────────────────────────────────────────────────

_openai_client: Optional[OpenAI] = None


def get_openai_client() -> Optional[OpenAI]:
    """Return a cached OpenAI client.  Returns None if no API key is set."""
    global _openai_client
    if _openai_client is None and OPENAI_API_KEY:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ─────────────────────────────────────────────────────────────────────────────
# OllamaLLM client (cached per (model, temperature, seed) key)
# ─────────────────────────────────────────────────────────────────────────────

_ollama_cache: Dict[Tuple[str, float, int, int], Any] = {}


def get_ollama_lm(
    model: str,
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 2500,
):
    """Return a cached OllamaLLM instance for the given parameters."""
    key = (model, temperature, seed, num_predict)
    if key not in _ollama_cache:
        try:
            from langchain_ollama import OllamaLLM
        except ImportError as exc:
            raise ImportError(
                "langchain-ollama is required.  "
                "Install it with:  pip install langchain-ollama"
            ) from exc
        _ollama_cache[key] = OllamaLLM(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            seed=seed,
            reasoning=True,
        )
    return _ollama_cache[key]


# ─────────────────────────────────────────────────────────────────────────────
# GPT helpers
# ─────────────────────────────────────────────────────────────────────────────

def call_gpt(
    prompt: str,
    model: str = "gpt-5.1",
    retries: int = 3,
) -> str:
    """
    Simple GPT call via the Responses API (no system prompt).
    Retries with exponential back-off on transient errors.
    Returns the response text, or "" on failure.
    """
    client = get_openai_client()
    if client is None:
        print("[llm_clients] WARNING: OPENAI_API_KEY not set – skipping GPT call.")
        return ""
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
            )
            try:
                return (resp.output_text or "").strip()
            except AttributeError:
                out = []
                for item in getattr(resp, "output", []) or []:
                    for c in item.get("content", []) or []:
                        if c.get("type") in ("output_text", "text"):
                            out.append(c.get("text", ""))
                return "\n".join(out).strip()
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [GPT retry {attempt + 1}/{retries}] {exc} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  [GPT error after {retries} attempts] {exc}")
                return ""
    return ""


def call_gpt_nlp(prompt: str, model: str = "gpt-5-mini") -> str:
    """
    GPT call with NLP-assistant system prompt.
    Used for paraphrasing, negation, and keyword extraction tasks.
    Returns the response text, or "" on failure.
    """
    client = get_openai_client()
    if client is None:
        print("[llm_clients] WARNING: OPENAI_API_KEY not set – skipping GPT call.")
        return ""
    try:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a helpful NLP assistant used for paraphrasing, "
                                "negation, and keyword extraction."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
            ],
        )
        return (response.output_text or "").strip()
    except Exception as exc:
        print(f"[llm_clients] GPT NLP call failed: {exc}")
        return ""


def call_gpt_reasoning(
    prompt: str,
    model: str = "gpt-5.1",
    effort: str = "medium",
) -> Tuple[str, Dict[str, Any]]:
    """
    GPT call with reasoning effort enabled.
    Returns (response_text, token_info_dict).
    token_info_dict keys: prompt_tokens, completion_tokens, total_duration_ns,
                          prompt_eval_duration_ns, eval_duration_ns.
    """
    client = get_openai_client()
    if client is None:
        print("[llm_clients] WARNING: OPENAI_API_KEY not set – skipping GPT call.")
        return '{"answer": "", "justification": "OPENAI_API_KEY not set"}', empty_token_info()
    try:
        response = client.responses.create(
            model=model,
            reasoning={"effort": effort},
            input=[{"role": "user", "content": prompt}],
        )
        text = (response.output_text or "").strip()
        token_info = {
            "prompt_tokens": getattr(response.usage, "input_tokens", 0) if response.usage else 0,
            "completion_tokens": getattr(response.usage, "output_tokens", 0) if response.usage else 0,
            "total_duration_ns": 0,
            "prompt_eval_duration_ns": 0,
            "eval_duration_ns": 0,
        }
        return text, token_info
    except Exception as exc:
        print(f"[llm_clients] GPT reasoning call failed: {exc}")
        return '{"answer": "", "justification": "gpt call failed"}', empty_token_info()


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(
    prompt: str,
    model: str = "qwen3:14b",
    temperature: float = 0.0,
    seed: int = 42,
    num_predict: int = 5000,
) -> str:
    """
    Simple Ollama call (no token logging).
    Returns the response text, or "" on failure.
    """
    temperature = _effective_temperature(temperature)
    try:
        lm = get_ollama_lm(model=model, temperature=temperature, seed=seed, num_predict=num_predict)
        result = lm.invoke(prompt)
        return result if isinstance(result, str) else str(result)
    except Exception as exc:
        print(f"[llm_clients] Ollama call failed ({model}): {exc}")
        return '{"justification": "ollama call failed", "answer": ""}'


def call_ollama_logged(
    prompt: str,
    model: str = "qwen3:14b",
    temperature: float = 0.0,
    num_predict: int = 5000,
) -> str:
    """
    Ollama call that logs input/output token counts and elapsed time.
    Returns the response text, or "" on failure.
    """
    temperature = _effective_temperature(temperature)
    try:
        from langchain_ollama import OllamaLLM
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required.  Install with:  pip install langchain-ollama"
        ) from exc

    try:
        llm = OllamaLLM(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
            reasoning=True,
        )
        input_tokens = llm.get_num_tokens(prompt)
        t0 = time.perf_counter()
        resp = llm.invoke(prompt)
        t1 = time.perf_counter()
        out = resp if isinstance(resp, str) else str(resp)
        output_tokens = llm.get_num_tokens(out)
        elapsed = t1 - t0
        print(
            f"  [Ollama {model}] {len(out)} chars "
            f"| in_tok={input_tokens} | out_tok={output_tokens} "
            f"| {elapsed:.2f}s"
        )
        return out
    except Exception as exc:
        print(f"[llm_clients] Ollama logged call failed ({model}): {exc}")
        return '{"justification": "ollama call failed", "answer": ""}'
