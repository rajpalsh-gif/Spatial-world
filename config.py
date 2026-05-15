"""
config.py
=========
Central configuration for all experiments.

Setup
-----
1. Copy .env.example to .env
2. Add your OPENAI_API_KEY to .env
3. Run any experiment via:  python main.py <experiment> [--model MODEL] [--pull]
"""

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; fall back to os.environ only

# ── API Keys ────────────────────────────────────────────────────────────────
# Loaded from .env file or the system environment.  Never hardcode keys here.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Default models ───────────────────────────────────────────────────────────
DEFAULT_GPT_MODEL          = "gpt-5.1"
DEFAULT_GPT_MINI_MODEL     = "gpt-5-mini"
DEFAULT_OLLAMA_MODEL       = "qwen3:14b"
STEPGAME_OLLAMA_MODEL      = "qwen3:14b"   # used in stepgame/pipeline.py
SPARTUN_OLLAMA_MODEL       = "llama3.1:70b"
TEXT_ONLY_OLLAMA_MODEL     = "qwen3:32b"   # used in spartun/text_only.py
RESQ_OLLAMA_MODEL          = "qwen3:32b"
