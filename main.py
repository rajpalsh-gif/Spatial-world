"""
main.py
=======
Single entry point to run any experiment in this repository.

Usage
-----
    python main.py <experiment> [options]

Available experiments
---------------------
  stepgame-pipeline     Grid pipeline / validation on StepGame
  stepgame-switch       Switch-GPT pipeline on StepGame
  spartun-pipeline      Full-grid / pruned-grid pipeline on SPaRTUN
  spartun-text          Text-only + text+relations baselines on SPaRTUN
  spartun-switch        Switch pipeline on SPaRTUN
  error-taxonomy        StepGame error taxonomy (GPT-assisted classification)
  analysis-spartun      SPaRTUN switch analysis (23 tables)
  resq                  ResQ pipeline (GPT-5.1 throughout)

Options
-------
  --model MODEL         Override the Ollama model for the selected experiment
                        (e.g. qwen3:14b, llama3.1:70b)
  --gpt-model MODEL     Override the GPT/OpenAI model
                        (e.g. gpt-5.1, gpt-5-mini)
  --temperature TEMP    Override the Ollama sampling temperature
                        (0.0 = greedy/deterministic, e.g. --temperature 0.7)
  --pull                Pull the Ollama model via `ollama pull` before running.
                        Requires the Ollama daemon to be running.

All experiment-specific config (input paths, model names, thresholds) lives
inside each module under stepgame/, spartun/, analysis/, or resq/.
The shared API key lives in config.py (loaded from .env).

Examples
--------
    python main.py stepgame-switch
    python main.py spartun-switch --model llama3.1:70b --pull
    python main.py spartun-switch --model qwen3:14b --temperature 0.7
    python main.py resq --gpt-model gpt-5-mini
"""

import os
import sys
import argparse
import runpy

from config import DEFAULT_GPT_MODEL, DEFAULT_OLLAMA_MODEL

# Map CLI name → module path inside the package tree (runpy target)
EXPERIMENTS = {
    "stepgame-pipeline":  "stepgame.pipeline",
    "stepgame-grid":      "stepgame.pipeline",
    "stepgame-switch":    "stepgame.switch",
    "spartun-pipeline":   "spartun.pipeline",
    "spartun-grid":       "spartun.pipeline",
    "spartun-text":       "spartun.text_only",
    "spartun-switch":     "spartun.switch",
    "error-taxonomy":     "analysis.error_taxonomy",
    "stepgame-taxonomy":  "analysis.error_taxonomy",
    "analysis-spartun":   "analysis.spartun_switch",
    "spartun-switch-analysis": "analysis.spartun_switch",
    "resq":               "resq.pipeline",
    "resq-pipeline":      "resq.pipeline",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Spatial reasoning experiments runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [f"  {n:<25} → {m}" for n, m in EXPERIMENTS.items()]
        ),
    )
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENTS.keys()),
        metavar="experiment",
        help=(
            "Which experiment to run. Choices: "
            + ", ".join(EXPERIMENTS.keys())
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="OLLAMA_MODEL",
        help=(
            f"Override the Ollama model (default from config: {DEFAULT_OLLAMA_MODEL}). "
            "Example: --model qwen3:14b"
        ),
    )
    parser.add_argument(
        "--gpt-model",
        default=None,
        metavar="GPT_MODEL",
        help=(
            f"Override the GPT/OpenAI model (default from config: {DEFAULT_GPT_MODEL}). "
            "Example: --gpt-model gpt-5-mini"
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        metavar="TEMP",
        help=(
            "Override the Ollama sampling temperature (0.0 = greedy / deterministic). "
            "Example: --temperature 0.7"
        ),
    )
    parser.add_argument(
        "--pull",
        action="store_true",
        help="Pull the Ollama model before running (requires Ollama daemon).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ── Ollama model override ────────────────────────────────────────────────
    if args.model:
        os.environ["OVERRIDE_OLLAMA_MODEL"] = args.model
        if args.pull:
            from utils.ollama_utils import ensure_model
            ok = ensure_model(args.model, auto_pull=True)
            if not ok:
                print(f"[main] WARNING: could not pull '{args.model}'. Continuing anyway.")
        else:
            from utils.ollama_utils import model_is_available
            if not model_is_available(args.model):
                print(
                    f"[main] WARNING: model '{args.model}' does not appear to be pulled locally. "
                    "Run with --pull to download it, or ensure the Ollama daemon is running."
                )

    # ── GPT model override ───────────────────────────────────────────────────
    if args.gpt_model:
        os.environ["OVERRIDE_GPT_MODEL"] = args.gpt_model

    # ── Temperature override ─────────────────────────────────────────────────
    if args.temperature is not None:
        os.environ["OVERRIDE_TEMPERATURE"] = str(args.temperature)

    # ── Run the experiment ───────────────────────────────────────────────────
    module_path = EXPERIMENTS[args.experiment]
    print(f"[main] Running: {module_path}")
    if args.model:
        print(f"[main]   Ollama model : {args.model}")
    if args.gpt_model:
        print(f"[main]   GPT model    : {args.gpt_model}")
    if args.temperature is not None:
        print(f"[main]   Temperature  : {args.temperature}")

    runpy.run_module(module_path, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
