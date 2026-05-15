# Spatial Reasoning Experiments

Code for spatial reasoning experiments across StepGame, SPaRTUN, and ResQ.

## Repo structure

```
.
├── main.py                     # single CLI entry point
├── config.py                   # env loading and shared defaults
├── stepgame/                   # StepGame pipeline and switch runs
├── spartun/                    # SPaRTUN pipeline, text-only, and switch runs
├── analysis/                   # analysis scripts
├── resq/                       # ResQ pipeline
└── utils/                      # shared helpers
```

The maintained code lives in the package folders above. Old root-level one-file copies and notebook snapshots are not needed for normal runs.

---

## Experiments

### 1. Full-grid StepGame — `stepgame/pipeline.py`

Runs the three-prompt grid pipeline on StepGame:

1. Extract relations from a paragraph (two-stage coarse → fine)
2. Build a full spatial grid and prune it to the relevant entities
3. Run coords → coarse → straight/diagonal prompts to predict the final relation

Validates against gold labels from a JSONL file.

**Entry:**
```bash
python main.py stepgame-pipeline
```

Edit `validate_jsonl(input_jsonl=..., output_jsonl=...)` at the bottom of `stepgame/pipeline.py` to change input/output paths.

---

### 2. Pruned-grid StepGame — `stepgame/pipeline.py`

The same `stepgame/pipeline.py` also handles the pruned-grid variant (entity-selection + re-pruning). Both full and pruned grid runs are controlled by the `RUN_FULL_GRID` / `RUN_PRUNED_GRID` flags inside the module.

---

### 3. StepGame error taxonomy — `analysis/error_taxonomy.py`

Applies a GPT-assisted two-call taxonomy to text-only failures:
- **Call 1 (input):** `composite_spatial`, `transitivity`
- **Call 2 (output):** `composite_failure`, `hallucination`, `linguistic_difficulty`, `transitivity_failure`, `other_reasoning_error`

Results are cached in a JSON file; re-running skips already-classified records.

**Entry:**
```bash
python main.py error-taxonomy
```

Edit `DATA_FILE`, `SWITCH_FILE`, `CACHE_FILE` at the top of `analysis/error_taxonomy.py`.

---

### 4. SPaRTUN grid pipeline — `spartun/pipeline.py`

Runs full-grid and pruned-grid QA on SPaRTUN using the prompt library in `spartun/prompts_lib.py`. Supports YN and FR question types.

**Entry:**
```bash
python main.py spartun-pipeline
```

Reads from `SRC_JSON` and writes to `OUT_JSON` (both configurable at the top of the file). `prompts_lib.py` provides `build_prompt_grid_interpretation`, `build_prompt_question_plan_yn`, `build_prompt_grid_answer_yn`, and related builders.

---

### 5. Switch-GPT on StepGame — `stepgame/switch.py`

Implements the full three-wave switch pipeline for StepGame:

- **Q1 (baseline + support):** text-only answer + hard-language scoring (via GPT)
- **Q2 (paraphrase + flip):** 3× paraphrase + 1 flip, parallel
- **Q3 (complexity):** `CL + SD + ambiguity-aware HL`
- **Switch policy:** grid answer if complexity ≥ τ_c OR trust < τ_t

Supports a `MODE="rescore"` path that re-scores complexity only (no Ollama calls).

**Entry:**
```bash
python main.py stepgame-switch
```

Edit `input_path`, `output_json`, model names, and thresholds at the bottom of `stepgame/switch.py`.

---

### 6. Switch on SPaRTUN — `spartun/switch.py`

Same switch architecture adapted for SPaRTUN (multi-label FR questions, SPaRTUN semantics block, DC/EC/TPP/NTPP candidate handling).

**Entry:**
```bash
python main.py spartun-switch
```

Edit `input_path`, `out_json`, model names, and thresholds at the bottom of `spartun/switch.py`.

---

### 7. Text-only and text+relations for SPaRTUN — `spartun/text_only.py`

Runs two baselines:
- **Text-only:** story → YN/FR answer
- **Text + relations:** story + extracted relations → YN/FR answer

Both use chain-of-thought few-shot prompts with real SPaRTUN examples.

**Entry:**
```bash
python main.py spartun-text
```

Flags `RUN_TEXT_ONLY` and `RUN_TEXT_WITH_RELATIONS` inside the file control which baselines run.

---

### 8. ResQ pipeline — `resq/pipeline.py`

Full GPT-5.1 pipeline:

1. Extract spatial relations (GPT)
2. Generate a spatial grid (GPT)
3. Repair / validate the grid (GPT)
4. Select relevant entities (GPT)
5. Prune grid to selected entities
6. Answer each question (GPT) across four modes: text-only, relations-only, text+relations, grid-only

**Entry:**
```bash
python main.py resq
```

Configure `INPUT_JSON`, `OUTPUT_JSON`, `SKIP_CONTEXTS`, `PROCESS_CONTEXTS` at the top of `resq/pipeline.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a local `.env` file from the template:

```bash
cp .env.example .env
```

PowerShell:

```powershell
Copy-Item .env.example .env
```

Then set your OpenAI key in `.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

Make sure Ollama is running locally if any experiment uses a local model:

```bash
ollama serve
ollama pull qwen3:14b     # or whichever model is configured
```

---

## Running experiments

Use `main.py` as the single entry point:

```bash
python main.py --help
python main.py stepgame-grid
python main.py stepgame-switch
python main.py stepgame-taxonomy
python main.py spartun-grid
python main.py spartun-text
python main.py spartun-switch
python main.py spartun-switch-analysis
python main.py resq-pipeline
```

Optional runner args:

```bash
python main.py <experiment> --model qwen3:14b
python main.py <experiment> --gpt-model gpt-5-mini
python main.py <experiment> --temperature 0.7
python main.py <experiment> --pull
```

Direct module runs also work from the repo root:

```bash
python -m stepgame.pipeline
python -m stepgame.switch
python -m spartun.pipeline
python -m spartun.text_only
python -m spartun.switch
python -m analysis.error_taxonomy
python -m analysis.spartun_switch
python -m resq.pipeline
```

---

## Notes

- All input/output paths are relative to the repo root.
- `.env` is ignored by git; use `.env.example` as the template.
- Edit per-run paths and thresholds inside the target module when needed.
