"""
StepGame Error Taxonomy Analysis
=================================
Two SEPARATE GPT-5.1 calls per text-only failure:

  CALL 1 — INPUT analysis  (GPT sees story + question ONLY, no model output):
    1. composite_spatial   – story has many composite/diagonal relations that could cause issues
    2. k_hop               – from dataset annotation, NO GPT call (just read meta)
    3. transitivity        – story requires transitive reasoning (A left of B -> B right of A)

  CALL 2 — OUTPUT analysis (GPT sees story + question + model reasoning):
    1. composite_failure   – model couldn't handle 2 directions at once, dropped one
    2. hallucination       – model invented a relation not in the story
    3. linguistic_difficulty – model failed to understand language (clock dirs, etc.)
    4. transitivity_failure – model failed at inverse reasoning (A left B -> B right A)

  NOTE: other_reasoning_error was removed — all 20 cases were multi-hop reasoning errors,
  subsumed by the existing categories.

Run in a Jupyter cell — edit the config lines below and execute.
"""

import json, os, time, re
from collections import Counter, defaultdict
from utils.llm_clients import call_gpt as _call_gpt_llm
from config import OPENAI_API_KEY


# ============ EDIT THESE ============
DATA_FILE   = "./stepgame_switch_gpt5_1_250.jsonl"   # <-- path to your JSONL
SWITCH_FILE = "./trust_eval_finagpt51 (2) (1).json"         # <-- trust eval file ("" to skip)
CACHE_FILE  = "_error_taxonomy_cachegpt51.json"                  # <-- cache (created automatically)
MODEL       = "gpt-5.1"                                     # <-- GPT model for classification
API_KEY     = OPENAI_API_KEY                               # <-- set via config.py or env var
NO_GPT      = False  # True = skip GPT calls (use cache only);  False = run GPT
FORCE_FRESH = False  # True = ignore cache and re-run all GPT calls
TRUST_THRESHOLD      = 0.7   # tau_t: keep text if trust >= this AND complexity < tau_c
COMPLEXITY_THRESHOLD = 0.55   # tau_c: keep text if complexity < this AND trust >= tau_t


COMPOSITE = {"upper-left", "upper-right", "lower-left", "lower-right"}
INVERSE = {
    "above": "below", "below": "above",
    "left": "right",  "right": "left",
    "upper-left": "lower-right", "lower-right": "upper-left",
    "upper-right": "lower-left", "lower-left": "upper-right",
}

# --- Load switch file (trust eval) ---
switch_lookup = {}  # dataset_id -> {trust, complexity, would_switch}
original_switch_rule = None
if SWITCH_FILE and os.path.exists(SWITCH_FILE):
    decoder = json.JSONDecoder()
    with open(SWITCH_FILE, encoding="utf-8") as _f:
        _data = _f.read()
    _idx = 0
    while _idx < len(_data):
        _dt = _data[_idx:].lstrip()
        if not _dt:
            break
        _idx = len(_data) - len(_dt)
        try:
            _obj, _end = decoder.raw_decode(_data, _idx)
            _idx += _end - _idx
        except json.JSONDecodeError:
            break
        if isinstance(_obj, dict) and "scores" in _obj and "meta" in _obj:
            _did = _obj["meta"]["dataset_id"]
            _sc = _obj["scores"]
            if original_switch_rule is None and _sc.get("switch_policy_rule"):
                original_switch_rule = _sc.get("switch_policy_rule")
            _trust = _sc.get("trustworthiness_score", 0)
            _complex = _sc.get("complexity_score", 0)
            # Re-evaluate would_switch at the user's thresholds
            _would_switch = not (_complex < COMPLEXITY_THRESHOLD and _trust >= TRUST_THRESHOLD)
            # Extract text-only correctness from trust eval
            _lm_correct_obj = _obj.get("lm_correct", {})
            if isinstance(_lm_correct_obj, dict):
                _text_correct = _lm_correct_obj.get("baseline_symbolic_text_only")
            else:
                _text_correct = _lm_correct_obj  # already a bool
            _grid_correct_te = _obj.get("grid_correct")
            switch_lookup[_did] = {
                "trust": _trust,
                "complexity": _complex,
                "would_switch": _would_switch,
                "original_would_switch": _sc.get("would_switch"),
                "text_correct": _text_correct,
                "grid_correct": _grid_correct_te,
            }
    print(f"[switch] Loaded {len(switch_lookup)} records from {SWITCH_FILE}")
    print(f"         Thresholds: trust >= {TRUST_THRESHOLD}, complexity < {COMPLEXITY_THRESHOLD}")
    print(f"         Rule: switch UNLESS (complexity < {COMPLEXITY_THRESHOLD}) AND (trust >= {TRUST_THRESHOLD})")
elif SWITCH_FILE:
    print(f"[switch] WARNING: {SWITCH_FILE} not found, assuming all text failures switch.")

# (OpenAI client is managed inside utils/llm_clients)


def call_gpt(prompt: str, retries: int = 3) -> str:
    if NO_GPT:
        return ""
    return _call_gpt_llm(prompt, model=MODEL, retries=retries)


def parse_json_safe(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return {}


# --- Data ---
records = []
with open(DATA_FILE, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
N = len(records)

# Patch lm_correct / grid_correct from trust eval (trust eval is source of truth)
_patched_lm = 0
_patched_gc = 0
for r in records:
    did = r["meta"]["dataset_id"]
    sw = switch_lookup.get(did)
    if sw:
        if sw.get("text_correct") is not None:
            if r.get("lm_correct") != sw["text_correct"]:
                _patched_lm += 1
            r["lm_correct"] = sw["text_correct"]
        if sw.get("grid_correct") is not None:
            if r.get("grid_correct") != sw["grid_correct"]:
                _patched_gc += 1
            r["grid_correct"] = sw["grid_correct"]
if _patched_lm or _patched_gc:
    print(f"[patch] From trust eval: patched lm_correct for {_patched_lm} records, grid_correct for {_patched_gc} records")
_n_still_none = sum(1 for r in records if r.get("lm_correct") is None)
if _n_still_none:
    print(f"[WARNING] {_n_still_none} records still have lm_correct=None (no trust eval data). They will be counted as text failures.")

lm_fail = [r for r in records if not r["lm_correct"]]
print(f"Loaded {N} records, {len(lm_fail)} text-only failures to classify.\n")


# --- Format helpers ---
def format_reasoning(slr):
    if not slr:
        return "(no sentence-level reasoning available)"
    lines = []
    for i, s in enumerate(slr):
        pred = s.get("prediction", {})
        stage2 = s.get("two_stage", {}).get("stage2", {})
        lines.append(
            f"  Sentence {i+1}: \"{s['sentence']}\"\n"
            f"    -> Extracted: {pred.get('head','?')} is {pred.get('relation','?')} of {pred.get('tail','?')}\n"
            f"    -> Stage2: {stage2.get('sentence','?')}"
        )
    return "\n".join(lines)


def format_relations(re_data):
    gold = re_data.get("gold_relations", [])
    pred = re_data.get("predicted_relations", [])
    lines = ["  Gold relations:"]
    for r in gold:
        lines.append(f"    {r['head']} is {r['relation']} of {r['tail']}")
    lines.append("  Predicted relations:")
    for r in pred:
        lines.append(f"    {r['head']} is {r['relation']} of {r['tail']}")
    return "\n".join(lines)


# ===============================================
# GPT PROMPTS
# ===============================================

PROMPT_INPUT = """You are analyzing the INPUT difficulty of a spatial reasoning problem.
You can ONLY see the story and question — you do NOT see any model output.

Story:
{story}

Question: {question}

Analyze this input for the PRESENCE of each difficulty factor below.
For each factor, answer true/false and give a one-sentence reason.

1. composite_spatial: Does the story contain composite/diagonal spatial relations (upper-left, lower-right, etc.) or does answering the question require reasoning about diagonal/composite directions? These are harder than simple cardinal directions.

2. transitivity: Does the story require the reasoner to apply transitive/inverse reasoning — i.e., converting "A is left of B" to "B is right of A" or chaining through intermediaries — to arrive at the answer?

Return ONLY valid JSON (no markdown):
{{
  "composite_spatial": {{"present": true/false, "reason": "..."}},
  "transitivity": {{"present": true/false, "reason": "..."}}
}}"""


PROMPT_OUTPUT = """You are analyzing WHY a spatial reasoning model gave a WRONG answer.
You see the story, question, AND the model's step-by-step reasoning.

Story:
{story}

Question: {question}
Gold answer: {gold}
Model's answer: {model_answer}

Model's sentence-level reasoning:
{reasoning}

Extracted relations (gold vs predicted):
{relations}

The model got the answer WRONG. For each of the following failure categories,
determine whether it contributed to the error. Multiple categories CAN be true
simultaneously — check each one independently.

1. composite_failure: Did the model fail because it couldn't handle composite/diagonal
   directions (upper-left, lower-right, etc.)? E.g., it dropped one component and
   predicted "above" when the answer was "upper-right", or couldn't keep track of
   two directional components at once.

2. hallucination: Did the model invent or fabricate a spatial relation that is NOT
   stated or logically implied by the story? Its reasoning includes a claim that
   has no basis in the story text.

3. linguistic_difficulty: Did the model fail to understand the LANGUAGE of the story?
   E.g., misinterpreting clock-face positions ("at 5 o'clock"), unusual phrasing
   ("presents right to"), vague spatial language ("over there"), or other language
   comprehension issues that led to wrong relation extraction.

4. transitivity_failure: Did the model fail to apply inverse/transitive reasoning?
   E.g., it needed to convert "A is left of B" -> "B is right of A" (or chain
   through intermediaries) and failed to do so, leading to the wrong answer.

5. other_reasoning_error: The model understood the language fine, didn't hallucinate,
   handled composites OK, did transitivity OK, but STILL got it wrong through some
   other reasoning mistake (e.g., miscounting hops, confusing entities, arithmetic
   error on positions). Use this as catch-all only if none of 1-4 fully explain it.

Return ONLY valid JSON (no markdown):
{{
  "composite_failure": {{"present": true/false, "reason": "..."}},
  "hallucination": {{"present": true/false, "reason": "..."}},
  "linguistic_difficulty": {{"present": true/false, "reason": "..."}},
  "transitivity_failure": {{"present": true/false, "reason": "..."}},
  "other_reasoning_error": {{"present": true/false, "reason": "..."}}
}}"""


# ===============================================
# RUN GPT CALLS (or load cache)
# ===============================================
cache = {}
if not FORCE_FRESH and os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[cache] Loaded {len(cache)} cached records from {CACHE_FILE}")
    print(f"        Pass --fresh to re-run all GPT calls with {MODEL}.\n")

if not NO_GPT and client is not None and (FORCE_FRESH or not cache):
    print(f"{'='*90}")
    print(f"  ERROR TAXONOMY CLASSIFICATION  (model={MODEL})")
    print(f"  {len(lm_fail)} text-only failures x 2 calls each = {len(lm_fail)*2} GPT calls")
    print(f"{'='*90}")

    for i, r in enumerate(lm_fail):
        did = r["meta"]["dataset_id"]
        khop = r["meta"]["k_hop"]
        gold = r["gold_relation"]

        # -- CALL 1: INPUT analysis (story + question only) --
        print(f"\n  [{i+1}/{len(lm_fail)}] id={did}  k={khop}  gold={gold}")
        print(f"  +- INPUT call (story+question only)...")

        inp_prompt = PROMPT_INPUT.format(
            story=r["story"],
            question=r["question"],
            gold=gold,
            relations=format_relations(r["relation_extraction"]),
        )
        inp_raw = call_gpt(inp_prompt)
        inp_parsed = parse_json_safe(inp_raw)

        # Print input results
        for cat in ["composite_spatial", "transitivity"]:
            info = inp_parsed.get(cat, {})
            flag = "✓" if info.get("present") else "✗"
            reason = info.get("reason", "")
            print(f"  |  {flag} {cat}: {reason}")

        # k_hop from annotation (no GPT needed)
        print(f"  |  ✓ k_hop={khop} (from dataset)")

        # -- CALL 2: OUTPUT analysis (story + question + reasoning) --
        print(f"  |- OUTPUT call (story+question+reasoning)...")

        # figure out model answer
        slr = r.get("sentence_level_reasoning", [])
        model_answer = "unknown"
        if slr:
            last_pred = slr[-1].get("prediction", {})
            model_answer = last_pred.get("relation", "unknown")

        out_prompt = PROMPT_OUTPUT.format(
            story=r["story"],
            question=r["question"],
            gold=gold,
            model_answer=model_answer,
            reasoning=format_reasoning(slr),
            relations=format_relations(r["relation_extraction"]),
        )
        out_raw = call_gpt(out_prompt)
        out_parsed = parse_json_safe(out_raw)

        # Print output results
        output_cats = ["composite_failure", "hallucination", "linguistic_difficulty",
                       "transitivity_failure", "other_reasoning_error"]
        triggered = []
        for cat in output_cats:
            info = out_parsed.get(cat, {})
            flag = "✓" if info.get("present") else "✗"
            reason = info.get("reason", "")
            print(f"  |  {flag} {cat}: {reason}")
            if info.get("present"):
                triggered.append(cat)

        print(f"  +- Output failures: {triggered if triggered else ['none_classified']}")

        # Store
        cache[did] = {
            "dataset_id": did,
            "k_hop": khop,
            "gold_relation": gold,
            "model_answer": model_answer,
            "input_analysis": {
                "composite_spatial": inp_parsed.get("composite_spatial", {}),
                "transitivity": inp_parsed.get("transitivity", {}),
                "k_hop": khop,
                "raw_gpt_response": inp_raw,
            },
            "output_analysis": {
                "composite_failure": out_parsed.get("composite_failure", {}),
                "hallucination": out_parsed.get("hallucination", {}),
                "linguistic_difficulty": out_parsed.get("linguistic_difficulty", {}),
                "transitivity_failure": out_parsed.get("transitivity_failure", {}),
                "other_reasoning_error": out_parsed.get("other_reasoning_error", {}),
                "raw_gpt_response": out_raw,
            },
        }
        time.sleep(0.3)

    # Save cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print(f"\n[GPT] Saved {len(cache)} records to {CACHE_FILE}")

elif NO_GPT:
    print("[GPT] Skipped (--no-gpt flag).")


# ===============================================
# AGGREGATE & TABLE
# ===============================================
if not cache:
    print("No classification data available.")
    print("  -> Set NO_GPT = False to run GPT classification, or check CACHE_FILE path.")
    class _StopExecution(Exception):
        def _render_traceback_(self): return []
    raise _StopExecution()

n_classified = len(cache)

# --- Input-side counts ---
input_cats = ["composite_spatial", "transitivity"]
input_counts = {c: 0 for c in input_cats}
# k-hop distribution
khop_counts = Counter()

for did, entry in cache.items():
    inp = entry.get("input_analysis", {})
    for cat in input_cats:
        if inp.get(cat, {}).get("present"):
            input_counts[cat] += 1
    khop_counts[entry.get("k_hop", 0)] += 1

# --- Output-side counts ---
output_cats = ["composite_failure", "hallucination", "linguistic_difficulty",
               "transitivity_failure"]
output_counts = {c: 0 for c in output_cats}

# Track multi-label combos
combo_counter = Counter()

for did, entry in cache.items():
    out = entry.get("output_analysis", {})
    triggered = []
    for cat in output_cats:
        if out.get(cat, {}).get("present"):
            output_counts[cat] += 1
            triggered.append(cat)

    if not triggered:
        triggered = ["none_classified"]
    combo_counter[tuple(sorted(triggered))] += 1


# --- Cross-tab: which input difficulties co-occur with which output failures ---
cross_tab = defaultdict(lambda: defaultdict(int))
for did, entry in cache.items():
    inp = entry.get("input_analysis", {})
    out = entry.get("output_analysis", {})
    inp_flags = []
    for c in input_cats:
        if inp.get(c, {}).get("present"):
            inp_flags.append(c)
    inp_flags.append(f"k_hop={entry.get('k_hop', '?')}")

    for ic in inp_flags:
        for oc in output_cats:
            if out.get(oc, {}).get("present"):
                cross_tab[ic][oc] += 1


# ===============================================
# PRINT TABLE
# ===============================================
def pct(n, d):
    return f"{n}/{d} ({n/d*100:.1f}%)" if d else "0/0"

SEP = "-" * 95

# ===============================================
# PIPELINE OVERVIEW  (Text -> Switch decision -> Grid)
# ===============================================
lm_correct_records = [r for r in records if r["lm_correct"]]
lm_fail_records    = [r for r in records if not r["lm_correct"]]

# Real policy evaluation: every record is either switched to grid or kept as text.
policy_switched_records = []
policy_kept_records = []
for r in records:
    did = r["meta"]["dataset_id"]
    sw = switch_lookup.get(did)
    would_switch = True if not sw else sw["would_switch"]
    if would_switch:
        policy_switched_records.append(r)
    else:
        policy_kept_records.append(r)

switched_grid_ok = [r for r in policy_switched_records if r.get("grid_correct") is True]
grid_failed = [r for r in policy_switched_records if r.get("grid_correct") is False]

switch_helped = [r for r in policy_switched_records if not r["lm_correct"] and r.get("grid_correct") is True]
switch_hurt = [r for r in policy_switched_records if r["lm_correct"] and r.get("grid_correct") is False]
switch_both_right = [r for r in policy_switched_records if r["lm_correct"] and r.get("grid_correct") is True]
switch_both_wrong = [r for r in policy_switched_records if not r["lm_correct"] and r.get("grid_correct") is False]

kept_text = [r for r in policy_kept_records if not r["lm_correct"]]
kept_text_correct = [r for r in policy_kept_records if r["lm_correct"]]
missed_rescues = [r for r in policy_kept_records if not r["lm_correct"] and r.get("grid_correct") is True]
kept_both_wrong = [r for r in policy_kept_records if not r["lm_correct"] and r.get("grid_correct") is False]
dodged_breaks = [r for r in policy_kept_records if r["lm_correct"] and r.get("grid_correct") is False]

actually_switched = [r for r in lm_fail_records if r["meta"]["dataset_id"] in {x["meta"]["dataset_id"] for x in policy_switched_records}]
grid_recovered = switch_helped

n_text_ok    = len(lm_correct_records)
n_text_fail  = len(lm_fail_records)
n_switched   = len(policy_switched_records)
n_kept_total = len(policy_kept_records)
n_kept_text  = len(kept_text)
n_grid_ok    = len(switched_grid_ok)
n_grid_fail  = len(grid_failed)
n_switch_helped = len(switch_helped)
n_switch_hurt = len(switch_hurt)
n_final_ok   = len(switched_grid_ok) + len(kept_text_correct)
n_final_err  = N - n_final_ok

has_switch = bool(switch_lookup)
_switched_ids = {r["meta"]["dataset_id"] for r in policy_switched_records}
_kept_ids     = {r["meta"]["dataset_id"] for r in policy_kept_records}
_kept_wrong_ids = {r["meta"]["dataset_id"] for r in kept_text}
_kept_right_ids = {r["meta"]["dataset_id"] for r in kept_text_correct}
_grok_ids     = {r["meta"]["dataset_id"] for r in switched_grid_ok}
_grfl_ids     = {r["meta"]["dataset_id"] for r in grid_failed}
_helped_ids   = {r["meta"]["dataset_id"] for r in switch_helped}
_switch_hurt_ids = {r["meta"]["dataset_id"] for r in switch_hurt}
_switch_both_wrong_ids = {r["meta"]["dataset_id"] for r in switch_both_wrong}
_missed_rescue_ids = {r["meta"]["dataset_id"] for r in missed_rescues}

switch_text_buckets = defaultdict(lambda: {"total": 0, "grid_ok": 0})
switch_reason_counts = Counter()
switch_reason_examples = defaultdict(list)
if has_switch:
    for r in records:
        did = r["meta"]["dataset_id"]
        sw = switch_lookup.get(did)
        would_switch = True if not sw else sw["would_switch"]
        text_correct = bool(r["lm_correct"])
        bucket = switch_text_buckets[(would_switch, text_correct)]
        bucket["total"] += 1
        if r.get("grid_correct") is True:
            bucket["grid_ok"] += 1
        if would_switch:
            trust = 0 if not sw else sw.get("trust", 0)
            complexity = 0 if not sw else sw.get("complexity", 0)
            trust_trigger = trust < TRUST_THRESHOLD
            complexity_trigger = complexity >= COMPLEXITY_THRESHOLD
            if trust_trigger and complexity_trigger:
                reason = "low_trust_and_high_complexity"
            elif trust_trigger:
                reason = "low_trust_only"
            elif complexity_trigger:
                reason = "high_complexity_only"
            else:
                reason = "no_switch_data_default"
            switch_reason_counts[reason] += 1
            if len(switch_reason_examples[reason]) < 3:
                switch_reason_examples[reason].append({
                    "dataset_id": did,
                    "k_hop": r["meta"].get("k_hop"),
                    "trust": trust,
                    "complexity": complexity,
                    "text_correct": bool(r.get("lm_correct")),
                    "grid_correct": r.get("grid_correct"),
                })

original_policy_changes = []
original_policy_counts = None
if has_switch:
    _orig_keep = 0
    _orig_switch = 0
    for r in records:
        did = r["meta"]["dataset_id"]
        sw = switch_lookup.get(did, {})
        original_ws = sw.get("original_would_switch")
        if original_ws is None:
            continue
        if original_ws:
            _orig_switch += 1
        else:
            _orig_keep += 1
        current_ws = did in _switched_ids
        if current_ws != original_ws:
            original_policy_changes.append({
                "dataset_id": did,
                "k_hop": r["meta"].get("k_hop"),
                "trust": sw.get("trust"),
                "complexity": sw.get("complexity"),
                "text_correct": r.get("lm_correct"),
                "grid_correct": r.get("grid_correct"),
                "original_decision": "switch" if original_ws else "keep_text",
                "current_decision": "switch" if current_ws else "keep_text",
            })
    if _orig_keep or _orig_switch or original_policy_changes:
        original_policy_counts = {
            "kept_text": _orig_keep,
            "switched": _orig_switch,
        }

print(f"\n{'='*95}")
print(f"  STEPGAME END-TO-END ERROR ANALYSIS  (model={MODEL}, N={N})")
print(f"{'='*95}")

print(f"\n{SEP}")
print("  PIPELINE OVERVIEW")
if has_switch:
    print(f"  Thresholds: trust >= {TRUST_THRESHOLD}, complexity < {COMPLEXITY_THRESHOLD}")
    print(f"  Switch file coverage: {len(switch_lookup)} records have trust scores")
else:
    print("  (no switch file loaded -- assuming all text failures switch to grid)")
print(SEP)
print(f"  Total records:               {N}")
print(f"  +- Text correct:             {n_text_ok:3d}  ({n_text_ok/N*100:.1f}%)")
print(f"  +- Text FAIL:                {n_text_fail:3d}  ({n_text_fail/N*100:.1f}%)")
if has_switch:
    print(f"  Policy switched to grid:     {n_switched:3d}  ({n_switched/N*100:.1f}% of all records)")
    print(f"  Policy kept text:            {n_kept_total:3d}  ({n_kept_total/N*100:.1f}% of all records)")
    print(f"       +- Kept & text correct: {len(kept_text_correct):3d}")
    print(f"       +- Kept & text WRONG:   {n_kept_text:3d}  ({n_kept_text/n_text_fail*100:.1f}% of failures)")
else:
    print(f"  Policy switched to grid:     {n_switched:3d}  (all records, no threshold applied)")
if n_switched:
    print(f"       +- Switched & grid OK:  {n_grid_ok:3d}  ({n_grid_ok/n_switched*100:.1f}% of switches)")
    print(f"       +- Switched & grid BAD: {n_grid_fail:3d}  ({n_grid_fail/n_switched*100:.1f}% of switches)")
print(f"  ------------------------------------")
print(f"  Final accuracy:              {n_final_ok}/{N} = {n_final_ok/N*100:.1f}%")
print(f"  Final errors:                {n_final_err}/{N} = {n_final_err/N*100:.1f}%")
if has_switch:
    print(f"    missed rescues (stay):     {len(missed_rescues)}")
    print(f"    bad switches:              {n_switch_hurt}")
    print(f"    switched & both wrong:     {len(switch_both_wrong)}")
    print(f"    switch rescued:            {n_switch_helped}")

# Threshold sensitivity (if switch data loaded)
if has_switch:
    print(f"\n  Threshold sensitivity (tau_t, tau_c -> final accuracy):")
    print(f"  {'tau_t':>6s}  {'tau_c':>6s}  {'switched':>8s}  {'stay':>6s}  {'missed':>7s}  {'bad_sw':>7s}  {'final_acc':>10s}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*10}")
    for tt in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        for tc in [0.3, 0.4, 0.5, 0.6]:
            _sw, _stay = 0, 0
            _missed, _bad_sw = 0, 0
            _final_ok = 0
            for r in records:
                did = r["meta"]["dataset_id"]
                si = switch_lookup.get(did)
                if si and not (si["complexity"] < tc and si["trust"] >= tt):
                    ws = True
                elif si:
                    ws = False
                else:
                    ws = True
                if ws:
                    _sw += 1
                    if r.get("grid_correct") is True:
                        _final_ok += 1
                    elif r["lm_correct"]:
                        _bad_sw += 1
                else:
                    _stay += 1
                    if r["lm_correct"]:
                        _final_ok += 1
                    else:
                        _missed += 1
            _facc = _final_ok / N * 100
            marker = " <--" if tt == TRUST_THRESHOLD and tc == COMPLEXITY_THRESHOLD else ""
            print(f"  {tt:6.2f}  {tc:6.2f}  {_sw:8d}  {_stay:6d}  {_missed:7d}  {_bad_sw:7d}  {_facc:9.1f}%{marker}")

if has_switch:
    print(f"\n{SEP}")
    print("  SWITCH x TEXT-CORRECT x GRID-ACCURACY")
    print("  Same contingency view as analyze_switch.py, using the current thresholds above.")
    print(SEP)
    print(f"  {'would_switch':<14s} {'text_correct':<14s} {'#cases':>7s}   {'grid_true':>15s}   {'grid_acc':>9s}")
    print(f"  {'-'*14} {'-'*14} {'-'*7}   {'-'*15}   {'-'*9}")
    for ws in [True, False]:
        for tc in [True, False]:
            bucket = switch_text_buckets[(ws, tc)]
            total = bucket["total"]
            grid_ok = bucket["grid_ok"]
            print(
                f"  {str(ws):<14s} {str(tc):<14s} {total:7d}   "
                f"{grid_ok:5d} / {total:<5d}   {pct(grid_ok, total):>9s}"
            )
    _tf_st_gt = switch_text_buckets[(True, False)]
    print(f"\n  text_only=False, switch=True, grid=True: {_tf_st_gt['grid_ok']} / {_tf_st_gt['total']}")

if has_switch:
    print(f"\n{SEP}")
    print("  REASON FOR SWITCHING")
    print("  Why the policy switched to grid under the current thresholds.")
    print(SEP)
    print(f"  {'Reason':<32s} {'#switches':>10s}  {'% of switches':>13s}")
    print(f"  {'-'*32} {'-'*10}  {'-'*13}")
    for reason in ["low_trust_only", "high_complexity_only", "low_trust_and_high_complexity", "no_switch_data_default"]:
        count = switch_reason_counts.get(reason, 0)
        if count or reason == "no_switch_data_default":
            print(f"  {reason:<32s} {count:10d}  {pct(count, n_switched):>13s}")

    for reason in ["low_trust_only", "high_complexity_only", "low_trust_and_high_complexity", "no_switch_data_default"]:
        examples = switch_reason_examples.get(reason, [])
        if not examples:
            continue
        print(f"\n  {reason} examples:")
        for item in examples:
            grid_str = "?" if item["grid_correct"] is None else str(item["grid_correct"])
            print(
                f"    {item['dataset_id']}  k={item['k_hop']}  trust={item['trust']:.3f}  "
                f"complexity={item['complexity']:.3f}  text_ok={item['text_correct']}  grid_ok={grid_str}"
            )

if has_switch and original_policy_counts is not None:
    print(f"\n  Threshold audit vs trust file policy:")
    if original_switch_rule:
        print(f"    trust file rule: keep text if {original_switch_rule.get('no_switch_if', '?')}")
    print(f"    trust file kept_text: {original_policy_counts['kept_text']}   switched: {original_policy_counts['switched']}")
    print(f"    current     kept_text: {n_kept_text}   switched: {n_switched}")
    if original_policy_changes:
        print(f"    records whose switch decision changed: {len(original_policy_changes)}")
        print(f"\n  {'id':<12s}  {'k':>2s}  {'trust':>6s}  {'cmplx':>6s}  {'text_ok':<7s}  {'orig':<10s}  {'current':<10s}  {'grid_ok':<7s}")
        print(f"  {'-'*12}  {'-'*2}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*7}")
        for item in sorted(original_policy_changes, key=lambda x: (x['k_hop'], x['dataset_id'])):
            _grid_ok = item['grid_correct']
            _grid_str = '?' if _grid_ok is None else str(_grid_ok)
            _text_ok = '?' if item['text_correct'] is None else str(bool(item['text_correct']))
            print(
                f"  {item['dataset_id']:<12s}  {item['k_hop']:2d}  {item['trust']:6.3f}  {item['complexity']:6.3f}  {_text_ok:<7s}  "
                f"{item['original_decision']:<10s}  {item['current_decision']:<10s}  {_grid_str:<7s}"
            )
    else:
        print("    records whose switch decision changed: 0")

# k-hop pipeline breakdown
print(f"\n  Per k-hop:")
if has_switch:
    print(f"  {'k':>3s}  {'total':>5s}  {'txt_ok':>6s}  {'txt_fail':>8s}  {'stay':>5s}  {'switch':>6s}  {'missed':>7s}  {'bad_sw':>7s}  {'final':>9s}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*9}")
else:
    print(f"  {'k':>3s}  {'total':>5s}  {'txt_ok':>6s}  {'txt_fail':>8s}  {'grid_ok':>7s}  {'grid_fail':>9s}  {'final_acc':>10s}")
    print(f"  {'-'*3}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*10}")
for k in range(1, 11):
    rk = [r for r in records if r["meta"]["k_hop"] == k]
    nk = len(rk)
    if nk == 0:
        continue
    tok = sum(1 for r in rk if r["lm_correct"])
    tf  = nk - tok
    _ksw = sum(1 for r in rk if r["meta"]["dataset_id"] in _switched_ids)
    _kkt = sum(1 for r in rk if r["meta"]["dataset_id"] in _kept_ids)
    _kmiss = sum(1 for r in rk if r["meta"]["dataset_id"] in _missed_rescue_ids)
    _kbad = sum(1 for r in rk if r["meta"]["dataset_id"] in _switch_hurt_ids)
    _kfinal_ok = sum(
        1 for r in rk
        if (r["meta"]["dataset_id"] in _switched_ids and r.get("grid_correct") is True)
        or (r["meta"]["dataset_id"] in _kept_ids and r["lm_correct"])
    )
    facc = _kfinal_ok / nk * 100
    if has_switch:
        print(f"  {k:3d}  {nk:5d}  {tok:6d}  {tf:8d}  {_kkt:5d}  {_ksw:6d}  {_kmiss:7d}  {_kbad:7d}  {facc:8.1f}%")
    else:
        gok = sum(1 for r in rk if r.get("grid_correct") is True)
        gf  = sum(1 for r in rk if r.get("grid_correct") is False)
        print(f"  {k:3d}  {nk:5d}  {tok:6d}  {tf:8d}  {gok:7d}  {gf:9d}  {facc:9.1f}%")

# --- KEPT TEXT (NO SWITCH) DETAILS ---
if has_switch and n_kept_text > 0:
    print(f"\n{SEP}")
    print(f"  KEPT TEXT (NO SWITCH) -- {n_kept_text} records where text answer was KEPT despite being WRONG")
    print(f"  These records had high trust + low complexity, so the pipeline trusted the (wrong) text answer.")
    print(SEP)
    # Sort by k-hop then dataset_id
    _kept_sorted = sorted(kept_text, key=lambda r: (r["meta"]["k_hop"], r["meta"]["dataset_id"]))
    print(f"  {'id':<12s}  {'k':>2s}  {'gold':<12s}  {'lm_pred':<12s}  {'trust':>6s}  {'cmplx':>6s}  {'text_error_types'}")
    print(f"  {'-'*12}  {'-'*2}  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*30}")
    for r in _kept_sorted:
        did = r["meta"]["dataset_id"]
        k = r["meta"]["k_hop"]
        gold = r.get("gold") or r.get("gold_relation", "?")
        pred = r.get("lm_answer", r.get("lm_pred", "?"))
        si = switch_lookup.get(did, {})
        tr = si.get("trust", 0)
        cx = si.get("complexity", 0)
        # Get error types from cache
        _entry = cache.get(did, {})
        _outtypes = [c for c in output_cats if _entry.get("output_analysis", {}).get(c, {}).get("present")]
        _intypes = [c for c in input_cats if _entry.get("input_analysis", {}).get(c, {}).get("present")]
        _types_str = ", ".join(_outtypes) if _outtypes else "(not classified)"
        print(f"  {did:<12s}  {k:2d}  {gold:<12s}  {pred:<12s}  {tr:6.3f}  {cx:6.3f}  {_types_str}")

    # Summary by k-hop
    print(f"\n  Kept-text by k-hop:")
    for k in range(1, 11):
        _kk = [r for r in kept_text if r["meta"]["k_hop"] == k]
        if _kk:
            print(f"    k={k:2d}: {len(_kk):3d} kept (all wrong)")
    # Summary by error type
    print(f"\n  Kept-text by error type:")
    for cat in output_cats:
        _nk = sum(1 for r in kept_text
                   if cache.get(r["meta"]["dataset_id"], {}).get("output_analysis", {}).get(cat, {}).get("present"))
        if _nk:
            print(f"    {cat:<30s}  {_nk:3d}  ({_nk/n_kept_text*100:.1f}%)")
    # Avg trust/complexity
    _avg_t = sum(switch_lookup.get(r["meta"]["dataset_id"], {}).get("trust", 0) for r in kept_text) / n_kept_text
    _avg_c = sum(switch_lookup.get(r["meta"]["dataset_id"], {}).get("complexity", 0) for r in kept_text) / n_kept_text
    print(f"\n  Avg trust (kept-text):      {_avg_t:.3f}  (threshold: >= {TRUST_THRESHOLD})")
    print(f"  Avg complexity (kept-text):  {_avg_c:.3f}  (threshold: < {COMPLEXITY_THRESHOLD})")
    print(f"  --> All {n_kept_text} are WRONG answers that the pipeline incorrectly trusted.")
elif has_switch and n_kept_text == 0:
    print(f"\n{SEP}")
    print(f"  KEPT TEXT (NO SWITCH) -- 0 records")
    print(f"  At thresholds trust >= {TRUST_THRESHOLD}, complexity < {COMPLEXITY_THRESHOLD}, ALL text failures switch to grid.")
    print(SEP)

if has_switch and n_switch_hurt > 0:
    print(f"\n{SEP}")
    print(f"  BAD SWITCHES -- {n_switch_hurt} records where text was RIGHT but policy switched and grid was WRONG")
    print(f"  These are the realistic policy losses that the old optimistic accounting hid.")
    print(SEP)
    _hurt_sorted = sorted(switch_hurt, key=lambda r: (r["meta"]["k_hop"], r["meta"]["dataset_id"]))
    print(f"  {'id':<12s}  {'k':>2s}  {'gold':<12s}  {'trust':>6s}  {'cmplx':>6s}  {'grid_pred':<12s}")
    print(f"  {'-'*12}  {'-'*2}  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*12}")
    for r in _hurt_sorted:
        did = r["meta"]["dataset_id"]
        si = switch_lookup.get(did, {})
        gold = r.get("gold") or r.get("gold_relation", "?")
        print(
            f"  {did:<12s}  {r['meta']['k_hop']:2d}  {gold:<12s}  {si.get('trust', 0):6.3f}  "
            f"{si.get('complexity', 0):6.3f}  {str(r.get('grid_predicted', '?')):<12s}"
        )

# --- INPUT DIFFICULTY TABLE ---
print(f"\n\n{'='*95}")
print(f"  SECTION A: TEXT-ONLY ERROR TAXONOMY  ({n_classified} failures classified by GPT-5.1)")
print(f"{'='*95}")
print(f"\n{SEP}")
print("  INPUT-SIDE DIFFICULTY  (GPT sees story + question ONLY)")
print(SEP)
print(f"  {'Category':<25s}  {'Count':<12s}  {'% of failures'}")
print(f"  {'-'*25}  {'-'*12}  {'-'*15}")
for cat in input_cats:
    n = input_counts[cat]
    print(f"  {cat:<25s}  {n:<12d}  {n/n_classified*100:.1f}%")
print(f"  {'k_hop > 1':<25s}  {sum(v for k,v in khop_counts.items() if k>1):<12d}  "
      f"{sum(v for k,v in khop_counts.items() if k>1)/n_classified*100:.1f}%")
print(f"\n  k-hop distribution:")
for k in sorted(khop_counts.keys()):
    print(f"    k={k:2d}: {khop_counts[k]:3d} failures")


# --- OUTPUT FAILURE TABLE ---
print(f"\n{SEP}")
print("  OUTPUT-SIDE FAILURES  (GPT sees story + question + model reasoning)")
print("  NOTE: Multiple categories can be true per failure (multi-label)")
print(SEP)
print(f"  {'Category':<30s}  {'Count':<12s}  {'% of failures'}")
print(f"  {'-'*30}  {'-'*12}  {'-'*15}")
for cat in output_cats:
    n = output_counts[cat]
    print(f"  {cat:<30s}  {n:<12d}  {n/n_classified*100:.1f}%")


# --- FAILURE COMBO TABLE ---
print(f"\n{SEP}")
print("  OUTPUT FAILURE COMBINATIONS  (which categories co-occur)")
print(SEP)
for combo, cnt in sorted(combo_counter.items(), key=lambda x: -x[1]):
    label = " + ".join(combo)
    print(f"    {cnt:3d} ({cnt/n_classified*100:.1f}%)  {label}")


# --- INPUTxOUTPUT CROSS-TAB ---
print(f"\n{SEP}")
print("  INPUT x OUTPUT CROSS-TAB  (how input difficulty relates to output failure)")
print(SEP)
# Header
_io_label = "Input \\ Output"
hdr = f"  {_io_label:<25s}"
for oc in output_cats:
    hdr += f"  {oc[:12]:>12s}"
print(hdr)
print(f"  {'-'*25}" + "  ".join(["-"*12]*len(output_cats)))
for ic in input_cats:
    row = f"  {ic:<25s}"
    for oc in output_cats:
        v = cross_tab[ic][oc]
        row += f"  {v:12d}"
    print(row)
# k-hop rows
for k in sorted(khop_counts.keys()):
    ic = f"k_hop={k}"
    row = f"  {ic:<25s}"
    for oc in output_cats:
        v = cross_tab[ic][oc]
        row += f"  {v:12d}"
    print(row)


# --- BY K-HOP BREAKDOWN ---
print(f"\n{SEP}")
print("  OUTPUT FAILURES BY K-HOP")
print(SEP)
khop_output = defaultdict(lambda: {c: 0 for c in output_cats})
for did, entry in cache.items():
    k = entry.get("k_hop", 0)
    out = entry.get("output_analysis", {})
    for cat in output_cats:
        if out.get(cat, {}).get("present"):
            khop_output[k][cat] += 1

hdr = f"  {'k':>3s}  {'n':>4s}"
for oc in output_cats:
    hdr += f"  {oc[:14]:>14s}"
print(hdr)
print(f"  {'-'*3}  {'-'*4}" + "  ".join(["-"*14]*len(output_cats)))
for k in sorted(khop_output.keys()):
    nk = khop_counts[k]
    row = f"  {k:3d}  {nk:4d}"
    for oc in output_cats:
        v = khop_output[k][oc]
        row += f"  {v:14d}"
    print(row)


# ===============================================
# SECTION B: GRID FAILURE ANALYSIS (21 records)
# ===============================================
print(f"\n\n{'='*95}")
print(f"  SECTION B: GRID FAILURE ANALYSIS  ({n_grid_fail} switched cases where grid was WRONG)")
print(f"{'='*95}")

# --- Classify each grid failure ---
# Category 1: RE failure -> wrong grid
# Category 2: RE correct, grid construction error (grid doesn't match gold)
# Category 3: RE correct, grid correct (normalized match), but answer still wrong (grid reasoning error)
grid_cat_re_fail = []
grid_cat_construction = []
grid_cat_reasoning = []

for r in grid_failed:
    did = r["meta"]["dataset_id"]
    k = r["meta"]["k_hop"]
    gold = r["gold_relation"]
    grid_pred = r.get("grid_predicted", "?")

    re_data = r.get("relation_extraction", {})
    lenient = re_data.get("lenient_all_match", True)
    mismatches = re_data.get("mismatches", [])
    extras = re_data.get("extra_predictions", [])

    grids = r.get("grids", {})
    match = grids.get("match", {})
    grid_norm_ok = match.get("normalized_equal", False) if isinstance(match, dict) else False

    re_ok = lenient and len(mismatches) == 0

    entry = {
        "dataset_id": did, "k_hop": k, "gold": gold, "grid_pred": grid_pred,
        "text_correct": bool(r.get("lm_correct")),
        "re_ok": re_ok, "grid_norm_ok": grid_norm_ok,
        "n_mismatches": len(mismatches), "n_extras": len(extras),
    }

    if not re_ok and not grid_norm_ok:
        entry["category"] = "re_failure"
        grid_cat_re_fail.append(entry)
    elif not re_ok and grid_norm_ok:
        # RE had issues but grid still matched -> grid reasoning error
        entry["category"] = "grid_reasoning"
        grid_cat_reasoning.append(entry)
    elif re_ok and not grid_norm_ok:
        entry["category"] = "grid_construction"
        grid_cat_construction.append(entry)
    else:
        # RE ok, grid ok -> grid reasoning error
        entry["category"] = "grid_reasoning"
        grid_cat_reasoning.append(entry)

print(f"\n{SEP}")
print("  GRID FAILURE CATEGORIES")
print(SEP)
grid_cats = [
    ("RE failure -> wrong grid",       grid_cat_re_fail,      "Relation extraction missed/wrong relation -> grid built incorrectly"),
    ("Grid construction error",        grid_cat_construction, "RE correct but grid layout doesn't match gold grid"),
    ("Grid reasoning error",           grid_cat_reasoning,    "Grid correct (normalized match) but wrong answer read from it"),
]
print(f"  {'Category':<30s}  {'Count':<8s}  {'% of grid fails':<16s}  Description")
print(f"  {'-'*30}  {'-'*8}  {'-'*16}  {'-'*50}")
for label, cases, desc in grid_cats:
    nc = len(cases)
    print(f"  {label:<30s}  {nc:<8d}  {nc/n_grid_fail*100 if n_grid_fail else 0:>6.1f}%{'':9s}  {desc}")

# --- Grid failures by k-hop ---
print(f"\n{SEP}")
print("  GRID FAILURES BY K-HOP")
print(SEP)
print(f"  {'k':>3s}  {'switches':>8s}  {'grid_ok':>7s}  {'grid_fail':>9s}  {'bad_sw':>7s}  {'re_fail':>7s}  {'constr':>6s}  {'reason':>6s}")
print(f"  {'-'*3}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*6}")
for k in range(1, 11):
    switched_k = [r for r in policy_switched_records if r["meta"]["k_hop"] == k]
    if not switched_k:
        continue
    ns = len(switched_k)
    gok = sum(1 for r in switched_k if r.get("grid_correct") is True)
    gfl = sum(1 for r in switched_k if r.get("grid_correct") is False)
    bad_sw = sum(1 for r in switched_k if r["lm_correct"] and r.get("grid_correct") is False)
    re_f = sum(1 for e in grid_cat_re_fail if e["k_hop"] == k)
    co_f = sum(1 for e in grid_cat_construction if e["k_hop"] == k)
    gr_f = sum(1 for e in grid_cat_reasoning if e["k_hop"] == k)
    print(f"  {k:3d}  {ns:8d}  {gok:7d}  {gfl:9d}  {bad_sw:7d}  {re_f:7d}  {co_f:6d}  {gr_f:6d}")

# --- Grid failures detail (compact) ---
print(f"\n{SEP}")
print("  GRID FAILURE DETAILS")
print(SEP)
all_grid_entries = grid_cat_re_fail + grid_cat_construction + grid_cat_reasoning
all_grid_entries.sort(key=lambda e: (e["k_hop"], e["dataset_id"]))
print(f"  {'id':<12s}  {'k':>2s}  {'text_ok':<7s}  {'gold':<12s}  {'grid_pred':<12s}  {'category':<25s}  {'RE miss':>7s}  {'extras':>6s}")
print(f"  {'-'*12}  {'-'*2}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*25}  {'-'*7}  {'-'*6}")
for e in all_grid_entries:
    print(f"  {e['dataset_id']:<12s}  {e['k_hop']:2d}  {str(e['text_correct']):<7s}  {e['gold']:<12s}  {e['grid_pred']:<12s}  {e['category']:<25s}  {e['n_mismatches']:7d}  {e['n_extras']:6d}")

# --- Cross-reference: text taxonomy x grid outcome ---
print(f"\n{SEP}")
print("  TEXT ERROR TYPE x GRID OUTCOME  (did the grid recover from each text error type?)")
print(SEP)
print(f"  {'Text error type':<25s}  {'total':>5s}  {'kept':>5s}  {'grid_ok':>7s}  {'grid_fail':>9s}  {'recovery%':>10s}")
print(f"  {'-'*25}  {'-'*5}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*10}")

grid_fail_ids = _switch_both_wrong_ids
grid_ok_ids   = _helped_ids
kept_text_ids = _kept_wrong_ids

for cat in output_cats:
    # Find all text failures with this category
    cat_ids = [did for did, entry in cache.items()
               if entry.get("output_analysis", {}).get(cat, {}).get("present")]
    n_cat = len(cat_ids)
    n_kpt = sum(1 for d in cat_ids if d in kept_text_ids)
    n_gok = sum(1 for d in cat_ids if d in grid_ok_ids)
    n_gfl = sum(1 for d in cat_ids if d in grid_fail_ids)
    n_switched_cat = n_gok + n_gfl
    rec = n_gok / n_switched_cat * 100 if n_switched_cat else 0
    print(f"  {cat:<25s}  {n_cat:5d}  {n_kpt:5d}  {n_gok:7d}  {n_gfl:9d}  {rec:9.1f}%")

# Also show none_classified
none_ids = [did for did, entry in cache.items()
            if not any(entry.get("output_analysis", {}).get(c, {}).get("present") for c in output_cats)]
if none_ids:
    n_nc = len(none_ids)
    n_kpt_nc = sum(1 for d in none_ids if d in kept_text_ids)
    n_gok_nc = sum(1 for d in none_ids if d in grid_ok_ids)
    n_gfl_nc = sum(1 for d in none_ids if d in grid_fail_ids)
    n_sw_nc = n_gok_nc + n_gfl_nc
    rec_nc = n_gok_nc / n_sw_nc * 100 if n_sw_nc else 0
    print(f"  {'(none_classified)':<25s}  {n_nc:5d}  {n_kpt_nc:5d}  {n_gok_nc:7d}  {n_gfl_nc:9d}  {rec_nc:9.1f}%")

# ===============================================
# FINAL SUMMARY
# ===============================================
print(f"\n\n{'='*95}")
print(f"  FINAL SUMMARY")
print(f"{'='*95}")
_grid_rec_str = f"{n_grid_ok}/{n_switched} = {n_grid_ok/n_switched*100:.1f}%" if n_switched else "N/A (no switches)"
print(f"""
    Pipeline:  Text-only LM -> threshold policy -> (switch ? grid : keep text)
  {'Thresholds:  trust >= ' + str(TRUST_THRESHOLD) + ', complexity < ' + str(COMPLEXITY_THRESHOLD) if has_switch else '(no switch file -- all failures switched)'}

  Text accuracy:          {n_text_ok}/{N} = {n_text_ok/N*100:.1f}%
  Text failures:          {n_text_fail}/{N} = {n_text_fail/N*100:.1f}%
        kept text total:       {n_kept_total}   switched to grid: {n_switched}
        kept text wrong:       {n_kept_text}   bad switches:     {n_switch_hurt}
    Grid success rate:      {_grid_rec_str}  (of all switches)
  Final accuracy:         {n_final_ok}/{N} = {n_final_ok/N*100:.1f}%
    Remaining errors:       {n_final_err}/{N} = {n_final_err/N*100:.1f}%  (missed_rescues={len(missed_rescues)} + bad_switches={n_switch_hurt} + switched_both_wrong={len(switch_both_wrong)})

  Text failure breakdown (GPT-5.1 multi-label, {n_classified} failures):""")
for cat in output_cats:
    nc = output_counts[cat]
    print(f"    {cat:<30s}  {nc:3d}  ({nc/n_classified*100:.1f}%)")
if n_grid_fail:
    _re_pct = f"{len(grid_cat_re_fail)/n_grid_fail*100:.1f}%"
    _co_pct = f"{len(grid_cat_construction)/n_grid_fail*100:.1f}%"
    _gr_pct = f"{len(grid_cat_reasoning)/n_grid_fail*100:.1f}%"
else:
    _re_pct = _co_pct = _gr_pct = "N/A"
print(f"""
    Switched-grid failure breakdown ({n_grid_fail} failures):
    RE failure -> wrong grid:      {len(grid_cat_re_fail):3d}  ({_re_pct})
    Grid construction error:      {len(grid_cat_construction):3d}  ({_co_pct})
    Grid reasoning error:         {len(grid_cat_reasoning):3d}  ({_gr_pct})
""")
print("=" * 95)
output_json = {
    "meta": {
        "model": MODEL,
        "total_records": N,
        "failures_classified": n_classified,
    },
    "pipeline_overview": {
        "text_correct": n_text_ok,
        "text_fail": n_text_fail,
                "policy_kept_total": n_kept_total,
        "kept_text_no_switch": n_kept_text,
                "actually_switched": n_switched,
                "switched_grid_correct": n_grid_ok,
                "switched_grid_failed": n_grid_fail,
                "switch_helped": n_switch_helped,
                "switch_hurt": n_switch_hurt,
                "missed_rescues": len(missed_rescues),
                "switched_both_wrong": len(switch_both_wrong),
        "final_correct": n_final_ok,
        "final_errors": n_final_err,
        "final_accuracy_pct": round(n_final_ok / N * 100, 1),
        "thresholds": {"trust": TRUST_THRESHOLD, "complexity": COMPLEXITY_THRESHOLD} if has_switch else None,
        "switch_coverage": len(switch_lookup),
        "trust_file_policy": original_switch_rule,
    },
    "input_difficulty": {cat: {"count": input_counts[cat], "pct": round(input_counts[cat]/n_classified*100, 1)} for cat in input_cats},
    "output_failures": {cat: {"count": output_counts[cat], "pct": round(output_counts[cat]/n_classified*100, 1)} for cat in output_cats},
    "failure_combos": {" + ".join(k): v for k, v in sorted(combo_counter.items(), key=lambda x: -x[1])},
    "grid_failure_categories": {
        "re_failure": len(grid_cat_re_fail),
        "grid_construction": len(grid_cat_construction),
        "grid_reasoning": len(grid_cat_reasoning),
    },
    "grid_failure_details": all_grid_entries,
    "kept_text_details": [
        {
            "dataset_id": r["meta"]["dataset_id"],
            "k_hop": r["meta"]["k_hop"],
            "gold": r.get("gold") or r.get("gold_relation", "?"),
            "lm_pred": r.get("lm_answer", r.get("lm_pred", "?")),
            "trust": switch_lookup.get(r["meta"]["dataset_id"], {}).get("trust"),
            "complexity": switch_lookup.get(r["meta"]["dataset_id"], {}).get("complexity"),
            "error_types": [c for c in output_cats
                            if cache.get(r["meta"]["dataset_id"], {}).get("output_analysis", {}).get(c, {}).get("present")],
        }
        for r in sorted(kept_text, key=lambda r: (r["meta"]["k_hop"], r["meta"]["dataset_id"]))
    ],
    "bad_switch_details": [
        {
            "dataset_id": r["meta"]["dataset_id"],
            "k_hop": r["meta"]["k_hop"],
            "gold": r.get("gold") or r.get("gold_relation", "?"),
            "trust": switch_lookup.get(r["meta"]["dataset_id"], {}).get("trust"),
            "complexity": switch_lookup.get(r["meta"]["dataset_id"], {}).get("complexity"),
            "grid_pred": r.get("grid_predicted", "?"),
        }
        for r in sorted(switch_hurt, key=lambda r: (r["meta"]["k_hop"], r["meta"]["dataset_id"]))
    ],
    "threshold_audit": {
        "trust_file_policy_counts": original_policy_counts,
        "changed_records": original_policy_changes,
    },
    "switch_reasons": {
        "counts": dict(switch_reason_counts),
        "examples": dict(switch_reason_examples),
    },
    "per_record": cache,
}
with open("_error_taxonomy_resultspt51.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, indent=2)
print(f"\n[SAVED] Full results -> _error_taxonomy_resultspt51.json")
print("=" * 95)

