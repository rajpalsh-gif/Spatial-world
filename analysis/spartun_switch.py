#!/usr/bin/env python3
"""
analysis/spartun_switch.py  -  Deep analysis of trust-based switching pipeline
              results for SPaRTUN spatial reasoning dataset
==============================================================================

Jupyter-friendly: paste the entire file into a single cell and run.
Edit the CONFIGURATION block below before running.

Produces:
  Table 1  - would_switch x text-only-correct x grid accuracy (full + pruned)
  Table 2  - WHY switch is True/False  (trust vs complexity breakdown)
  Table 3  - Close-call / borderline decisions (within eps of thresholds)
  Table 4  - Very untrustworthy cases  (trust < 0.2)
  Table 5  - Switch-helped vs Switch-hurt analysis  (for each grid variant)
  Table 6  - Complexity C4 component breakdown (SB, EL, HL, CF)
             + accuracy by q_type (YN vs FR)
  Table 7  - Q1 vs Q2 contribution to trust score
  Table 8  - Oracle analysis (was the switch decision optimal?)
  Table 9  - Accuracy by ground-truth relation type  (FR) / by q_type (YN/FR)
  Table 10 - Optimal threshold grid search
  Table 11 - Error taxonomy: input difficulty (entity complexity, multi-hop chain)
             + output failure (hallucination, linguistic, multi-hop error) — two GPT calls
  Table 12 - Q1/Q2/Q3 individual sufficiency & marginal value
  Table 13 - Failure types mapped to trust & complexity scores
  Table 14 - C4 component predictors per failure type
  Table 15 - Trust x Complexity quadrant trade-off analysis
  Table 16 - C4 component ablation (leave-one-out, pairwise, forward selection)
  Table 17 - Deep sub-component signal forensics
  Table 18 - Method complementarity & answer confusion matrix
  Table 19 - "What if perfect [X]?" ceiling analysis
  Table 20 - Multi-label gold analysis (FR records with >1 gold answer)
  Table 21 - Switching ROC & precision-recall
  Table 22 - Difficulty ordering concordance (Kendall tau)
  Table 23 - Switch regret & switch happy sub-signal forensics
  + Loss/suboptimal cases exported to a separate JSON file
"""

# +======================================================================+
# |  CONFIGURATION -- edit these before running                          |
# +======================================================================+

THRESH_TRUST      = 0.80    # no-switch if trust >= this
THRESH_COMPLEXITY = 0.45    # no-switch if complexity < this
EPSILON           = 0.10    # +- band for "close call" analysis

# Threshold sensitivity grid for reporting switch-count tradeoffs
SENSITIVITY_TRUST_VALUES = [round(i * 0.10, 2) for i in range(11)]
SENSITIVITY_COMPLEXITY_VALUES = [round(i * 0.10, 2) for i in range(11)]

# Which grid to use as the "switch target" for primary analysis:
#   "full"   -> full_grid_yn_correct
#   "pruned" -> pruned_grid_yn_correct
#   "best"   -> True if EITHER full or pruned is True (oracle-best)
GRID_MODE = "full"

# File path to trust eval JSON — relative to the directory where you run this script
FILE_PATH = "./trustnew_eval_spartun_qwen314b.json"

# Path to grid results JSON (e.g. qwen32bbgridgen.json) for HARD accuracy
# Set to "" or None to skip hard grid accuracy (fall back to trust eval booleans)
GRID_RESULTS_FILE = r"./qwen14bgridgen (3).json"

# Where to write the loss / suboptimal-case JSON
LOSS_CASES_OUTPUT = "newspartun_switch_loss_casesqwen314b.json"
REGRET_CASES_OUTPUT = "newspartun_switch_regret_casesqwen314b.json"

# --- Error taxonomy (Table 11) settings ---
TAXONOMY_API_KEY = ""                          # blank → uses OPENAI_API_KEY env var
TAXONOMY_MODEL   = "gpt-5.1"                  # model for both INPUT + OUTPUT taxonomy calls
TAXONOMY_CACHE   = "newspartun_taxonomy_cacheqwen314b.json"  # cached results (auto-created)
TAXONOMY_NO_GPT  = False                       # True = skip GPT, use cache only
TAXONOMY_FORCE   = False                       # True = ignore cache, reclassify all

# =======================================================================

import json
import os
import math
import textwrap
from pathlib import Path
from collections import Counter, defaultdict

from config import OPENAI_API_KEY

# --- Hard accuracy helpers -------------------------------------------

def _hard_correct_yn(gold_list, pred):
    """YN hard check: gold and pred must match exactly (case-insensitive)."""
    if pred is None or gold_list is None:
        return None
    gold_set = {g.strip().lower() for g in gold_list}
    if isinstance(pred, list):
        pred_set = {str(p).strip().lower() for p in pred}
    else:
        pred_set = {str(pred).strip().lower()}
    return gold_set == pred_set


def _hard_correct_fr(gold_list, pred):
    """FR hard check: ALL gold answers must appear in pred (set subset)."""
    if pred is None or gold_list is None:
        return None
    gold_set = {g.strip().lower() for g in gold_list}
    if isinstance(pred, list):
        pred_set = {str(p).strip().lower() for p in pred}
    else:
        pred_set = {str(pred).strip().lower()}
    # Remove empty strings
    pred_set.discard("")
    if not pred_set:
        return False
    return gold_set.issubset(pred_set)


def _hard_check(q_type, gold_list, pred):
    """Dispatch to YN or FR hard check."""
    if q_type == "YN":
        return _hard_correct_yn(gold_list, pred)
    else:
        return _hard_correct_fr(gold_list, pred)


def _load_grid_results(grid_path):
    """Load grid results JSON and build lookup by (identifier, question_lower)."""
    if not grid_path:
        return {}
    p = Path(grid_path)
    if not p.exists():
        print(f"  WARNING: Grid results file not found: {p}")
        return {}
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    lookup = {}
    for rec in data:
        key = (rec.get("identifier", ""), rec.get("question", "").strip().lower())
        lookup[key] = rec
    print(f"  Grid results loaded: {len(data)} records ({len(set(r.get('identifier','') for r in data))} unique contexts)")
    return lookup


# --- File resolution ------------------------------------------------

def _resolve_file(explicit_path=None):
    try:
        _base = Path(__file__).resolve().parent
    except NameError:
        _base = Path.cwd()
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates += [
        _base / "trustnew_eval_spartun_qwen332b.json",
        _base.parent / "trustnew_eval_spartun_qwen332b.json",
        Path(explicit_path) if explicit_path else Path("noop"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    for d in [_base, _base.parent, Path.cwd()]:
        if d.is_dir():
            hits = list(d.glob("*spartun*.json")) + list(d.glob("*trust_eval*.json"))
            if hits:
                return hits[0].resolve()
    raise FileNotFoundError(
        f"Cannot find trust eval JSON. Set FILE_PATH explicitly.\nSearched from: {_base}"
    )


# --- Helpers --------------------------------------------------------

def load_records(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    if text[0] == "[":
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError:
            pass
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        recs = [json.loads(ln) for ln in lines]
        if recs and isinstance(recs[0], dict):
            return recs
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    recs, pos = [], 0
    while pos < len(text):
        try:
            obj, end = decoder.raw_decode(text, pos)
            recs.append(obj)
            pos = end
            while pos < len(text) and text[pos] in " \t\n\r":
                pos += 1
        except json.JSONDecodeError:
            break
    return recs


def safe(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def pct(num, den):
    return f"{num/den*100:5.1f}%" if den else "  -  "


def bar(num, den, width=20):
    if den == 0:
        return " " * width
    filled = int(round(num / den * width))
    return "#" * filled + "." * (width - filled)


def section(title):
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def subsection(title):
    print(f"\n  {'-' * 74}")
    print(f"  {title}")
    print(f"  {'-' * 74}")


def _pick_grid(rec, mode):
    """Pick the grid_correct value based on GRID_MODE."""
    fg = rec.get("full_grid_yn_correct")
    pg = rec.get("pruned_grid_yn_correct")
    if mode == "full":
        return bool(fg) if fg is not None else None
    elif mode == "pruned":
        return bool(pg) if pg is not None else None
    elif mode == "best":
        if fg is None and pg is None:
            return None
        return bool(fg) or bool(pg)
    return bool(fg) if fg is not None else None


# --- Extract fields (recomputes would_switch from user thresholds) --

def get_fields(rec, grid_lookup=None):
    scores    = rec.get("scores", {}) or {}
    lm        = rec.get("lm_correct", {}) or {}
    meta      = rec.get("meta", {}) or {}
    inputs    = rec.get("inputs", {}) or {}
    q1        = rec.get("Q1", {}) or {}
    q2        = rec.get("Q2", {}) or {}
    q3        = rec.get("Q3", {}) or {}
    complexity_obj = q3.get("complexity", {}) or {}
    components_c4  = complexity_obj.get("components", {}) or {}
    po        = rec.get("prompts_and_outputs", {}) or {}
    candidate_answers = rec.get("candidate_answers", []) or []
    timing    = rec.get("timing", {}) or {}

    trust_stored = safe(scores.get("trustworthiness_score"))
    complexity = safe(scores.get("complexity_score"))

    # Q1 sub-scores
    q1_s      = safe(q1.get("S_support_only", {}).get("score"))
    q1_c_score = safe(q1.get("C_story_minus_support", {}).get("score"))
    q1_score  = float((q1_s + q1_c_score) / 2.0)   # mean(suf, nec) — matches pipeline

    # Q2 sub-scores
    q2_score   = safe(q2.get("q2_score"))
    q2_avail   = q2.get("available", True)
    q2_short   = q2.get("short_circuited", False)
    q2_comps   = q2.get("components", {}) or {}
    para_stab  = safe(q2_comps.get("paraphrase_stability", q2.get("paraphrase_stability")))
    flip_score = safe(q2_comps.get("flip_score", 0))

    # Recompute trust from sub-signals (same formula as the maintained switch pipeline)
    trust = 0.60 * q1_score + 0.40 * float(q2_score)

    # -- Recompute would_switch from USER thresholds --
    # Match pipeline: strict > for trust, strict < for complexity
    would_switch = not (complexity < THRESH_COMPLEXITY and trust > THRESH_TRUST)
    would_switch_orig = scores.get("would_switch")

    # Grid correctness from trust eval (SOFT -- legacy)
    soft_full_grid_correct   = bool(rec.get("full_grid_yn_correct"))  if rec.get("full_grid_yn_correct") is not None else None
    soft_pruned_grid_correct = bool(rec.get("pruned_grid_yn_correct")) if rec.get("pruned_grid_yn_correct") is not None else None
    soft_grid_correct = _pick_grid(rec, GRID_MODE)

    # Text method correctness from trust eval (SOFT -- legacy)
    soft_text_correct      = bool(lm.get("baseline_symbolic_text_only"))
    soft_text_rels_correct = bool(lm.get("text_only_with_relations"))

    # Ground truth (list after merge, or legacy string)
    gt_raw = rec.get("ground_truth", "")
    if isinstance(gt_raw, list):
        gt_list = [str(g).lower() for g in gt_raw]
        gt = "|".join(sorted(gt_list))  # display string
    else:
        gt_list = [str(gt_raw).lower()]
        gt = str(gt_raw).lower()

    # Meta
    q_type       = meta.get("q_type", "?")
    asked_rel    = meta.get("asked_relation", "")
    idx          = meta.get("index", "?")
    dataset_id   = meta.get("dataset_id", "?")

    # Story and question
    story    = inputs.get("story", "") or ""
    question = inputs.get("question", "") or ""

    # Predictions from trust eval
    text_pred_raw  = (po.get("baseline_symbolic_text_only", {}) or {}).get("selected_option", "")
    rels_pred_raw  = (po.get("text_only_with_relations", {}) or {}).get("selected_option", "")

    # ---- HARD text accuracy: re-check text pred vs gold ----
    hard_text_correct = _hard_check(q_type, gt_list, text_pred_raw) if text_pred_raw else False
    hard_text_rels_correct = _hard_check(q_type, gt_list, rels_pred_raw) if rels_pred_raw else False

    # Stringify for display
    text_pred = text_pred_raw
    text_rels_pred = rels_pred_raw
    if isinstance(text_pred, list):
        text_pred = "|".join(sorted(str(x) for x in text_pred))
    if isinstance(text_rels_pred, list):
        text_rels_pred = "|".join(sorted(str(x) for x in text_rels_pred))

    # ---- HARD grid accuracy from external grid results file ----
    hard_full_grid_correct = None
    hard_pruned_grid_correct = None
    grid_matched = False
    grid_rec = None
    grid_full_pred = None
    grid_pruned_pred = None

    if grid_lookup is not None:
        lookup_key = (dataset_id, question.strip().lower())
        grid_rec = grid_lookup.get(lookup_key)
        if grid_rec is not None:
            grid_matched = True
            grid_gold = grid_rec.get("gold", [])
            grid_q_type = grid_rec.get("q_type", q_type)
            runs = grid_rec.get("runs", {})

            # Full grid YN
            fg_run = runs.get("full_grid_yn", {}) or {}
            grid_full_pred = fg_run.get("selected_option")
            if grid_full_pred is not None and grid_gold:
                hard_full_grid_correct = _hard_check(grid_q_type, grid_gold, grid_full_pred)

            # Pruned grid YN (rerun variant)
            pg_run = runs.get("pruned_grid_yn", {}) or {}
            grid_pruned_pred = pg_run.get("selected_option")
            if grid_pruned_pred is not None and grid_gold:
                hard_pruned_grid_correct = _hard_check(grid_q_type, grid_gold, grid_pruned_pred)

    # Pick which grid correct value to use (hard if available, else soft)
    if hard_full_grid_correct is not None:
        full_grid_correct = hard_full_grid_correct
    else:
        full_grid_correct = soft_full_grid_correct

    if hard_pruned_grid_correct is not None:
        pruned_grid_correct = hard_pruned_grid_correct
    else:
        pruned_grid_correct = soft_pruned_grid_correct

    # Pick which text correct value to use (hard always overrides)
    text_correct = hard_text_correct if hard_text_correct is not None else soft_text_correct
    text_rels_correct = hard_text_rels_correct if hard_text_rels_correct is not None else soft_text_rels_correct

    # Grid correct for chosen GRID_MODE
    if GRID_MODE == "full":
        grid_correct = full_grid_correct
    elif GRID_MODE == "pruned":
        grid_correct = pruned_grid_correct
    elif GRID_MODE == "best":
        grid_correct = bool(full_grid_correct) or bool(pruned_grid_correct)
    else:
        grid_correct = full_grid_correct

    # SPaRTUN C4 complexity components
    SB = safe(components_c4.get("support_burden_SB"))
    EL = safe(components_c4.get("entity_load_EL"))
    HL = safe(components_c4.get("hard_language_HL"))
    CF = safe(components_c4.get("coref_difficulty_CF"))

    return {
        "would_switch": would_switch,
        "would_switch_orig": would_switch_orig,
        "trust": trust,
        "trust_stored": trust_stored,
        "complexity": complexity,
        "full_grid_correct":   full_grid_correct,
        "pruned_grid_correct": pruned_grid_correct,
        "grid_correct":        grid_correct,
        "text_correct":        text_correct,
        "text_rels_correct":   text_rels_correct,
        # Soft (legacy) values for comparison
        "soft_text_correct":        soft_text_correct,
        "soft_text_rels_correct":   soft_text_rels_correct,
        "soft_full_grid_correct":   soft_full_grid_correct,
        "soft_pruned_grid_correct": soft_pruned_grid_correct,
        "soft_grid_correct":        soft_grid_correct,
        # Hard accuracy values
        "hard_text_correct":        hard_text_correct,
        "hard_text_rels_correct":   hard_text_rels_correct,
        "hard_full_grid_correct":   hard_full_grid_correct,
        "hard_pruned_grid_correct": hard_pruned_grid_correct,
        "grid_matched":             grid_matched,
        "grid_full_pred":           grid_full_pred,
        "grid_pruned_pred":         grid_pruned_pred,
        #
        "q1_score": q1_score, "q2_score": q2_score,
        "q2_avail": q2_avail, "q2_short": q2_short,
        "para_stab": para_stab, "flip_score": flip_score,
        "q1_s": q1_s, "q1_c": q1_c_score,
        "gt": gt, "gt_list": gt_list,
        "story": story, "question": question,
        "q_type": q_type, "asked_rel": asked_rel,
        "idx": idx, "dataset_id": dataset_id,
        "text_pred": text_pred, "text_rels_pred": text_rels_pred,
        "SB": SB, "EL": EL, "HL": HL, "CF": CF,
        "raw_record": rec,
        "meta": meta,
        "inputs": inputs,
        "scores": scores,
        "timing": timing,
        "candidate_answers": candidate_answers,
        "lm_correct": lm,
        "prompts_and_outputs": po,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "grid_record": grid_rec,
        "grid_runs": (grid_rec.get("runs", {}) or {}) if grid_rec else {},
        "grid_old_runs_snapshot": (grid_rec.get("old_runs_snapshot", {}) or {}) if grid_rec else {},
    }


def _build_case_payload(f, case_type, label):
    fc = f.get("failure_class", {}) if isinstance(f.get("failure_class"), dict) else {}
    return {
        "case_type": case_type,
        "label": label,
        "decision_summary": {
            "would_switch": f["would_switch"],
            "would_switch_orig": f.get("would_switch_orig"),
            "switch_regret": bool(f["would_switch"] and f["text_correct"] and not f["grid_correct"]),
            "stay_regret": bool((not f["would_switch"]) and (not f["text_correct"]) and f["grid_correct"]),
            "text_correct": f["text_correct"],
            "text_rels_correct": f["text_rels_correct"],
            "full_grid_correct": f["full_grid_correct"],
            "pruned_grid_correct": f["pruned_grid_correct"],
            "grid_correct": f["grid_correct"],
        },
        "metadata": {
            "idx": f["idx"],
            "q_type": f["q_type"],
            "asked_rel": f["asked_rel"],
            "dataset_id": f["dataset_id"],
            "ground_truth": f["gt"],
            "ground_truth_list": f["gt_list"],
            "candidate_answers": f.get("candidate_answers", []),
        },
        "text_answers": {
            "baseline_symbolic_text_only": {
                "prediction": f["text_pred"],
                "correct": f["text_correct"],
                "soft_correct": f.get("soft_text_correct"),
                "hard_correct": f.get("hard_text_correct"),
                "payload": (f.get("prompts_and_outputs", {}) or {}).get("baseline_symbolic_text_only", {}),
            },
            "text_only_with_relations": {
                "prediction": f["text_rels_pred"],
                "correct": f["text_rels_correct"],
                "soft_correct": f.get("soft_text_rels_correct"),
                "hard_correct": f.get("hard_text_rels_correct"),
                "payload": (f.get("prompts_and_outputs", {}) or {}).get("text_only_with_relations", {}),
            },
        },
        "scores": {
            "trust": round(f["trust"], 4),
            "trust_stored": round(f.get("trust_stored", 0.0), 4),
            "complexity": round(f["complexity"], 4),
            "q1_score": round(f["q1_score"], 4),
            "q2_score": round(f["q2_score"], 4),
            "q1_S": round(f["q1_s"], 4),
            "q1_C": round(f["q1_c"], 4),
            "para_stab": round(f["para_stab"], 4),
            "flip_score": round(f["flip_score"], 4),
            "SB": round(f["SB"], 4),
            "EL": round(f["EL"], 4),
            "HL": round(f["HL"], 4),
            "CF": round(f["CF"], 4),
            "scores_block": f.get("scores", {}),
        },
        "failure_taxonomy": {
            "failure_type": f.get("failure_type", ""),
            "failure_reasoning": fc.get("reasoning", ""),
            "failure_confidence": fc.get("confidence", None),
            "failure_secondary": fc.get("secondary_category", None),
            "raw_failure_class": fc,
            "input_analysis": f.get("tx_input", {}),
            "output_analysis": f.get("tx_output", {}),
        },
        "eval_record": {
            "meta": f.get("meta", {}),
            "inputs": f.get("inputs", {}),
            "timing": f.get("timing", {}),
            "lm_correct": f.get("lm_correct", {}),
            "Q1": f.get("Q1", {}),
            "Q2": f.get("Q2", {}),
            "Q3": f.get("Q3", {}),
            "prompts_and_outputs": f.get("prompts_and_outputs", {}),
            "raw_record": f.get("raw_record", {}),
        },
        "grid_record": {
            "grid_matched": f.get("grid_matched", False),
            "grid_full_pred": f.get("grid_full_pred"),
            "grid_pruned_pred": f.get("grid_pruned_pred"),
            "runs": f.get("grid_runs", {}),
            "old_runs_snapshot": f.get("grid_old_runs_snapshot", {}),
            "raw_record": f.get("grid_record", {}),
        },
    }


def print_example(f, label="", limit_story=200):
    story_d = f["story"][:limit_story] + ("..." if len(f["story"]) > limit_story else "")
    print(f"      [{label}] idx={f['idx']}  q_type={f['q_type']}  id={f['dataset_id']}")
    print(f"        Story:    {story_d}")
    print(f"        Question: {f['question']}")
    print(f"        GT={f['gt']}  grid(full)={f['full_grid_correct']}  "
          f"grid(prun)={f['pruned_grid_correct']}  text={f['text_correct']}")
    matched_tag = "  [grid-matched]" if f.get("grid_matched") else "  [no grid match]"
    print(f"        HARD text={f.get('hard_text_correct')}  "
          f"HARD grid(full)={f.get('hard_full_grid_correct')}  "
          f"HARD grid(prun)={f.get('hard_pruned_grid_correct')}{matched_tag}")
    print(f"        Soft text={f.get('soft_text_correct')}  "
          f"Soft grid(full)={f.get('soft_full_grid_correct')}  "
          f"Soft grid(prun)={f.get('soft_pruned_grid_correct')}")
    print(f"        Trust={f['trust']:.3f}  Complexity={f['complexity']:.3f}  "
          f"Q1={f['q1_score']:.2f} Q2={f['q2_score']:.2f}  would_switch={f['would_switch']}")
    print(f"        Q1_S(support)={f['q1_s']:.2f}  Q1_C(ablation)={f['q1_c']:.2f}  "
          f"Para_stab={f['para_stab']:.2f}  Flip={f['flip_score']:.2f}")
    print(f"        C4 components: SB={f['SB']:.2f} EL={f['EL']:.2f} "
          f"HL={f['HL']:.2f} CF={f['CF']:.2f}")
    print()


# ====================================================================
#   LOAD DATA
# ====================================================================

fpath = _resolve_file(FILE_PATH)
print(f"\n  Loading trust eval: {fpath}")
records = load_records(str(fpath))
print(f"  Records loaded: {len(records)}")

# Load external grid results file for hard accuracy
grid_lookup = _load_grid_results(GRID_RESULTS_FILE)

all_fields = [get_fields(r, grid_lookup=grid_lookup if grid_lookup else None) for r in records]
N = len(all_fields)

n_grid_matched = sum(1 for f in all_fields if f.get("grid_matched"))
print(f"  Grid-matched records: {n_grid_matched}/{N}")

print(f"  Thresholds:  trust >= {THRESH_TRUST}  (no-switch),  "
      f"complexity < {THRESH_COMPLEXITY}  (no-switch)")
print(f"  Grid mode: {GRID_MODE.upper()} (full_grid_yn_correct | pruned_grid_yn_correct | best-of-two)")
print(f"  Close-call epsilon: +-{EPSILON}")

# Summary of original vs recomputed would_switch
orig_ws  = sum(1 for r in records if r.get("scores", {}).get("would_switch"))
new_ws   = sum(1 for f in all_fields if f["would_switch"])
if orig_ws != new_ws:
    print(f"  !  Recomputed would_switch: {new_ws} switch (was {orig_ws} in file)")
else:
    print(f"  would_switch matches file: {new_ws} switch / {N - new_ws} stay")

q_type_counts = Counter(f["q_type"] for f in all_fields)
print(f"  Question types: {dict(q_type_counts)}")
multi_label = sum(1 for f in all_fields if len(f["gt_list"]) > 1)
print(f"  Multi-label gold records: {multi_label}")

# Convenience sublists
switch_cases   = [f for f in all_fields if f["would_switch"]]
noswitch_cases = [f for f in all_fields if not f["would_switch"]]


# ====================================================================
#   TABLE 0:  HARD vs SOFT accuracy comparison
# ====================================================================
section("TABLE 0: HARD vs SOFT accuracy comparison")

print(f"\n  HARD accuracy: YN = exact match, FR = ALL gold labels must be in prediction")
print(f"  SOFT accuracy: trust eval's original boolean fields (FR = any 1 gold label = correct)\n")

# --- Overall ---
def _acc_report(fields, label, key):
    total = sum(1 for f in fields if f.get(key) is not None)
    ok = sum(1 for f in fields if f.get(key) is True)
    return ok, total

for source_label, t_key, tr_key, fg_key, pg_key in [
    ("HARD", "hard_text_correct", "hard_text_rels_correct", "hard_full_grid_correct", "hard_pruned_grid_correct"),
    ("SOFT", "soft_text_correct", "soft_text_rels_correct", "soft_full_grid_correct", "soft_pruned_grid_correct"),
]:
    subsection(f"{source_label} accuracy summary")
    for metric_label, mkey in [
        ("Text-only (symbolic)", t_key),
        ("Text + relations", tr_key),
        ("Full grid YN", fg_key),
        ("Pruned grid YN", pg_key),
    ]:
        ok, total = _acc_report(all_fields, metric_label, mkey)
        print(f"    {metric_label:<30}  {ok:>4} / {total:<4} = {pct(ok, total)}")

# --- By q_type ---
subsection("HARD accuracy by question type")
for qt in ["YN", "FR"]:
    qt_fields = [f for f in all_fields if f["q_type"] == qt]
    if not qt_fields:
        continue
    print(f"\n    {qt} questions ({len(qt_fields)}):")
    for metric_label, mkey in [
        ("Text-only", "text_correct"),
        ("Text+rels", "text_rels_correct"),
        ("Full grid", "full_grid_correct"),
        ("Pruned grid", "pruned_grid_correct"),
    ]:
        ok = sum(1 for f in qt_fields if f.get(mkey) is True)
        n = len(qt_fields)
        print(f"      {metric_label:<20}  {ok:>4} / {n:<4} = {pct(ok, n)}")

# --- FR multi-label specifically ---
multi_fr_fields = [f for f in all_fields if f["q_type"] == "FR" and len(f["gt_list"]) > 1]
if multi_fr_fields:
    subsection(f"HARD accuracy on FR multi-label gold ({len(multi_fr_fields)} records)")
    for metric_label, mkey in [
        ("Text-only (hard)", "text_correct"),
        ("Text+rels (hard)", "text_rels_correct"),
        ("Full grid (hard)", "full_grid_correct"),
        ("Pruned grid (hard)", "pruned_grid_correct"),
    ]:
        ok = sum(1 for f in multi_fr_fields if f.get(mkey) is True)
        n = len(multi_fr_fields)
        print(f"    {metric_label:<25}  {ok:>4} / {n:<4} = {pct(ok, n)}")

    # Show what's typically missing
    print(f"\n    Missing label analysis (FR multi-label, grid-matched):")
    matched_multi = [f for f in multi_fr_fields if f.get("grid_matched")]
    missing_labels = Counter()
    for f in matched_multi:
        gold_set = set(f["gt_list"])
        if f.get("grid_full_pred"):
            pred_set = {str(p).strip().lower() for p in (f["grid_full_pred"] if isinstance(f["grid_full_pred"], list) else [f["grid_full_pred"]])}
            for g in gold_set - pred_set:
                missing_labels[g] += 1
    if missing_labels:
        print(f"    Full grid - most commonly missed gold labels:")
        for lbl, cnt in missing_labels.most_common(10):
            print(f"      {lbl:<20}  missed {cnt} times")

# --- Soft vs Hard delta ---
subsection("Soft -> Hard accuracy CHANGE (how many flipped)")
for metric_label, hard_key, soft_key in [
    ("Text-only", "hard_text_correct", "soft_text_correct"),
    ("Text+rels", "hard_text_rels_correct", "soft_text_rels_correct"),
    ("Full grid", "hard_full_grid_correct", "soft_full_grid_correct"),
    ("Pruned grid", "hard_pruned_grid_correct", "soft_pruned_grid_correct"),
]:
    soft_ok = sum(1 for f in all_fields if f.get(soft_key) is True)
    hard_ok = sum(1 for f in all_fields if f.get(hard_key) is True)
    # Count records where hard is available
    hard_avail = sum(1 for f in all_fields if f.get(hard_key) is not None)
    flipped_down = sum(1 for f in all_fields if f.get(soft_key) is True and f.get(hard_key) is False)
    flipped_up   = sum(1 for f in all_fields if f.get(soft_key) is False and f.get(hard_key) is True)
    print(f"    {metric_label:<20}  soft={soft_ok}  hard={hard_ok} (of {hard_avail} available)  "
          f"flipped_down={flipped_down}  flipped_up={flipped_up}")


# ====================================================================
#   TABLE 1:  would_switch x text-correct x grid accuracy
#             Shown for BOTH full and pruned grids
# ====================================================================
section("TABLE 1: would_switch x text-correct -> grid accuracy  (full & pruned)")

for grid_label, grid_key in [("FULL_GRID", "full_grid_correct"),
                               ("PRUNED_GRID", "pruned_grid_correct")]:
    print(f"\n  -- {grid_label} --")
    buckets = defaultdict(lambda: {"total": 0, "grid_ok": 0})
    for f in all_fields:
        key = (bool(f["would_switch"]), bool(f["text_correct"]))
        buckets[key]["total"] += 1
        if f[grid_key]:
            buckets[key]["grid_ok"] += 1

    total_grid_ok = sum(b["grid_ok"] for b in buckets.values())
    print(f"  {'would_switch':<14} {'text correct':<20} {'#cases':>7}   "
          f"{'grid accuracy':>18}   {'%':>7}")
    print(f"  {'-'*14} {'-'*20} {'-'*7}   {'-'*18}   {'-'*7}")
    for ws in [True, False]:
        for tc in [True, False]:
            b = buckets[(ws, tc)]
            n, g = b["total"], b["grid_ok"]
            print(f"  {str(ws):<14} {str(tc):<20} {n:>7}   "
                  f"{g:>5} / {n:<5}           {pct(g, n):>7}")
    print(f"  {'-'*14} {'-'*20} {'-'*7}   {'-'*18}   {'-'*7}")
    print(f"  {'TOTAL':<14} {'':20} {N:>7}   "
          f"{total_grid_ok:>5} / {N:<5}           {pct(total_grid_ok, N):>7}")

# Overall
text_ok    = sum(1 for f in all_fields if f["text_correct"])
rels_ok    = sum(1 for f in all_fields if f["text_rels_correct"])
fg_ok      = sum(1 for f in all_fields if f["full_grid_correct"])
pg_ok      = sum(1 for f in all_fields if f["pruned_grid_correct"])
best_ok    = sum(1 for f in all_fields if f["full_grid_correct"] or f["pruned_grid_correct"])
grid_ok    = sum(1 for f in all_fields if f["grid_correct"])

print(f"\n  Overall accuracy summary:")
print(f"    Baseline symbolic text-only: {text_ok}/{N} = {pct(text_ok,N)}")
print(f"    Text with relations:         {rels_ok}/{N} = {pct(rels_ok,N)}")
print(f"    Full grid YN:                {fg_ok}/{N} = {pct(fg_ok,N)}")
print(f"    Pruned grid YN:              {pg_ok}/{N} = {pct(pg_ok,N)}")
print(f"    Best-of-two grids:           {best_ok}/{N} = {pct(best_ok,N)}")
print(f"    Grid [GRID_MODE={GRID_MODE}]: {grid_ok}/{N} = {pct(grid_ok,N)}")

sw_grid  = sum(1 for f in switch_cases   if f["grid_correct"])
ns_grid  = sum(1 for f in noswitch_cases if f["grid_correct"])
print(f"    Grid acc (switch=T): {sw_grid}/{len(switch_cases)} = {pct(sw_grid,len(switch_cases))}")
print(f"    Grid acc (switch=F): {ns_grid}/{len(noswitch_cases)} = {pct(ns_grid,len(noswitch_cases))}")


# ====================================================================
#   TABLE 2:  WHY would_switch = True vs False
# ====================================================================
section("TABLE 2: WHY would_switch = True vs False  (trust & complexity breakdown)")

print(f"\n  RULE: stay text-only (no switch) iff  "
      f"complexity < {THRESH_COMPLEXITY}  AND  trust >= {THRESH_TRUST}")
print(f"        otherwise -> switch to grid\n")

reason_counts   = Counter()
reason_examples = defaultdict(list)

for f in all_fields:
    low_t   = f["trust"]      <  THRESH_TRUST
    high_c  = f["complexity"] >= THRESH_COMPLEXITY
    ws      = f["would_switch"]
    if ws:
        if low_t and high_c:
            reason = "switch: low trust AND high complexity"
        elif low_t:
            reason = "switch: low trust only"
        elif high_c:
            reason = "switch: high complexity only"
        else:
            reason = "switch: UNEXPECTED (neither condition)"
    else:
        reason = "no-switch: high trust AND low complexity"

    reason_counts[reason] += 1
    if len(reason_examples[reason]) < 2:
        reason_examples[reason].append(f)

print(f"  {'Reason':<52} {'#cases':>7}  {'% of total':>10}")
print(f"  {'-'*52} {'-'*7}  {'-'*10}")
for reason in sorted(reason_counts.keys()):
    cnt = reason_counts[reason]
    print(f"  {reason:<52} {cnt:>7}  {pct(cnt, N):>10}")

subsection("Trust & Complexity score distributions")
trust_vals  = [f["trust"]      for f in all_fields]
compl_vals  = [f["complexity"] for f in all_fields]
print(f"  Trust:      min={min(trust_vals):.3f}  max={max(trust_vals):.3f}  "
      f"mean={sum(trust_vals)/N:.3f}  median={sorted(trust_vals)[N//2]:.3f}")
print(f"  Complexity: min={min(compl_vals):.3f}  max={max(compl_vals):.3f}  "
      f"mean={sum(compl_vals)/N:.3f}  median={sorted(compl_vals)[N//2]:.3f}")

subsection("Trust score histogram")
bins = [(i/10, (i+1)/10) for i in range(10)]
bins[-1] = (0.9, 1.01)
for lo, hi in bins:
    cnt    = sum(1 for t in trust_vals if lo <= t < hi)
    marker = " <- threshold" if lo <= THRESH_TRUST < hi else ""
    print(f"  [{lo:.1f}, {hi:.1f})  {cnt:>4}  {bar(cnt, N, 30)}{marker}")

subsection("Complexity score histogram")
for lo, hi in bins:
    cnt    = sum(1 for c in compl_vals if lo <= c < hi)
    marker = " <- threshold" if lo <= THRESH_COMPLEXITY < hi else ""
    print(f"  [{lo:.1f}, {hi:.1f})  {cnt:>4}  {bar(cnt, N, 30)}{marker}")

subsection("Examples for each switch reason")
for reason in sorted(reason_examples.keys()):
    print(f"\n    > {reason}")
    for f in reason_examples[reason]:
        print_example(f, label=reason.split(":")[0])


# ====================================================================
#   TABLE 3:  Close-call / borderline decisions
# ====================================================================
section("TABLE 3: Borderline / close-call decisions")

trust_close = [f for f in all_fields if abs(f["trust"]      - THRESH_TRUST)      <= EPSILON]
compl_close = [f for f in all_fields if abs(f["complexity"] - THRESH_COMPLEXITY) <= EPSILON]
both_close  = [f for f in all_fields
               if abs(f["trust"] - THRESH_TRUST) <= EPSILON
               and abs(f["complexity"] - THRESH_COMPLEXITY) <= EPSILON]

print(f"\n  Trust within +-{EPSILON} of {THRESH_TRUST}:       {len(trust_close):>4} cases")
print(f"  Complexity within +-{EPSILON} of {THRESH_COMPLEXITY}:  {len(compl_close):>4} cases")
print(f"  Both close simultaneously:          {len(both_close):>4} cases")

tc_sw   = [f for f in trust_close if f["would_switch"]]
tc_stay = [f for f in trust_close if not f["would_switch"]]
print(f"\n  Among trust-borderline:")
print(f"    switch=True:  {len(tc_sw)}  (grid_ok: {sum(1 for f in tc_sw if f['grid_correct'])})")
print(f"    switch=False: {len(tc_stay)}  (grid_ok: {sum(1 for f in tc_stay if f['grid_correct'])})")
print(f"    text_correct: {sum(1 for f in trust_close if f['text_correct'])}/{len(trust_close)}")

if trust_close:
    subsection("Examples: trust-borderline decisions")
    for f in sorted(trust_close, key=lambda x: abs(x["trust"] - THRESH_TRUST))[:4]:
        gap = f["trust"] - THRESH_TRUST
        print_example(f, label=f"trust_gap={gap:+.3f}")


# ====================================================================
#   TABLE 4:  Very untrustworthy cases (trust < 0.2)
# ====================================================================
section("TABLE 4: Very untrustworthy cases (trust < 0.20)")

low_trust = sorted([f for f in all_fields if f["trust"] < 0.20], key=lambda x: x["trust"])
print(f"\n  Count: {len(low_trust)} / {N}")
lt_grid = sum(1 for f in low_trust if f["grid_correct"])
lt_text = sum(1 for f in low_trust if f["text_correct"])
print(f"  Grid accuracy:      {lt_grid}/{len(low_trust)} = {pct(lt_grid, len(low_trust))}")
print(f"  Text-only accuracy: {lt_text}/{len(low_trust)} = {pct(lt_text, len(low_trust))}")
print(f"  -> Grid improvement over text: {lt_grid - lt_text:+d}")

subsection("Examples: very untrustworthy (trust < 0.20)")
for f in low_trust[:4]:
    print_example(f, label=f"trust={f['trust']:.3f}")


# ====================================================================
#   TABLE 5:  Switch-helped vs Switch-hurt (BOTH grid variants)
# ====================================================================
section("TABLE 5: Switch decision outcome analysis  (full & pruned grids)")

for grid_label, grid_key in [("FULL_GRID", "full_grid_correct"),
                               ("PRUNED_GRID", "pruned_grid_correct")]:
    print(f"\n  -- {grid_label} --")
    sw_helped     = [f for f in switch_cases   if not f["text_correct"] and f[grid_key]]
    sw_hurt       = [f for f in switch_cases   if f["text_correct"]     and not f[grid_key]]
    sw_both_right = [f for f in switch_cases   if f["text_correct"]     and f[grid_key]]
    sw_both_wrong = [f for f in switch_cases   if not f["text_correct"] and not f[grid_key]]
    ns_tx_gd = [f for f in noswitch_cases if f["text_correct"]     and f[grid_key]]
    ns_tx_gb = [f for f in noswitch_cases if f["text_correct"]     and not f[grid_key]]
    ns_bx_gd = [f for f in noswitch_cases if not f["text_correct"] and f[grid_key]]
    ns_bx_gb = [f for f in noswitch_cases if not f["text_correct"] and not f[grid_key]]

    print(f"  When switch=TRUE ({len(switch_cases)} cases):")
    print(f"    Grid RESCUED  (textX->gridV): {len(sw_helped):>4}   {pct(len(sw_helped), len(switch_cases))}")
    print(f"    Grid BROKE IT (textV->gridX): {len(sw_hurt):>4}   {pct(len(sw_hurt), len(switch_cases))}")
    print(f"    Both correct  (textV, gridV): {len(sw_both_right):>4}   {pct(len(sw_both_right), len(switch_cases))}")
    print(f"    Both wrong    (textX, gridX): {len(sw_both_wrong):>4}   {pct(len(sw_both_wrong), len(switch_cases))}")
    print(f"    Net gain:  +{len(sw_helped)} rescued  -{len(sw_hurt)} broken  = {len(sw_helped)-len(sw_hurt):+d} net")

    print(f"  When switch=FALSE ({len(noswitch_cases)} cases):")
    print(f"    Stayed & correct (textV): {len(ns_tx_gd) + len(ns_tx_gb):>4}")
    print(f"    Stayed & wrong   (textX): {len(ns_bx_gd) + len(ns_bx_gb):>4}")
    print(f"    Missed rescue (textX, gridV): {len(ns_bx_gd):>4}  <- could have switched")
    print(f"    Dodged break  (textV, gridX): {len(ns_tx_gb):>4}  <- correctly avoided grid")

if any(not f["text_correct"] and f["grid_correct"] for f in switch_cases):
    subsection("Examples: Grid RESCUED (text wrong -> grid right)")
    for f in [x for x in switch_cases if not x["text_correct"] and x["grid_correct"]][:3]:
        print_example(f, label="rescued")

if any(f["text_correct"] and not f["grid_correct"] for f in switch_cases):
    subsection("Examples: Grid BROKE IT (text right -> grid wrong)")
    for f in [x for x in switch_cases if x["text_correct"] and not x["grid_correct"]][:3]:
        print_example(f, label="broken")


# ====================================================================
#   TABLE 6:  Complexity C4 component breakdown + q_type accuracy
# ====================================================================
section("TABLE 6: C4 complexity components + accuracy by question type (YN/FR)")

comp_keys = [("SB", "Support Burden"),
             ("EL", "Entity Load"),
             ("HL", "Hard Language"),
             ("CF", "Coref Difficulty")]

print(f"\n  {'Component':<22} {'mean':>7} {'median':>7} {'> 0.5':>6} "
      f"{'= 0':>6}  {'corr w/ grid_incorrect':>22}")
print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*6} {'-'*6}  {'-'*22}")

for key, label in comp_keys:
    vals        = [f[key] for f in all_fields]
    mean_v      = sum(vals) / N
    med_v       = sorted(vals)[N // 2]
    high        = sum(1 for v in vals if v > 0.5)
    zero        = sum(1 for v in vals if v == 0.0)
    wrong_vals  = [f[key] for f in all_fields if not f["grid_correct"]]
    right_vals  = [f[key] for f in all_fields if f["grid_correct"]]
    mean_wrong  = sum(wrong_vals) / max(1, len(wrong_vals))
    mean_right  = sum(right_vals) / max(1, len(right_vals))
    delta       = mean_wrong - mean_right
    print(f"  {label:<22} {mean_v:>7.3f} {med_v:>7.3f} {high:>6} {zero:>6}  "
          f"wrong={mean_wrong:.3f} right={mean_right:.3f} D={delta:+.3f}")

subsection("Accuracy by question type (YN vs FR)")
qtype_groups = defaultdict(list)
for f in all_fields:
    qtype_groups[f["q_type"]].append(f)

print(f"\n  {'Q-type':>8}  {'#cases':>7}  {'text_acc':>10}  {'rels_acc':>10}  "
      f"{'grid_acc':>10}  {'switch%':>11}  {'mean_T':>8}  {'mean_C':>8}")
print(f"  {'-'*8}  {'-'*7}  {'-'*10}  {'-'*10}  "
      f"{'-'*10}  {'-'*11}  {'-'*8}  {'-'*8}")
for qt in sorted(qtype_groups.keys()):
    grp = qtype_groups[qt]
    tc  = sum(1 for f in grp if f["text_correct"])
    rc  = sum(1 for f in grp if f["text_rels_correct"])
    gc  = sum(1 for f in grp if f["grid_correct"])
    sw  = sum(1 for f in grp if f["would_switch"])
    mt  = sum(f["trust"]      for f in grp) / len(grp)
    mc  = sum(f["complexity"] for f in grp) / len(grp)
    print(f"  {qt:>8}  {len(grp):>7}  {pct(tc,len(grp)):>10}  {pct(rc,len(grp)):>10}  "
          f"{pct(gc,len(grp)):>10}  {pct(sw,len(grp)):>11}  {mt:>8.3f}  {mc:>8.3f}")

subsection("FR questions: accuracy by asked_relation")
rel_groups = defaultdict(list)
for f in all_fields:
    if f["q_type"] == "FR":
        rel_groups[f["asked_rel"]].append(f)

if rel_groups:
    print(f"\n  {'Relation':<15} {'#cases':>7} {'text_acc':>10} {'rels_acc':>10} "
          f"{'grid_acc':>10} {'switch%':>12} {'mean_T':>8}")
    print(f"  {'-'*15} {'-'*7} {'-'*10} {'-'*10} "
          f"{'-'*10} {'-'*12} {'-'*8}")
    for rel in sorted(rel_groups.keys()):
        grp = rel_groups[rel]
        tc  = sum(1 for f in grp if f["text_correct"])
        rc  = sum(1 for f in grp if f["text_rels_correct"])
        gc  = sum(1 for f in grp if f["grid_correct"])
        sw  = sum(1 for f in grp if f["would_switch"])
        mt  = sum(f["trust"] for f in grp) / len(grp)
        print(f"  {rel:<15} {len(grp):>7} {pct(tc,len(grp)):>10} {pct(rc,len(grp)):>10} "
              f"{pct(gc,len(grp)):>10} {pct(sw,len(grp)):>12} {mt:>8.3f}")


# ====================================================================
#   TABLE 7:  Q1 vs Q2 contribution to trust
# ====================================================================
section("TABLE 7: Q1 (faithfulness) vs Q2 (plausibility) contribution to trust")

q2_full  = [f for f in all_fields if f["q2_avail"] and not f["q2_short"]]
q2_short = [f for f in all_fields if f["q2_short"]]
q2_none  = [f for f in all_fields if not f["q2_avail"]]

print(f"\n  Q2 available (full trust formula): {len(q2_full)}")
print(f"  Q2 short-circuited:                {len(q2_short)}")
print(f"  Q2 unavailable:                    {len(q2_none)}")

q1_vals   = [f["q1_score"]  for f in all_fields]
q1s_vals  = [f["q1_s"]      for f in all_fields]
q1c_vals  = [f["q1_c"]      for f in all_fields]
q2_vals   = [f["q2_score"]  for f in all_fields]
sub_para  = [f["para_stab"] for f in all_fields]
sub_flip  = [f["flip_score"] for f in all_fields]

subsection("Q1 sub-scores: S (support-only) and C (ablation)")
print(f"  Q1 overall: mean={sum(q1_vals)/N:.3f}  "
      f"Q1_S: mean={sum(q1s_vals)/N:.3f}  Q1_C: mean={sum(q1c_vals)/N:.3f}")
print(f"  Q2 overall: mean={sum(q2_vals)/N:.3f}  "
      f"(para_stab mean={sum(sub_para)/N:.3f}  flip mean={sum(sub_flip)/N:.3f})")

subsection("Q1 vs Q2 disagreement (|Q1-Q2| > 0.4)")
q1q2_disagree = [f for f in all_fields
                 if abs(f["q1_score"] - f["q2_score"]) > 0.4 and f["q2_avail"]]
print(f"  Count: {len(q1q2_disagree)}")
for f in q1q2_disagree[:3]:
    print_example(f, label=f"Q1={f['q1_score']:.2f} Q2={f['q2_score']:.2f}")

subsection("Q1/Q2 split by q_type")
for qt in sorted(qtype_groups.keys()):
    grp = qtype_groups[qt]
    mq1 = sum(f["q1_score"] for f in grp) / len(grp)
    mq2 = sum(f["q2_score"] for f in grp) / len(grp)
    mt  = sum(f["trust"]    for f in grp) / len(grp)
    print(f"  {qt}: Q1={mq1:.3f}  Q2={mq2:.3f}  Trust={mt:.3f}")


# ====================================================================
#   TABLE 8:  Oracle analysis
# ====================================================================
section("TABLE 8: Oracle analysis  -- was the switch decision optimal?")

oracle_correct = sum(1 for f in all_fields if f["text_correct"] or f["grid_correct"])
optimal, suboptimal = [], []

for f in all_fields:
    ws, tc, gc = f["would_switch"], bool(f["text_correct"]), bool(f["grid_correct"])
    if tc:
        oracle = False
    elif gc:
        oracle = True
    else:
        oracle = None
    if oracle is not None:
        (optimal if ws == oracle else suboptimal).append(f)

decidable = len(optimal) + len(suboptimal)
print(f"\n  Decidable cases (text XOR grid correct): {decidable}")
print(f"  Optimal decisions:    {len(optimal):>4}  {pct(len(optimal), decidable)}")
print(f"  Suboptimal decisions: {len(suboptimal):>4}  {pct(len(suboptimal), decidable)}")

sub_should_sw   = [f for f in suboptimal if not f["would_switch"] and not f["text_correct"] and f["grid_correct"]]
sub_should_stay = [f for f in suboptimal if f["would_switch"] and f["text_correct"] and not f["grid_correct"]]
print(f"\n  Suboptimal: should have switched:  {len(sub_should_sw)}")
print(f"  Suboptimal: should have stayed:    {len(sub_should_stay)}")

if sub_should_sw:
    subsection("Examples: Should have switched (textX gridV, but stayed)")
    for f in sub_should_sw[:3]:
        print_example(f, label="missed_switch")

if sub_should_stay:
    subsection("Examples: Should have stayed (textV gridX, but switched)")
    for f in sub_should_stay[:3]:
        print_example(f, label="bad_switch")


# ====================================================================
#   TABLE 9:  Accuracy by ground-truth relation type + q_type
# ====================================================================
section("TABLE 9: Accuracy by ground-truth relation and question type")

# YN: just yes/no
# FR: per relation in gold list (handle multi-label by primary label)
gt_groups = defaultdict(list)
for f in all_fields:
    primary_gt = f["gt_list"][0] if f["gt_list"] else "?"
    gt_groups[primary_gt].append(f)

print(f"\n  {'Relation / GT':<18} {'#cases':>7} {'text_acc':>10} {'rels_acc':>9} "
      f"{'grid_acc':>10} {'switch%':>12} {'mean_T':>8}")
print(f"  {'-'*18} {'-'*7} {'-'*10} {'-'*9} "
      f"{'-'*10} {'-'*12} {'-'*8}")
for gt_rel in sorted(gt_groups.keys()):
    grp = gt_groups[gt_rel]
    tc  = sum(1 for f in grp if f["text_correct"])
    rc  = sum(1 for f in grp if f["text_rels_correct"])
    gc  = sum(1 for f in grp if f["grid_correct"])
    sw  = sum(1 for f in grp if f["would_switch"])
    mt  = sum(f["trust"] for f in grp) / len(grp)
    print(f"  {gt_rel:<18} {len(grp):>7} {pct(tc,len(grp)):>10} {pct(rc,len(grp)):>9} "
          f"{pct(gc,len(grp)):>10} {pct(sw,len(grp)):>12} {mt:>8.3f}")


# ====================================================================
#   TABLE 10:  Optimal threshold search
# ====================================================================
section("TABLE 10: Optimal threshold search  (grid sweep T x C)")

print(f"\n  Sweeping trust in [0,1] x complexity in [0,1]  (step=0.05)")
print(f"  Rule: no-switch iff complexity < C_thresh AND trust >= T_thresh\n")

curr_correct = sum(
    1 for f in all_fields
    if (f["would_switch"] and f["grid_correct"])
    or (not f["would_switch"] and f["text_correct"])
)

STEP = 0.05
results_grid = []
best_correct_t10, best_t_t10, best_c_t10 = 0, 0, 0
t_range = [round(i * STEP, 3) for i in range(int(1 / STEP) + 1)]
c_range = [round(i * STEP, 3) for i in range(int(1 / STEP) + 1)]

for t_th in t_range:
    for c_th in c_range:
        pc = ns = 0
        for f in all_fields:
            ws = not (f["complexity"] < c_th and f["trust"] >= t_th)
            if ws:
                ns += 1
                if f["grid_correct"]:
                    pc += 1
            else:
                if f["text_correct"]:
                    pc += 1
        results_grid.append((t_th, c_th, pc, ns))
        if pc > best_correct_t10:
            best_correct_t10, best_t_t10, best_c_t10 = pc, t_th, c_th

results_grid.sort(key=lambda x: (-x[2], x[3]))
print(f"  {'Rank':>4}  {'T_thresh':>10}  {'C_thresh':>10}  "
      f"{'policy_acc':>11}  {'correct':>8}  {'#switch':>8}  {'#stay':>6}")
print(f"  {'-'*4}  {'-'*10}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*8}  {'-'*6}")
for rank, (tt, ct, pc, ns) in enumerate(results_grid[:15], 1):
    marker = " <- BEST" if rank == 1 else ""
    curr   = " (current)" if abs(tt - THRESH_TRUST) < 0.001 and abs(ct - THRESH_COMPLEXITY) < 0.001 else ""
    print(f"  {rank:>4}  {tt:>10.2f}  {ct:>10.2f}  "
          f"{pct(pc, N):>11}  {pc:>8}  {ns:>8}  {N - ns:>6}{marker}{curr}")

print(f"\n  Current ({THRESH_TRUST},{THRESH_COMPLEXITY}):  {curr_correct}/{N} = {pct(curr_correct, N)}")
print(f"  Best found ({best_t_t10},{best_c_t10}):  {best_correct_t10}/{N} = {pct(best_correct_t10, N)}")
print(f"  Oracle:                           {oracle_correct}/{N} = {pct(oracle_correct, N)}")
print(f"  Improvement from threshold tuning:  {best_correct_t10 - curr_correct:+d} cases")
print(f"  Remaining gap to oracle:            {oracle_correct - best_correct_t10:+d} cases")
print(f"  Always-switch (grid only):          {grid_ok}/{N} = {pct(grid_ok,N)}")
print(f"  Never-switch  (text only):          {text_ok}/{N} = {pct(text_ok,N)}")

subsection("Threshold sensitivity (tau_t, tau_c -> final accuracy)")
print(f"\n  {'tau_t':>6}  {'tau_c':>6}  {'switched':>8}  {'stay':>6}  {'missed':>7}  {'bad_sw':>7}  {'final_acc':>10}")
print(f"  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*10}")

_sens_rows = []
for tau_t in SENSITIVITY_TRUST_VALUES:
    for tau_c in SENSITIVITY_COMPLEXITY_VALUES:
        switched = 0
        stay = 0
        missed = 0
        bad_sw = 0
        final_correct = 0
        for f in all_fields:
            ws = not (f["complexity"] < tau_c and f["trust"] >= tau_t)
            if ws:
                switched += 1
                if f["grid_correct"]:
                    final_correct += 1
                if f["text_correct"] and not f["grid_correct"]:
                    bad_sw += 1
            else:
                stay += 1
                if f["text_correct"]:
                    final_correct += 1
                if (not f["text_correct"]) and f["grid_correct"]:
                    missed += 1
        _sens_rows.append((tau_t, tau_c, switched, stay, missed, bad_sw, final_correct))

_sens_best = max(row[6] for row in _sens_rows) if _sens_rows else 0
for tau_t, tau_c, switched, stay, missed, bad_sw, final_correct in _sens_rows:
    marker = " <--" if final_correct == _sens_best else ""
    print(f"  {tau_t:>6.2f}  {tau_c:>6.2f}  {switched:>8}  {stay:>6}  {missed:>7}  {bad_sw:>7}  {pct(final_correct, N):>10}{marker}")

# --- Save oracle-gap cases (suboptimal even at best thresholds) ---
_gap_cases = []
for f in all_fields:
    best_ws = not (f["complexity"] < best_c_t10 and f["trust"] >= best_t_t10)
    best_policy_correct = (best_ws and f["grid_correct"]) or (not best_ws and f["text_correct"])
    oracle_would_correct = f["text_correct"] or f["grid_correct"]
    if oracle_would_correct and not best_policy_correct:
        _gap_cases.append({
            "idx":                f["idx"],
            "dataset_id":         f["dataset_id"],
            "q_type":             f["q_type"],
            "asked_rel":          f["asked_rel"],
            "story":              f["story"],
            "question":           f["question"],
            "ground_truth":       f["gt"],
            "text_pred":          f["text_pred"],
            "text_correct":       bool(f["text_correct"]),
            "full_grid_correct":  f["full_grid_correct"],
            "pruned_grid_correct":f["pruned_grid_correct"],
            "trust":              round(f["trust"], 4),
            "complexity":         round(f["complexity"], 4),
            "q1_score":           round(f["q1_score"], 4),
            "q2_score":           round(f["q2_score"], 4),
            "q1_S":               round(f["q1_s"], 4),
            "q1_C":               round(f["q1_c"], 4),
            "para_stab":          round(f["para_stab"], 4),
            "flip_score":         round(f["flip_score"], 4),
            "SB": round(f["SB"], 4), "EL": round(f["EL"], 4),
            "HL": round(f["HL"], 4), "CF": round(f["CF"], 4),
            "best_thresh_would_switch": best_ws,
            "oracle_should_switch":     not f["text_correct"] and f["grid_correct"],
            "reason": "should_have_switched" if (not f["text_correct"] and f["grid_correct"]) else "should_have_stayed",
        })
_gap_cases.sort(key=lambda x: (x["reason"], x["q_type"], x["idx"]))
_gap_out = (fpath.parent if fpath.parent.is_dir() else Path.cwd()) / "oracle_gap_cases.json"
with open(_gap_out, "w", encoding="utf-8") as _gf:
    json.dump(_gap_cases, _gf, indent=2, ensure_ascii=False)
_gap_sw = sum(1 for c in _gap_cases if c["reason"] == "should_have_switched")
_gap_st = sum(1 for c in _gap_cases if c["reason"] == "should_have_stayed")
print(f"\n  Oracle-gap cases saved to: {_gap_out}")
print(f"    Total: {len(_gap_cases)}  (should_have_switched={_gap_sw}  should_have_stayed={_gap_st})")

subsection("Accuracy heatmap (rows=T_thresh, cols=C_thresh)")
HSTEP   = 0.10
ht_r    = [round(i * HSTEP, 2) for i in range(int(1 / HSTEP) + 1)]
hc_r    = [round(i * HSTEP, 2) for i in range(int(1 / HSTEP) + 1)]
header  = "  T \\ C  " + "".join(f"{c:>7}" for c in hc_r)
print(header)
print("  " + "-" * (len(header) - 2))
for tt in ht_r:
    row = f"  {tt:>5}  "
    for ct in hc_r:
        pc = 0
        for f in all_fields:
            ws = not (f["complexity"] < ct and f["trust"] > tt)
            pc += int(bool(f["grid_correct"])) if ws else int(bool(f["text_correct"]))
        acc = pc / N * 100
        if abs(tt - best_t_t10) < HSTEP / 2 and abs(ct - best_c_t10) < HSTEP / 2:
            row += f" [{acc:4.1f}]"
        elif abs(tt - THRESH_TRUST) < HSTEP / 2 and abs(ct - THRESH_COMPLEXITY) < HSTEP / 2:
            row += f" *{acc:4.1f}*"
        else:
            row += f"  {acc:4.1f} "
    print(row)
print(f"\n  [ ] = best pair,  * * = current pair")


# ====================================================================
#   TABLE 11:  SPaRTUN error taxonomy (INPUT difficulty + OUTPUT failure, two calls)
# ====================================================================
section("TABLE 11: Error taxonomy -- input difficulty + output failure analysis")

# ---------- config ----------
_TX_API_KEY = TAXONOMY_API_KEY or OPENAI_API_KEY
_TX_MODEL   = TAXONOMY_MODEL
_TX_CACHE_F = TAXONOMY_CACHE
_TX_NO_GPT  = TAXONOMY_NO_GPT
_TX_FORCE   = TAXONOMY_FORCE

# ---------- categories ----------
INPUT_CATEGORIES = [
    "entity_containment_complexity",  # many entities + many containment/topology rels
    "multihop_reasoning",             # answer requires chaining 2+ relations
]

FAILURE_CATEGORIES = [
    "hallucination",
    "linguistic_difficulty",
    "multihop_reasoning_error",
    "other_reasoning_error",
]

fc_labels = {
    "entity_containment_complexity": "Multi-entity containment complexity",
    "multihop_reasoning":            "Multi-hop reasoning chain",
    "hallucination":                 "Hallucination",
    "linguistic_difficulty":         "Linguistic difficulty",
    "multihop_reasoning_error":      "Multi-hop reasoning error",
    "other_reasoning_error":         "Other reasoning error",
}

# ---------- prompts ----------
_TAXONOMY_INPUT_PROMPT = (
    "You are analyzing the INPUT difficulty of a spatial reasoning problem.\n"
    "You can ONLY see the story and question -- do NOT see any model output.\n"
    "\n"
    "Story:\n{story}\n"
    "\n"
    "Question: {question}\n"
    "Ground truth answer: {gt}\n"
    "\n"
    "Analyze this input for the PRESENCE of each difficulty factor below.\n"
    "For each factor, answer true/false and give a one-sentence reason.\n"
    "\n"
    "1. entity_containment_complexity\n"
    "   Does the story involve MANY entities (5 or more) connected by multiple containment\n"
    "   or topology relations (ntpp, tpp, ntppi, tppi, ec, po, dc)? High complexity means\n"
    "   a model must track a dense relational graph just to read the relations correctly.\n"
    "   Simple stories with 2-3 entities and 1-2 direct relations should be marked false.\n"
    "\n"
    "2. multihop_reasoning\n"
    "   Does answering the question require CHAINING through 2 or more relation steps?\n"
    '   E.g., "A is in B, B is in C -- is A in C?" (2-hop), or longer chains, or any step\n'
    "   that requires applying an inverse (if A is inside B then B contains A) as part of\n"
    "   a multi-step derivation. Mark chain_length as the number of reasoning steps needed.\n"
    "   Single-step or direct one-relation lookups should be marked false (chain_length=1).\n"
    "\n"
    "Return ONLY valid JSON (no markdown):\n"
    '{{\n'
    '  "entity_containment_complexity": {{"present": true/false, "entity_count": <int>, "reason": "..."}},\n'
    '  "multihop_reasoning": {{"present": true/false, "chain_length": <int>, "reason": "..."}}\n'
    '}}'
)

_TAXONOMY_OUTPUT_PROMPT = (
    "You are analyzing WHY a spatial reasoning model gave a WRONG answer.\n"
    "You see the story, question, AND the model's wrong prediction.\n"
    "\n"
    "Story:\n{story}\n"
    "\n"
    "Question: {question}\n"
    "Ground truth answer: {gt}\n"
    "Model's (wrong) answer: {prediction}\n"
    "Question type: {q_type}\n"
    "\n"
    "The model got this WRONG. For EACH category below, decide independently whether it\n"
    "contributed to the error. Multiple categories CAN be true simultaneously.\n"
    "\n"
    "1. hallucination\n"
    "   Did the model introduce or rely on a spatial relation NOT stated or logically implied\n"
    "   by the story? This includes inventing entity positions, silently dropping entities,\n"
    "   or asserting containment/adjacency that contradicts the story.\n"
    "\n"
    "2. linguistic_difficulty\n"
    "   Did the model likely fail to PARSE the language of the story correctly?\n"
    '   E.g., misinterpreting topology terms ("touching", "overlapping", "partially overlaps"),\n'
    "   clock-face positions, unusual phrasing, or confusing entity names so that the wrong\n"
    "   relation was assigned to the wrong entity pair.\n"
    "\n"
    "3. multihop_reasoning_error\n"
    "   Did the model likely understand individual relations but fail to CHAIN them?\n"
    "   E.g., it needed A-in-B plus B-in-C to conclude A-in-C but stopped at one hop, or it\n"
    "   needed to apply an inverse step (A inside B => B contains A) as part of a chain and\n"
    "   failed to do so. This category covers transitivity failures, inverse-relation errors,\n"
    "   and coreference-chain errors that arise from following a reasoning chain across steps.\n"
    "\n"
    "4. other_reasoning_error\n"
    "   The model parsed language fine, did not hallucinate, and did not need multi-hop\n"
    "   reasoning, but STILL got it wrong -- e.g., confused entity roles, mis-selected from\n"
    "   the candidate list despite apparently correct partial reasoning, or made an error that\n"
    "   does not fit any of the above. Use as a CATCH-ALL only when 1-3 do not explain it.\n"
    "\n"
    "Return ONLY valid JSON (no markdown):\n"
    '{{\n'
    '  "hallucination": {{"present": true/false, "reason": "..."}},\n'
    '  "linguistic_difficulty": {{"present": true/false, "reason": "..."}},\n'
    '  "multihop_reasoning_error": {{"present": true/false, "reason": "..."}},\n'
    '  "other_reasoning_error": {{"present": true/false, "reason": "..."}}\n'
    '}}'
)

# ---------- cache ----------
import openai as _oai_t11
from concurrent.futures import ThreadPoolExecutor as _ClassTPE, as_completed as _cls_done
import re as _re11

_tx_cache = {}
if not _TX_FORCE and os.path.exists(_TX_CACHE_F):
    with open(_TX_CACHE_F, encoding="utf-8") as _cf:
        _tx_cache = json.load(_cf)
    print(f"  [cache] Loaded {len(_tx_cache)} taxonomy records from {_TX_CACHE_F}")

text_incorrect = [f for f in all_fields if not f["text_correct"]]
print(f"\n  Text-incorrect cases: {len(text_incorrect)}")

# ---------- GPT call ----------
def _tx_call(prompt_text):
    _c = _oai_t11.OpenAI(api_key=_TX_API_KEY)
    resp = _c.responses.create(
        model=_TX_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text":
                "You are a spatial reasoning analyst. Return only valid JSON, no markdown."}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt_text}]},
        ],
    )
    return resp.output_text.strip()

def _parse_tx_flags(raw):
    m = _re11.search(r'\{.*\}', raw, _re11.DOTALL)
    try:
        return json.loads(m.group() if m else raw)
    except Exception:
        return {}

def _classify_tx(f_dict):
    did = str(f_dict["dataset_id"])
    idx = str(f_dict["idx"])
    key = f"{did}::{idx}"

    # CALL 1: INPUT difficulty (story + question only, no model output)
    try:
        inp_raw = _tx_call(_TAXONOMY_INPUT_PROMPT.format(
            story=f_dict["story"],
            question=f_dict["question"],
            gt=f_dict["gt"],
        ))
        inp_obj = _parse_tx_flags(inp_raw)
    except Exception as e:
        inp_raw, inp_obj = str(e), {}

    # CALL 2: OUTPUT failure analysis (story + question + wrong prediction)
    try:
        out_raw = _tx_call(_TAXONOMY_OUTPUT_PROMPT.format(
            story=f_dict["story"],
            question=f_dict["question"],
            gt=f_dict["gt"],
            prediction=f_dict["text_pred"] or "(empty/unparseable)",
            q_type=f_dict["q_type"],
        ))
        out_obj = _parse_tx_flags(out_raw)
    except Exception as e:
        out_raw, out_obj = str(e), {}

    return key, {
        "dataset_id": did, "idx": idx,
        "input_analysis":  {cat: inp_obj.get(cat, {}) for cat in INPUT_CATEGORIES},
        "output_analysis": {cat: out_obj.get(cat, {}) for cat in FAILURE_CATEGORIES},
        "input_raw": inp_raw, "output_raw": out_raw,
    }

if not _TX_API_KEY:
    print("\n  !! No API key -- skipping taxonomy GPT calls.")
    print("     Set TAXONOMY_API_KEY in config or OPENAI_API_KEY env var.")
    _TX_CLASSIFY_RAN = False
elif _TX_NO_GPT:
    print("  [TAXONOMY_NO_GPT=True] Using cache only.")
    _TX_CLASSIFY_RAN = False
else:
    _TX_CLASSIFY_RAN = True

if _TX_CLASSIFY_RAN:
    _to_classify = [f for f in text_incorrect
                    if f"{f['dataset_id']}::{f['idx']}" not in _tx_cache or _TX_FORCE]
    print(f"  Running GPT taxonomy: {len(_to_classify)} new + "
          f"{len(text_incorrect)-len(_to_classify)} from cache ...")
    with _ClassTPE(max_workers=12) as _pool11:
        _futs11 = {_pool11.submit(_classify_tx, f): f["dataset_id"] for f in _to_classify}
        _done11 = 0
        for fut in _cls_done(_futs11):
            key, result = fut.result()
            _tx_cache[key] = result
            _done11 += 1
            if _done11 % 20 == 0:
                print(f"    {_done11}/{len(_to_classify)} done ...")
    with open(_TX_CACHE_F, "w", encoding="utf-8") as _cf:
        json.dump(_tx_cache, _cf, indent=2)
    print(f"  [cache] Saved {len(_tx_cache)} records to {_TX_CACHE_F}")

# Attach taxonomy results to each failure record
def _primary_output_cat(out_obj):
    for cat in ["multihop_reasoning_error", "hallucination",
                "linguistic_difficulty", "other_reasoning_error"]:
        if out_obj.get(cat, {}).get("present"):
            return cat
    return "none_classified"

failure_classifications = {}
for f in text_incorrect:
    key = f"{f['dataset_id']}::{f['idx']}"
    entry = _tx_cache.get(key, {})
    f["tx_input"]      = entry.get("input_analysis", {})
    f["tx_output"]     = entry.get("output_analysis", {})
    f["failure_class"] = entry
    f["failure_type"]  = _primary_output_cat(f["tx_output"])
    failure_classifications[key] = entry

# ---- 11a. Input difficulty distribution ----
subsection("11a. Input difficulty distribution")

n_fails = len(text_incorrect)
print(f"\n  {'Input Factor':<42} {'present':>8} {'% of fails':>11}")
print(f"  {'-'*42} {'-'*8} {'-'*11}")
for cat in INPUT_CATEGORIES:
    cnt = sum(1 for f in text_incorrect if f["tx_input"].get(cat, {}).get("present"))
    print(f"  {fc_labels[cat]:<42} {cnt:>8} {pct(cnt, n_fails):>11}")

# Chain-length distribution for multihop
_mh_chains = [
    f["tx_input"].get("multihop_reasoning", {}).get("chain_length")
    for f in text_incorrect
    if f["tx_input"].get("multihop_reasoning", {}).get("present")
]
_valid_chains = [c for c in _mh_chains if isinstance(c, (int, float)) and c > 0]
if _valid_chains:
    _chain_ctr = Counter(int(c) for c in _valid_chains)
    print(f"\n  Multi-hop chain-length distribution (among present):")
    for _length in sorted(_chain_ctr):
        print(f"    chain={_length}: {_chain_ctr[_length]}")

# Entity count distribution for containment complexity
_ent_counts = [
    f["tx_input"].get("entity_containment_complexity", {}).get("entity_count")
    for f in text_incorrect
    if f["tx_input"].get("entity_containment_complexity", {}).get("present")
]
_valid_ents = [e for e in _ent_counts if isinstance(e, (int, float)) and e > 0]
if _valid_ents:
    print(f"\n  Entity count distribution (among entity_containment_complexity=true):")
    _ent_ctr = Counter(int(e) for e in _valid_ents)
    for _ec in sorted(_ent_ctr):
        print(f"    entities={_ec}: {_ent_ctr[_ec]}")

# ---- 11b. Output failure type distribution ----
subsection("11b. Output failure type distribution (by q_type)")

_output_flag_counts = {cat: 0 for cat in FAILURE_CATEGORIES}
for f in text_incorrect:
    for cat in FAILURE_CATEGORIES:
        if f["tx_output"].get(cat, {}).get("present"):
            _output_flag_counts[cat] += 1

print(f"\n  {'Failure Type':<42} {'present':>8} {'YN':>5} {'FR':>5} {'rescue%':>9}")
print(f"  {'-'*42} {'-'*8} {'-'*5} {'-'*5} {'-'*9}")
for cat in FAILURE_CATEGORIES:
    cnt = _output_flag_counts[cat]
    grp = [f for f in text_incorrect if f["tx_output"].get(cat, {}).get("present")]
    yn_cnt  = sum(1 for f in grp if f["q_type"] == "YN")
    fr_cnt  = sum(1 for f in grp if f["q_type"] == "FR")
    rescued = sum(1 for f in grp if f["grid_correct"])
    print(f"  {fc_labels[cat]:<42} {cnt:>8} {yn_cnt:>5} {fr_cnt:>5} {pct(rescued, cnt):>9}")

_none_count = sum(1 for f in text_incorrect if f["failure_type"] == "none_classified")
if _none_count:
    print(f"  {'(none triggered)':<42} {_none_count:>8}")

# ---- 11c. Cross-tab: input difficulty x output failure ----
subsection("11c. Input difficulty co-occurrence with output failures")

_short = {"hallucination": "hall", "linguistic_difficulty": "ling",
          "multihop_reasoning_error": "mhop", "other_reasoning_error": "othr"}
print(f"\n  {'Input Factor':<40}", end="")
for cat in FAILURE_CATEGORIES:
    print(f"  {_short.get(cat, cat[:4]):>5}", end="")
print(f"  {'n':>5}")
print(f"  {'-'*40}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'-----':>5}  {'-----':>5}")
for inp_cat in INPUT_CATEGORIES:
    grp = [f for f in text_incorrect if f["tx_input"].get(inp_cat, {}).get("present")]
    print(f"  {fc_labels[inp_cat]:<40}", end="")
    for out_cat in FAILURE_CATEGORIES:
        cnt = sum(1 for f in grp if f["tx_output"].get(out_cat, {}).get("present"))
        print(f"  {cnt:>5}", end="")
    print(f"  {len(grp):>5}")

# ---- 11d. Examples per output failure type ----
subsection("11d. Examples per output failure type")
for cat in FAILURE_CATEGORIES:
    grp = [f for f in text_incorrect if f["tx_output"].get(cat, {}).get("present")]
    if not grp:
        continue
    print(f"\n    >> {fc_labels.get(cat, cat)}  ({len(grp)} cases)")
    for f in grp[:2]:
        out_info = f["tx_output"].get(cat, {})
        print(f"      [{f['dataset_id']}] q_type={f['q_type']}  GT={f['gt']}  pred={f['text_pred']}")
        print(f"        T={f['trust']:.3f}  C={f['complexity']:.3f}  switch={f['would_switch']}")
        print(f"        GPT: {out_info.get('reason', '')}")

#   TABLE 12:  Q1/Q2/Q3 individual sufficiency
# ====================================================================
section("TABLE 12: Q1 vs Q2 vs Q3 -- what does each bring?")

def _sweep_single_low(score_key, all_f, step=0.05):
    """Switch if score < threshold (lower = more unreliable)."""
    thresholds = [round(i * step, 3) for i in range(int(1 / step) + 1)]
    best_c, best_t, best_ns = 0, 0, 0
    for th in thresholds:
        pc = ns = 0
        for f in all_f:
            ws = f[score_key] < th
            if ws:
                ns += 1
                pc += int(bool(f["grid_correct"]))
            else:
                pc += int(bool(f["text_correct"]))
        if pc > best_c:
            best_c, best_t, best_ns = pc, th, ns
    return best_c / max(1, len(all_f)), best_t, best_c, best_ns

def _sweep_single_high(score_key, all_f, step=0.05):
    """Switch if score >= threshold (higher = more complex => switch)."""
    thresholds = [round(i * step, 3) for i in range(int(1 / step) + 1)]
    best_c, best_t, best_ns = 0, 0, 0
    for th in thresholds:
        pc = ns = 0
        for f in all_f:
            ws = f[score_key] >= th
            if ws:
                ns += 1
                pc += int(bool(f["grid_correct"]))
            else:
                pc += int(bool(f["text_correct"]))
        if pc > best_c:
            best_c, best_t, best_ns = pc, th, ns
    return best_c / max(1, len(all_f)), best_t, best_c, best_ns

q1_acc,  q1_th,  q1_cor,  q1_ns  = _sweep_single_low("q1_score", all_fields)
q2_acc,  q2_th,  q2_cor,  q2_ns  = _sweep_single_low("q2_score", all_fields)
tr_acc,  tr_th,  tr_cor,  tr_ns  = _sweep_single_low("trust", all_fields)
q3_acc,  q3_th,  q3_cor,  q3_ns  = _sweep_single_high("complexity", all_fields)
comb_cor = curr_correct

subsection("12a. Single-score switching: best achievable via each score alone")
print(f"\n  {'Decision basis':<35} {'best_acc':>10} {'correct':>8} {'threshold':>10} "
      f"{'#switch':>8} {'gap_oracle':>11}")
print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*11}")
for label, acc, cor, th_v, ns in [
    ("Q1 alone",              q1_acc, q1_cor, q1_th, q1_ns),
    ("Q2 alone",              q2_acc, q2_cor, q2_th, q2_ns),
    ("Trust (0.6*Q1+0.4*Q2)", tr_acc, tr_cor, tr_th, tr_ns),
    ("Q3 alone (complexity)", q3_acc, q3_cor, q3_th, q3_ns),
    ("Combined (T+C rule)",   comb_cor/N, comb_cor, f"T={THRESH_TRUST},C={THRESH_COMPLEXITY}", N - len(noswitch_cases)),
]:
    gap = oracle_correct - (cor if isinstance(cor, int) else int(acc * N))
    print(f"  {label:<35} {pct(cor,N) if isinstance(cor,int) else pct(int(acc*N),N):>10} "
          f"{cor:>8} {str(th_v):>10} {ns:>8} {gap:>11}")
print(f"  {'Oracle':<35} {pct(oracle_correct,N):>10} {oracle_correct:>8}")

subsection("12b. Marginal value: Q1+Q3, Q2+Q3, full T+Q3")

def _sweep_2d_generic(fields, get_trust, get_cmplx, step=0.05):
    thresholds = [round(i * step, 3) for i in range(int(1 / step) + 1)]
    best = 0
    for t_th in thresholds:
        for c_th in thresholds:
            pc = 0
            for f in fields:
                ws = not (get_cmplx(f) < c_th and get_trust(f) >= t_th)
                pc += int(bool(f["grid_correct"])) if ws else int(bool(f["text_correct"]))
            best = max(best, pc)
    return best

bq1q3 = _sweep_2d_generic(all_fields, lambda f: f["q1_score"],  lambda f: f["complexity"])
bq2q3 = _sweep_2d_generic(all_fields, lambda f: f["q2_score"],  lambda f: f["complexity"])
bfull = best_correct_t10

print(f"\n  {'Combination':<35} {'best_correct':>13} {'vs Q1-only':>10} {'vs Q3-only':>10}")
print(f"  {'-'*35} {'-'*13} {'-'*10} {'-'*10}")
print(f"  {'Q1 only':<35} {q1_cor:>13} {'+0':>10}")
print(f"  {'Q3 only':<35} {q3_cor:>13} {'':>10} {'+0':>10}")
print(f"  {'Q1 + Q3 (no Q2)':<35} {bq1q3:>13} {bq1q3-q1_cor:>+10} {bq1q3-q3_cor:>+10}")
print(f"  {'Q2 + Q3 (no Q1)':<35} {bq2q3:>13} {bq2q3-q1_cor:>+10} {bq2q3-q3_cor:>+10}")
print(f"  {'Trust(Q1+Q2) + Q3 = full':<35} {bfull:>13} {bfull-q1_cor:>+10} {bfull-q3_cor:>+10}")
print(f"  {'Oracle':<35} {oracle_correct:>13} {oracle_correct-q1_cor:>+10} {oracle_correct-q3_cor:>+10}")
print(f"\n  Q2 adds {bfull-bq1q3:+d} over Q1+Q3  |  Q1 adds {bfull-bq2q3:+d} over Q2+Q3  "
      f"|  Q3 adds {bq1q3-q1_cor:+d} over Q1-only")

subsection("12c. Q1 vs Q2 agreement analysis")
q1hi_q2hi = [f for f in all_fields if f["q1_score"]>=0.7 and f["q2_score"]>=0.7]
q1hi_q2lo = [f for f in all_fields if f["q1_score"]>=0.7 and f["q2_score"]<0.7]
q1lo_q2hi = [f for f in all_fields if f["q1_score"]<0.7  and f["q2_score"]>=0.7]
q1lo_q2lo = [f for f in all_fields if f["q1_score"]<0.7  and f["q2_score"]<0.7]

def _grp_stats(grp):
    if not grp:
        return "(none)"
    t_ok = sum(1 for f in grp if f["text_correct"])
    g_ok = sum(1 for f in grp if f["grid_correct"])
    return f"n={len(grp)} txt={pct(t_ok,len(grp))} grd={pct(g_ok,len(grp))}"

print(f"\n  {'Q1/Q2':<20} {'Q2>=0.7':>20} {'Q2<0.7':>20}")
print(f"  {'-'*20} {'-'*20} {'-'*20}")
print(f"  {'Q1>=0.7':<20} {_grp_stats(q1hi_q2hi):>20} {_grp_stats(q1hi_q2lo):>20}")
print(f"  {'Q1<0.7':<20}  {_grp_stats(q1lo_q2hi):>20} {_grp_stats(q1lo_q2lo):>20}")


# ====================================================================
#   TABLE 13:  Failure types x trust/complexity
# ====================================================================
section("TABLE 13: Failure types mapped to trust & complexity scores")

print(f"\n  {'Failure Type':<38} {'n':>4} {'mean_T':>7} {'mean_C':>7} {'mean_Q1':>8} "
      f"{'mean_Q2':>8} {'switch%':>8} {'rescue%':>8}")
print(f"  {'-'*38} {'-'*4} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

for cat in FAILURE_CATEGORIES + ["unclassified", "not_classified", "classification_error"]:
    grp = [f for f in text_incorrect if f.get("failure_type") == cat]
    if not grp:
        continue
    n   = len(grp)
    mt  = sum(f["trust"]      for f in grp) / n
    mc  = sum(f["complexity"] for f in grp) / n
    mq1 = sum(f["q1_score"]   for f in grp) / n
    mq2 = sum(f["q2_score"]   for f in grp) / n
    sw  = sum(1 for f in grp if f["would_switch"])
    rsc = sum(1 for f in grp if f["grid_correct"])
    label = fc_labels.get(cat, cat)[:38]
    print(f"  {label:<38} {n:>4} {mt:>7.3f} {mc:>7.3f} {mq1:>8.3f} "
          f"{mq2:>8.3f} {pct(sw,n):>8} {pct(rsc,n):>8}")

subsection("13b. Low trust vs high trust text-wrong")
lo_t_wrong = [f for f in text_incorrect if f["trust"] < THRESH_TRUST]
hi_t_wrong = [f for f in text_incorrect if f["trust"] >= THRESH_TRUST]
print(f"  Low trust text-wrong: {len(lo_t_wrong)}  "
      f"types: {Counter(f.get('failure_type','?') for f in lo_t_wrong).most_common(3)}")
print(f"  High trust text-wrong (blind spots): {len(hi_t_wrong)}  "
      f"types: {Counter(f.get('failure_type','?') for f in hi_t_wrong).most_common(3)}")


# ====================================================================
#   TABLE 14:  C4 component predictors per failure type
# ====================================================================
section("TABLE 14: C4 component predictors per failure type")

_C4_KEYS = ["SB", "EL", "HL", "CF"]

print(f"\n  {'Failure Type':<38} {'SB':>6} {'EL':>6} {'HL':>6} {'CF':>6} {'C4':>7}")
print(f"  {'-'*38} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

all_means_c4 = {k: sum(f[k] for f in all_fields)/N for k in _C4_KEYS}
all_means_c4["complexity"] = sum(f["complexity"] for f in all_fields)/N
print(f"  {'ALL RECORDS (baseline)':<38} {all_means_c4['SB']:>6.3f} {all_means_c4['EL']:>6.3f} "
      f"{all_means_c4['HL']:>6.3f} {all_means_c4['CF']:>6.3f} {all_means_c4['complexity']:>7.3f}")

for cat in FAILURE_CATEGORIES:
    grp = [f for f in text_incorrect if f.get("failure_type") == cat]
    if not grp:
        continue
    n = len(grp)
    vals = {k: sum(f[k] for f in grp)/n for k in _C4_KEYS}
    vals["complexity"] = sum(f["complexity"] for f in grp)/n
    label = fc_labels.get(cat, cat)[:38]
    row_parts = []
    for k in _C4_KEYS:
        d = vals[k] - all_means_c4[k]
        row_parts.append(f"{vals[k]:>5.3f}+" if d > 0.05 else (f"{vals[k]:>5.3f}-" if d < -0.05 else f"{vals[k]:>6.3f}"))
    print(f"  {label:<38} {''.join(f'{p:>6}' if len(p)==6 else f' {p}' for p in row_parts)} {vals['complexity']:>7.3f}")


# ====================================================================
#   TABLE 15:  Trust x Complexity quadrant
# ====================================================================
section("TABLE 15: Trust x Complexity quadrant trade-off")

quadrants = {
    "HI_T + LO_C (confident, easy)": [f for f in all_fields if f["trust"]>=THRESH_TRUST and f["complexity"]<THRESH_COMPLEXITY],
    "HI_T + HI_C (confident, hard)": [f for f in all_fields if f["trust"]>=THRESH_TRUST and f["complexity"]>=THRESH_COMPLEXITY],
    "LO_T + LO_C (uncertain, easy)": [f for f in all_fields if f["trust"]<THRESH_TRUST  and f["complexity"]<THRESH_COMPLEXITY],
    "LO_T + HI_C (uncertain, hard)": [f for f in all_fields if f["trust"]<THRESH_TRUST  and f["complexity"]>=THRESH_COMPLEXITY],
}

print(f"\n  {'Quadrant':<35} {'n':>4} {'text%':>7} {'grid%':>7} {'sw':>5} "
      f"{'policy%':>9} {'oracle%':>9} {'Q1':>6} {'Q2':>6}")
print(f"  {'-'*35} {'-'*4} {'-'*7} {'-'*7} {'-'*5} "
      f"{'-'*9} {'-'*9} {'-'*6} {'-'*6}")

for ql, qg in quadrants.items():
    if not qg:
        continue
    n   = len(qg)
    tok = sum(1 for f in qg if f["text_correct"])
    gok = sum(1 for f in qg if f["grid_correct"])
    sw  = sum(1 for f in qg if f["would_switch"])
    pol = sum(1 for f in qg if (f["would_switch"] and f["grid_correct"]) or
              (not f["would_switch"] and f["text_correct"]))
    orc = sum(1 for f in qg if f["text_correct"] or f["grid_correct"])
    mq1 = sum(f["q1_score"] for f in qg) / n
    mq2 = sum(f["q2_score"] for f in qg) / n
    print(f"  {ql:<35} {n:>4} {pct(tok,n):>7} {pct(gok,n):>7} {sw:>5} "
          f"{pct(pol,n):>9} {pct(orc,n):>9} {mq1:>6.2f} {mq2:>6.2f}")

subsection("15b. Fine-grained trust x complexity bins")
trust_bins = [(0,.3,"T<0.3"),(0.3,.5,"0.3<=T<0.5"),(0.5,.7,"0.5<=T<0.7"),
              (0.7,.85,"0.7<=T<0.85"),(0.85,1.01,"T>=0.85")]
cmplx_bins = [(0,.3,"C<0.3"),(0.3,.45,"0.3<=C<0.45"),(0.45,.55,"0.45<=C<0.55"),(0.55,1.01,"C>=0.55")]

print(f"\n  {'Trust bin':<16} {'Cmplx bin':<16} {'n':>4} {'text%':>7} {'grid%':>7} "
      f"{'policy':>7} {'sw%':>7}")
print(f"  {'-'*16} {'-'*16} {'-'*4} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
for t_lo,t_hi,t_lbl in trust_bins:
    for c_lo,c_hi,c_lbl in cmplx_bins:
        grp = [f for f in all_fields if t_lo<=f["trust"]<t_hi and c_lo<=f["complexity"]<c_hi]
        if not grp:
            continue
        n   = len(grp)
        tok = sum(1 for f in grp if f["text_correct"])
        gok = sum(1 for f in grp if f["grid_correct"])
        pol = sum(1 for f in grp if (f["would_switch"] and f["grid_correct"]) or
                  (not f["would_switch"] and f["text_correct"]))
        sw  = sum(1 for f in grp if f["would_switch"])
        print(f"  {t_lbl:<16} {c_lbl:<16} {n:>4} {pct(tok,n):>7} {pct(gok,n):>7} "
              f"{pct(pol,n):>7} {pct(sw,n):>7}")


# ====================================================================
#   TABLE 16:  C4 component ablation
# ====================================================================
section("TABLE 16: C4 component ablation  -- are all 4 components needed?")

_C4_WEIGHTS = {"SB": 0.35, "EL": 0.10, "HL": 0.30, "CF": 0.25}

def _recompute_c4_without(fields, drop_key):
    rem   = {k: v for k, v in _C4_WEIGHTS.items() if k != drop_key}
    w_sum = sum(rem.values())
    norm  = {k: v / w_sum for k, v in rem.items()} if w_sum else {}
    return [min(1.0, max(0.0, sum(f[k] * norm.get(k, 0) for k in _C4_KEYS))) for f in fields]

def _sweep_2d(fields, cmplx_vals, step=0.05):
    thresholds = [round(i*step,3) for i in range(int(1/step)+1)]
    best_c, best_t, best_ct = 0, 0, 0
    for t_th in thresholds:
        for c_th in thresholds:
            pc = 0
            for f, cv in zip(fields, cmplx_vals):
                ws = not (cv < c_th and f["trust"] >= t_th)
                pc += int(bool(f["grid_correct"])) if ws else int(bool(f["text_correct"]))
            if pc > best_c:
                best_c, best_t, best_ct = pc, t_th, c_th
    return best_c, best_t, best_ct

orig_c4_vals = [f["complexity"] for f in all_fields]
full_best_c4, ft, fc = _sweep_2d(all_fields, orig_c4_vals)

print(f"\n  {'Configuration':<35} {'best_correct':>13} {'accuracy':>10} {'delta':>7}")
print(f"  {'-'*35} {'-'*13} {'-'*10} {'-'*7}")
print(f"  {'Full C4 (all 4 components)':<35} {full_best_c4:>13} {pct(full_best_c4,N):>10} {'base':>7}")
comp_names_c4 = {"SB":"Support Burden","EL":"Entity Load","HL":"Hard Language","CF":"Coref Difficulty"}
for drop_k in _C4_KEYS:
    nv = _recompute_c4_without(all_fields, drop_k)
    bc, _, _ = _sweep_2d(all_fields, nv)
    delta = bc - full_best_c4
    marker = "  <- HELPS (redundant?)" if delta > 0 else ("  <- HURTS" if delta < 0 else "  <- no change")
    print(f"  {'Drop ' + comp_names_c4[drop_k]:<35} {bc:>13} {pct(bc,N):>10} {delta:>+7}{marker}")
print(f"  {'Oracle':<35} {oracle_correct:>13} {pct(oracle_correct,N):>10}")

subsection("16b. Component correlation with correct switch decisions")
pol_correct_l = [f for f in all_fields if (f["would_switch"] and f["grid_correct"]) or
                 (not f["would_switch"] and f["text_correct"])]
pol_wrong_l   = [f for f in all_fields if f not in pol_correct_l]
print(f"\n  {'Component':<20} {'mean(correct)':>14} {'mean(wrong)':>14} {'delta':>8}")
print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*8}")
for k in _C4_KEYS:
    mc = sum(f[k] for f in pol_correct_l) / max(1, len(pol_correct_l))
    mw = sum(f[k] for f in pol_wrong_l)   / max(1, len(pol_wrong_l))
    print(f"  {comp_names_c4[k]:<20} {mc:>14.4f} {mw:>14.4f} {mw-mc:>+8.4f}")


# ====================================================================
#   TABLE 17:  Deep sub-component signal forensics
# ====================================================================
section("TABLE 17: Deep sub-component signal forensics")

_SUB_KEYS  = ["q1_s", "q1_c", "para_stab", "flip_score", "SB", "EL", "HL", "CF"]
_SUB_NAMES = {
    "q1_s":      "Q1_S (support)",
    "q1_c":      "Q1_C (ablation)",
    "para_stab": "Para stability",
    "flip_score":"Flip score",
    "SB":  "Support Burden",
    "EL":  "Entity Load",
    "HL":  "Hard Language",
    "CF":  "Coref Difficulty",
}

_tfgr = [f for f in all_fields if not f["text_correct"] and f["grid_correct"]]
_tfgf = [f for f in all_fields if not f["text_correct"] and not f["grid_correct"]]
_tok  = [f for f in all_fields if f["text_correct"]]
_tall = [f for f in all_fields if not f["text_correct"]]

subsection("17a. Sub-component fingerprint: rescued vs failed vs correct")
print(f"  RESCUED={len(_tfgr)}  FAILED={len(_tfgf)}  CORRECT={len(_tok)}\n")
print(f"  {'Sub-component':<22} {'RESCUED':>10} {'FAILED':>10} {'CORRECT':>10} {'R-F delta':>10}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for k in _SUB_KEYS:
    mr = sum(f[k] for f in _tfgr) / max(1, len(_tfgr))
    mf = sum(f[k] for f in _tfgf) / max(1, len(_tfgf))
    mc = sum(f[k] for f in _tok)  / max(1, len(_tok))
    print(f"  {_SUB_NAMES[k]:<22} {mr:>10.4f} {mf:>10.4f} {mc:>10.4f} {mr-mf:>+10.4f}")

subsection("17b. Conditional accuracy: high-vs-low split per sub-component")
print(f"\n  {'Sub-component':<22} {'split':>6} {'n_lo':>5} {'t_lo':>7} {'g_lo':>7} {'p_lo':>7}"
      f" | {'n_hi':>5} {'t_hi':>7} {'g_hi':>7} {'p_hi':>7}")
print(f"  {'-'*22} {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7}"
      f" | {'-'*5} {'-'*7} {'-'*7} {'-'*7}")
for k in _SUB_KEYS:
    vs = sorted(f[k] for f in all_fields)
    med = vs[N//2]
    lo  = [f for f in all_fields if f[k] <  med] or [f for f in all_fields if f[k] <= med]
    hi  = [f for f in all_fields if f[k] >= med] or [f for f in all_fields if f[k] >  med]
    def _acc(g):
        n=len(g); t=sum(1 for f in g if f["text_correct"]); gk=sum(1 for f in g if f["grid_correct"])
        p=sum(1 for f in g if (f["would_switch"] and f["grid_correct"]) or (not f["would_switch"] and f["text_correct"]))
        return n,t,gk,p
    nl,tl,gl,pl = _acc(lo)
    nh,th,gh,ph = _acc(hi)
    print(f"  {_SUB_NAMES[k]:<22} {med:>6.3f} {nl:>5} {pct(tl,nl):>7} {pct(gl,nl):>7} {pct(pl,nl):>7}"
          f" | {nh:>5} {pct(th,nh):>7} {pct(gh,nh):>7} {pct(ph,nh):>7}")

subsection("17c. Sub-component profile per GPT failure type")
_ti_means = {k: sum(f[k] for f in text_incorrect)/max(1,len(text_incorrect)) for k in _SUB_KEYS}
hdr_17c   = f"  {'Failure type':<35} {'n':>4} {'resc%':>6} " + " ".join(f"{_SUB_NAMES[k][:7]:>8}" for k in _SUB_KEYS)
print(hdr_17c)
print("  " + "-" * len(hdr_17c))
print(f"  {'ALL text-incorrect (baseline)':<35} {len(text_incorrect):>4} "
      f"{pct(sum(1 for f in text_incorrect if f['grid_correct']),len(text_incorrect)):>6} " +
      " ".join(f"{_ti_means[k]:>8.3f}" for k in _SUB_KEYS))
for cat in FAILURE_CATEGORIES + ["not_classified"]:
    grp = [f for f in text_incorrect if f.get("failure_type") == cat]
    if not grp:
        continue
    n   = len(grp)
    rsc = sum(1 for f in grp if f["grid_correct"])
    vals = " ".join(f"{sum(f[k] for f in grp)/n:>8.3f}" for k in _SUB_KEYS)
    label = fc_labels.get(cat, cat)[:35]
    print(f"  {label:<35} {n:>4} {pct(rsc,n):>6} {vals}")

subsection("17d. Effect size (Cohen's d) per sub-component per failure type")
def _cohens_d(a, b):
    n1,n2 = len(a),len(b)
    if n1<2 or n2<2: return 0.0
    m1,m2 = sum(a)/n1, sum(b)/n2
    v1 = sum((x-m1)**2 for x in a)/(n1-1)
    v2 = sum((x-m2)**2 for x in b)/(n2-1)
    sd = math.sqrt(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2))
    return (m1-m2)/sd if sd>1e-9 else 0.0

for cat in FAILURE_CATEGORIES:
    grp = [f for f in text_incorrect if f.get("failure_type") == cat]
    rest = [f for f in text_incorrect if f.get("failure_type") != cat]
    if len(grp) < 3: continue
    print(f"\n  >> {fc_labels.get(cat,cat)} (n={len(grp)})")
    effs = sorted([(k, _cohens_d([f[k] for f in grp],[f[k] for f in rest])) for k in _SUB_KEYS],
                  key=lambda x:-abs(x[1]))
    for k, d in effs:
        mag = "LARGE" if abs(d)>0.8 else ("medium" if abs(d)>0.5 else ("small" if abs(d)>0.2 else "~"))
        print(f"    {_SUB_NAMES[k]:<22} d={d:>+6.3f}  ({'^ elevated' if d>0 else 'v depressed'})  {mag}")


# ====================================================================
#   TABLE 18:  Method complementarity
# ====================================================================
section("TABLE 18: Method complementarity & answer confusion matrix")

ab_right = [f for f in all_fields if f["text_correct"]     and f["grid_correct"]]
ab_wrong = [f for f in all_fields if not f["text_correct"] and not f["grid_correct"]]
text_only_right = [f for f in all_fields if f["text_correct"] and not f["grid_correct"]]
grid_only_right = [f for f in all_fields if not f["text_correct"] and f["grid_correct"]]

print(f"\n  {'':20} {'Grid RIGHT':>15} {'Grid WRONG':>15} {'total':>8}")
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*8}")
print(f"  {'Text RIGHT':<20} {len(ab_right):>15} {len(text_only_right):>15} "
      f"{len(ab_right)+len(text_only_right):>8}")
print(f"  {'Text WRONG':<20} {len(grid_only_right):>15} {len(ab_wrong):>15} "
      f"{len(grid_only_right)+len(ab_wrong):>8}")
print(f"  {'total':<20} {len(ab_right)+len(grid_only_right):>15} "
      f"{len(text_only_right)+len(ab_wrong):>15} {N:>8}")

print(f"\n  Complementarity: {(len(text_only_right)+len(grid_only_right))/N:.1%}")
print(f"  Both wrong (unreachable): {len(ab_wrong)}/{N} = {len(ab_wrong)/N:.1%}")
print(f"  Grid rescues {len(grid_only_right)} text-fails; text rescues {len(text_only_right)} grid-fails")

subsection("18b. Sub-component profile when methods disagree")
print(f"\n  {'Sub-component':<22} {'grid_only_right':>16} {'text_only_right':>16} {'delta':>8}")
print(f"  {'-'*22} {'-'*16} {'-'*16} {'-'*8}")
for k in _SUB_KEYS:
    mg = sum(f[k] for f in grid_only_right) / max(1, len(grid_only_right))
    mt = sum(f[k] for f in text_only_right) / max(1, len(text_only_right))
    print(f"  {_SUB_NAMES[k]:<22} {mg:>16.4f} {mt:>16.4f} {mg-mt:>+8.4f}")

subsection("18c. Answer confusion matrix (text prediction vs ground truth)")
_confusion = Counter()
for f in text_incorrect:
    pred = (f.get("text_pred") or "").strip().lower()
    gt_r = f["gt_list"][0] if f["gt_list"] else ""
    if pred and gt_r:
        _confusion[(gt_r, pred)] += 1

if _confusion:
    print(f"\n  Top 15 confusions (GT -> Predicted):")
    print(f"  {'GT':<18} {'Predicted':<18} {'count':>6} {'rescue%':>10}")
    print(f"  {'-'*18} {'-'*18} {'-'*6} {'-'*10}")
    for (gt_r, pred), cnt in _confusion.most_common(15):
        rsc = sum(1 for f in text_incorrect
                  if (f["gt_list"][0] if f["gt_list"] else "")==gt_r
                  and f["text_pred"]==pred
                  and f["grid_correct"])
        print(f"  {gt_r:<18} {pred:<18} {cnt:>6} {pct(rsc,cnt):>10}")

    # YN-specific: yes<->no confusion
    yn_yes_to_no = sum(1 for f in text_incorrect
                       if f["q_type"]=="YN" and f["text_pred"]=="no"  and "yes" in f["gt_list"])
    yn_no_to_yes = sum(1 for f in text_incorrect
                       if f["q_type"]=="YN" and f["text_pred"]=="yes" and "no"  in f["gt_list"])
    print(f"\n  YN confusion: predicted 'no' when gold='yes': {yn_yes_to_no}")
    print(f"  YN confusion: predicted 'yes' when gold='no':  {yn_no_to_yes}")


# ====================================================================
#   TABLE 19:  "What if perfect [X]?" ceiling
# ====================================================================
section("TABLE 19: 'What if perfect [X]?' -- improvement ceiling")

_baseline_pol = sum(1 for f in all_fields
                    if (f["would_switch"] and f["grid_correct"]) or
                    (not f["would_switch"] and f["text_correct"]))

def _eval_oracle_trust(fields, ot_vals):
    best = 0
    for c_th in [round(i*.05,3) for i in range(21)]:
        for t_th in [round(i*.05,3) for i in range(21)]:
            pc=0
            for f,ot in zip(fields,ot_vals):
                ws = not (f["complexity"]<c_th and ot>=t_th)
                pc += int(bool(f["grid_correct"])) if ws else int(bool(f["text_correct"]))
            best=max(best,pc)
    return best

def _eval_oracle_cmplx(fields, oc_vals):
    best=0
    for t_th in [round(i*.05,3) for i in range(21)]:
        for c_th in [round(i*.05,3) for i in range(21)]:
            pc=0
            for f,oc in zip(fields,oc_vals):
                ws = not (oc<c_th and f["trust"]>=t_th)
                pc += int(bool(f["grid_correct"])) if ws else int(bool(f["text_correct"]))
            best=max(best,pc)
    return best

perf_trust_val = _eval_oracle_trust(all_fields, [1.0 if f["text_correct"] else 0.0 for f in all_fields])
perf_cmplx_val = _eval_oracle_cmplx(all_fields, [1.0 if (not f["text_correct"] and f["grid_correct"]) else 0.0 for f in all_fields])
perf_q1_val    = _eval_oracle_trust(all_fields, [min(1.,0.6*(1. if f["text_correct"] else 0.)+0.4*f["q2_score"]) for f in all_fields])
perf_q2_val    = _eval_oracle_trust(all_fields, [min(1.,0.6*f["q1_score"]+0.4*(1. if f["text_correct"] else 0.)) for f in all_fields])

print(f"\n  {'Scenario':<45} {'accuracy':>10} {'correct':>8} {'delta':>7}")
print(f"  {'-'*45} {'-'*10} {'-'*8} {'-'*7}")
for lbl, val in [
    ("Current policy",                    _baseline_pol),
    ("Perfect Q1 (real Q2 + C4)",         perf_q1_val),
    ("Perfect Q2 (real Q1 + C4)",         perf_q2_val),
    ("Perfect trust (oracle Q1+Q2)",      perf_trust_val),
    ("Perfect C4 (oracle complexity)",    perf_cmplx_val),
    ("Perfect everything (oracle switch)",oracle_correct),
]:
    print(f"  {lbl:<45} {pct(val,N):>10} {val:>8} {val-_baseline_pol:>+7}")

_imps = sorted([("Q1",perf_q1_val-_baseline_pol),("Q2",perf_q2_val-_baseline_pol),
                ("Trust",perf_trust_val-_baseline_pol),("C4",perf_cmplx_val-_baseline_pol)],
               key=lambda x:-x[1])
print(f"\n  Biggest improvement potential: {_imps[0][0]} (+{_imps[0][1]})  |  "
      f"Second: {_imps[1][0]} (+{_imps[1][1]})")


# ====================================================================
#   TABLE 20:  Multi-label gold analysis  (FR records)
# ====================================================================
section("TABLE 20: Multi-label gold analysis (FR records with >1 gold answer)")

ml_records = [f for f in all_fields if len(f["gt_list"]) > 1]
sl_records = [f for f in all_fields if len(f["gt_list"]) == 1]
print(f"\n  Single-label gold: {len(sl_records)}")
print(f"  Multi-label gold:  {len(ml_records)}")

if ml_records:
    ml_text_ok = sum(1 for f in ml_records if f["text_correct"])
    ml_grid_ok = sum(1 for f in ml_records if f["grid_correct"])
    ml_rels_ok = sum(1 for f in ml_records if f["text_rels_correct"])
    print(f"\n  Multi-label records accuracy:")
    print(f"    Baseline text:       {ml_text_ok}/{len(ml_records)} = {pct(ml_text_ok,len(ml_records))}")
    print(f"    Text w/ rels:        {ml_rels_ok}/{len(ml_records)} = {pct(ml_rels_ok,len(ml_records))}")
    print(f"    Grid ({GRID_MODE}):  {ml_grid_ok}/{len(ml_records)} = {pct(ml_grid_ok,len(ml_records))}")

    # Gold set size distribution
    size_ctr = Counter(len(f["gt_list"]) for f in ml_records)
    print(f"\n  Gold set size distribution: {dict(sorted(size_ctr.items()))}")

    # Most common gold sets
    gold_set_ctr = Counter(f["gt"] for f in ml_records)
    print(f"\n  Top 10 most common multi-label gold sets:")
    for gs, cnt in gold_set_ctr.most_common(10):
        ok = sum(1 for f in ml_records if f["gt"]==gs and f["text_correct"])
        print(f"    {gs:<35} n={cnt}  text_ok={ok}")

    # Compare single vs multi label accuracy
    sl_text_ok = sum(1 for f in sl_records if f["text_correct"])
    sl_grid_ok = sum(1 for f in sl_records if f["grid_correct"])
    print(f"\n  Single-label accuracy: text={pct(sl_text_ok,len(sl_records))}  "
          f"grid={pct(sl_grid_ok,len(sl_records))}")
    print(f"  Multi-label accuracy:  text={pct(ml_text_ok,len(ml_records))}  "
          f"grid={pct(ml_grid_ok,len(ml_records))}")
    print(f"  -> Multi-label is {'EASIER' if ml_text_ok/max(1,len(ml_records))>sl_text_ok/max(1,len(sl_records)) else 'HARDER'} "
          f"for text and {'EASIER' if ml_grid_ok/max(1,len(ml_records))>sl_grid_ok/max(1,len(sl_records)) else 'HARDER'} "
          f"for grid")


# ====================================================================
#   TABLE 21:  Switching ROC & Precision-Recall
# ====================================================================
section("TABLE 21: Switching as binary classification -- ROC & Precision-Recall")

_labels_21 = [0 if f["text_correct"] else 1 for f in all_fields]
_n_pos_21  = sum(_labels_21)

def _compute_auc(scores, labels, higher_pos=True):
    paired = sorted(zip(scores, labels), key=lambda x: -x[0] if higher_pos else x[0])
    tp=fp=0; points=[(0.,0.)]
    n_p = sum(labels); n_n = len(labels)-n_p
    if n_p==0 or n_n==0: return 0.5, points
    for score, label in paired:
        if label==1: tp+=1
        else:        fp+=1
        points.append((fp/n_n, tp/n_p))
    auc=sum((points[i][0]-points[i-1][0])*(points[i][1]+points[i-1][1])/2
            for i in range(1,len(points)))
    return auc, points

auc_trust, _   = _compute_auc([-f["trust"]      for f in all_fields], _labels_21)
auc_cmplx, _   = _compute_auc([ f["complexity"] for f in all_fields], _labels_21)
auc_comb,  _   = _compute_auc([ f["complexity"] - f["trust"] for f in all_fields], _labels_21)

print(f"\n  {'Score function':<35} {'AUC-ROC':>10}")
print(f"  {'-'*35} {'-'*10}")
print(f"  {'-Trust:':<35} {auc_trust:>10.4f}")
print(f"  {'Complexity:':<35} {auc_cmplx:>10.4f}")
print(f"  {'C - T combined:':<35} {auc_comb:>10.4f}")
for k in _SUB_KEYS:
    scores = [-all_fields[i][k] for i in range(N)] if k in ["q1_s","q1_c","para_stab","flip_score"] else [all_fields[i][k] for i in range(N)]
    auc_k, _ = _compute_auc(scores, _labels_21)
    print(f"  {_SUB_NAMES[k]+':':<35} {auc_k:>10.4f}")

subsection("21b. Precision-Recall at current thresholds")
_tp = sum(1 for f in all_fields if f["would_switch"] and not f["text_correct"])
_fp = sum(1 for f in all_fields if f["would_switch"] and f["text_correct"])
_fn = sum(1 for f in all_fields if not f["would_switch"] and not f["text_correct"])
_tn = sum(1 for f in all_fields if not f["would_switch"] and f["text_correct"])
prec = _tp/max(1,_tp+_fp); rec = _tp/max(1,_tp+_fn)
f1   = 2*prec*rec/max(1e-9,prec+rec)
print(f"\n  TP={_tp}  FP={_fp}  FN={_fn}  TN={_tn}")
print(f"  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")
useful = sum(1 for f in all_fields if f["would_switch"] and not f["text_correct"] and f["grid_correct"])
print(f"  'Useful' precision (switch + text_wrong + grid_right): {useful}/{_tp+_fp} = {pct(useful,_tp+_fp)}")


# ====================================================================
#   TABLE 22:  Difficulty ordering concordance (Kendall tau)
# ====================================================================
section("TABLE 22: Difficulty ordering concordance (Kendall tau)")

for f in all_fields:
    tc, gc = bool(f["text_correct"]), bool(f["grid_correct"])
    f["_difficulty"] = 0 if (tc and gc) else (1 if (tc or gc) else 2)

diff_ctr = Counter(f["_difficulty"] for f in all_fields)
print(f"\n  easy(0)={diff_ctr[0]}  medium(1)={diff_ctr[1]}  hard(2)={diff_ctr[2]}")

def _kendall_tau(x_vals, y_vals, sample=500):
    n=len(x_vals); c=d=tx=ty=0
    for i in range(n):
        for j in range(i+1, min(i+sample, n)):
            dx=x_vals[i]-x_vals[j]; dy=y_vals[i]-y_vals[j]
            if dx*dy>0: c+=1
            elif dx*dy<0: d+=1
            else:
                if dx==0: tx+=1
                if dy==0: ty+=1
    denom=math.sqrt(max(1,(c+d+tx)*(c+d+ty)))
    return (c-d)/denom if denom>0 else 0.

_diff_vals = [f["_difficulty"] for f in all_fields]
print(f"\n  {'Metric':<25} {'Kendall tau':>12} {'expected':>12} {'strength':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
_tau_metrics = [
    ("trust",      [f["trust"]      for f in all_fields], "negative"),
    ("complexity", [f["complexity"] for f in all_fields], "positive"),
    ("q1_score",   [f["q1_score"]   for f in all_fields], "negative"),
    ("q2_score",   [f["q2_score"]   for f in all_fields], "negative"),
]
for k in _SUB_KEYS:
    exp = "negative" if k in ["q1_s","q1_c","para_stab","flip_score"] else "positive"
    _tau_metrics.append((k, [f[k] for f in all_fields], exp))
for name, vals, exp in _tau_metrics:
    tau = _kendall_tau(vals, _diff_vals)
    ok  = (exp=="negative" and tau<0) or (exp=="positive" and tau>0)
    str_= "strong" if abs(tau)>0.15 else ("moderate" if abs(tau)>0.08 else "weak")
    dn  = _SUB_NAMES.get(name, name)
    print(f"  {dn:<25} {tau:>+12.4f} {'expected '+exp:>12} {str_ + (' OK' if ok else ' !INVERTED'):>10}")


# ====================================================================
#   TABLE 23:  Switch regret & happy analysis
# ====================================================================
section("TABLE 23: Switch regret & happy  -- sub-signal forensics")

_happy_sw   = [f for f in all_fields if f["would_switch"] and f["grid_correct"] and not f["text_correct"]]
_happy_stay = [f for f in all_fields if not f["would_switch"] and f["text_correct"]]
_regret_sw  = [f for f in all_fields if f["would_switch"] and f["text_correct"] and not f["grid_correct"]]
_regret_sta = [f for f in all_fields if not f["would_switch"] and not f["text_correct"] and f["grid_correct"]]
_all_happy  = _happy_sw + _happy_stay
_all_regret = _regret_sw + _regret_sta

print(f"\n  HAPPY_SWITCH  (textX, switched, gridV): {len(_happy_sw)}")
print(f"  HAPPY_STAY    (textV, stayed):           {len(_happy_stay)}")
print(f"  REGRET_SWITCH (textV, switched, gridX):  {len(_regret_sw)}")
print(f"  REGRET_STAY   (textX, stayed, gridV):    {len(_regret_sta)}")
print(f"  TOTAL HAPPY={len(_all_happy)}  REGRET={len(_all_regret)}")

subsection("23a. Sub-signal fingerprint: REGRET vs HAPPY")
print(f"\n  {'Sub-signal':<22} {'HAPPY':>10} {'REGRET':>10} {'delta':>10} {'blame':>20}")
print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*20}")
regret_effs = []
for k in _SUB_KEYS:
    mh = sum(f[k] for f in _all_happy)  / max(1,len(_all_happy))
    mr = sum(f[k] for f in _all_regret) / max(1,len(_all_regret))
    d  = mr - mh
    regret_effs.append((k, d, abs(d), mh, mr))
regret_effs.sort(key=lambda x: -x[2])
for k, d, ad, mh, mr in regret_effs:
    if k in ["q1_s","q1_c","para_stab","flip_score"]:
        blame = "trust TOO HIGH" if d>0.03 else ("trust TOO LOW" if d<-0.03 else "~neutral")
    else:
        blame = "complexity TOO HIGH" if d>0.03 else ("complexity TOO LOW" if d<-0.03 else "~neutral")
    print(f"  {_SUB_NAMES[k]:<22} {mh:>10.4f} {mr:>10.4f} {d:>+10.4f} {blame:>20}")

subsection("23b. Switch-regret vs Stay-regret breakdown")

for label_, grp_, compare_, title_ in [
    ("SWITCH-REGRET", _regret_sw,  _happy_stay, "textV, switched, gridX"),
    ("STAY-REGRET",   _regret_sta, _happy_sw,   "textX, stayed, gridV"),
]:
    print(f"\n  {label_} n={len(grp_)} ({title_})")
    if not grp_:
        print(f"    (none)")
        continue
    mt_r = sum(f["trust"]      for f in grp_) / len(grp_)
    mc_r = sum(f["complexity"] for f in grp_) / len(grp_)
    print(f"    mean trust={mt_r:.3f}  mean complexity={mc_r:.3f}")
    if label_ == "SWITCH-REGRET" and mt_r < THRESH_TRUST:
        print(f"    -> Trust was LOW ({mt_r:.3f}) = Q1/Q2 mistakenly doubted the correct answer")
    if label_ == "STAY-REGRET" and mt_r >= THRESH_TRUST:
        print(f"    -> Trust was HIGH ({mt_r:.3f}) = Q1/Q2 fooled by wrong answer (confident but wrong)")

    print(f"    {'Sub-signal':<22} {label_[:12]:>14} {'compare':>14} {'delta':>8}")
    print(f"    {'-'*22} {'-'*14} {'-'*14} {'-'*8}")
    for k in _SUB_KEYS:
        mg  = sum(f[k] for f in grp_)     / len(grp_)
        mc  = sum(f[k] for f in compare_) / max(1, len(compare_))
        print(f"    {_SUB_NAMES[k]:<22} {mg:>14.4f} {mc:>14.4f} {mg-mc:>+8.4f}")

subsection("23c. GPT failure types in regret cases")
_regret_stay_ids = set((f["dataset_id"],f["idx"]) for f in _regret_sta)
_regret_sw_ids   = set((f["dataset_id"],f["idx"]) for f in _regret_sw)
sta_r_ti = [f for f in text_incorrect if (f["dataset_id"],f["idx"]) in _regret_stay_ids]
sw_r_ti  = [f for f in text_incorrect if (f["dataset_id"],f["idx"]) in _regret_sw_ids]

if sta_r_ti:
    fc_stay = Counter(f.get("failure_type","?") for f in sta_r_ti)
    print(f"\n  STAY-REGRET failure types (text wrong, should have switched but didn't):")
    for cat, cnt in fc_stay.most_common():
        print(f"    {fc_labels.get(cat,cat):<42} {cnt}")


# ====================================================================
#   LOSS CASES -> JSON EXPORT
# ====================================================================
section("LOSS CASES EXPORT")

ns_text_bad_grid_ok = [f for f in noswitch_cases if not f["text_correct"] and f["grid_correct"]]
hurt_cases = [f for f in switch_cases if f["text_correct"] and not f["grid_correct"]]
both_wrong_sw = [f for f in switch_cases if not f["text_correct"] and not f["grid_correct"]]
both_wrong_ns = [f for f in noswitch_cases if not f["text_correct"] and not f["grid_correct"]]

loss_cases = []
for f in ns_text_bad_grid_ok:
    loss_cases.append(_build_case_payload(f,"missed_rescue",
        "Text WRONG, didn't switch, grid was RIGHT"))
for f in hurt_cases:
    loss_cases.append(_build_case_payload(f,"bad_switch",
        "Text RIGHT, switched, grid was WRONG"))
for f in both_wrong_sw:
    loss_cases.append(_build_case_payload(f,"both_wrong_switched","Both wrong, switched"))
for f in both_wrong_ns:
    loss_cases.append(_build_case_payload(f,"both_wrong_stayed","Both wrong, stayed"))

regret_cases = []
for f in _regret_sw:
    regret_cases.append(_build_case_payload(f, "switch_regret",
        "Text was RIGHT, policy switched, and grid was WRONG"))
for f in _regret_sta:
    regret_cases.append(_build_case_payload(f, "stay_regret",
        "Text was WRONG, policy stayed, and grid was RIGHT"))

loss_cases.sort(key=lambda x: (x["case_type"], x["metadata"]["q_type"], x["metadata"]["idx"]))
regret_cases.sort(key=lambda x: (x["case_type"], x["metadata"]["q_type"], x["metadata"]["idx"]))

loss_out_dir = fpath.parent if fpath.parent.is_dir() else Path.cwd()
loss_out = loss_out_dir / LOSS_CASES_OUTPUT
regret_out = loss_out_dir / REGRET_CASES_OUTPUT
with open(loss_out,"w",encoding="utf-8") as fp:
    json.dump(loss_cases,fp,indent=2,ensure_ascii=False)
with open(regret_out,"w",encoding="utf-8") as fp:
    json.dump(regret_cases,fp,indent=2,ensure_ascii=False)

loss_type_ctr = Counter(c["case_type"] for c in loss_cases)
regret_type_ctr = Counter(c["case_type"] for c in regret_cases)
print(f"\n  Total loss cases: {len(loss_cases)}")
print(f"  Written to: {loss_out}\n")
print(f"  Total regret cases: {len(regret_cases)}")
print(f"  Written to: {regret_out}\n")
_ld = {"missed_rescue":"Text wrong, didn't switch -> grid could have saved",
       "bad_switch":"Text right, switched -> grid was wrong  (broke it)",
       "both_wrong_switched":"Both wrong, switched (no rescue possible)",
       "both_wrong_stayed":"Both wrong, stayed (no rescue possible)"}
for ct in ["missed_rescue","bad_switch","both_wrong_switched","both_wrong_stayed"]:
    print(f"  {ct:<28} {loss_type_ctr.get(ct,0):>5}  {_ld.get(ct,'')}")
actionable = loss_type_ctr.get("missed_rescue",0) + loss_type_ctr.get("bad_switch",0)
unavoid    = loss_type_ctr.get("both_wrong_switched",0) + loss_type_ctr.get("both_wrong_stayed",0)
print(f"\n  Actionable: {actionable}  |  Unavoidable: {unavoid}")
print(f"  Switch regret: {regret_type_ctr.get('switch_regret',0)}  |  Stay regret: {regret_type_ctr.get('stay_regret',0)}")


# ====================================================================
#   SUMMARY
# ====================================================================
section("SUMMARY")

policy_correct = sum(
    1 for f in all_fields
    if (f["would_switch"] and f["grid_correct"])
    or (not f["would_switch"] and f["text_correct"])
)

print(f"\n  Overall accuracy comparison  (grid_mode={GRID_MODE}):")
print(f"    Baseline text-only:       {text_ok:>4}/{N}  = {pct(text_ok,N)}")
print(f"    Text with relations:      {rels_ok:>4}/{N}  = {pct(rels_ok,N)}")
print(f"    Full grid YN:             {fg_ok:>4}/{N}  = {pct(fg_ok,N)}")
print(f"    Pruned grid YN:           {pg_ok:>4}/{N}  = {pct(pg_ok,N)}")
print(f"    Best-of-two grids:        {best_ok:>4}/{N}  = {pct(best_ok,N)}")
print(f"    Switch policy:            {policy_correct:>4}/{N}  = {pct(policy_correct,N)}")
print(f"    Oracle (best of two):     {oracle_correct:>4}/{N}  = {pct(oracle_correct,N)}")
print(f"\n  Switch policy gain over text-only: {policy_correct - text_ok:+d}")
print(f"  Switch policy gain over grid-only: {policy_correct - grid_ok:+d}")
print(f"  Gap to oracle:                     {oracle_correct - policy_correct:+d}")
print(f"\n  Question type breakdown:")
for qt in sorted(qtype_groups.keys()):
    grp = qtype_groups[qt]
    pol = sum(1 for f in grp if (f["would_switch"] and f["grid_correct"]) or
              (not f["would_switch"] and f["text_correct"]))
    print(f"    {qt}: policy={pct(pol,len(grp))}  text={pct(sum(1 for f in grp if f['text_correct']),len(grp))}  "
          f"grid={pct(sum(1 for f in grp if f['grid_correct']),len(grp))}")
print(f"\n  Thresholds: trust >= {THRESH_TRUST},  complexity < {THRESH_COMPLEXITY}")
print(f"  (Edit THRESH_TRUST & THRESH_COMPLEXITY at top to re-run with different thresholds)")
print()
