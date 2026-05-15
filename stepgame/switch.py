"""
switch_stepgame_fast.py
=======================
Drop-in faster replacement for switch_stepgame.py.

Three speed-ups applied:
  1. Cached Ollama / GPT clients   – avoids re-creating the LLM object every call.
  2. ThreadPool parallelism         – independent LLM calls run concurrently.
  3. Short-circuit logic            – skips Q2 (4 Ollama calls) when the decision
     is already determined from C1+C3 alone.

  Typical worst-case: 3 parallel rounds instead of 10 sequential LLM calls.
  Best-case (short-circuit): 2 rounds + 0 Q2 calls.

  Public API is identical to the original: run_dataset_one_pretty_json(), etc.
"""

import os
import json
import re
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from utils.timing import _now, _secs, _timed_call
from utils.parsing import _clip01, extract_json_object, split_sentences, _norm_for_match
from utils.llm_clients import (
    call_gpt_nlp as call_gpt_5_1,
    call_gpt_reasoning as _call_gpt_reasoning,
    empty_token_info as _empty_token_info,
)
from utils.switch_helpers import (
    _submit,
    _bool_to_float,
    _safe_mean,
    relations_to_sentences_verbatim,
    match_support_sentences_to_story,
    build_ablated_story_remove_sentences,
)
from stepgame.switch_prompts import (
    REL_OPTIONS_NO_DK, REL_OPTIONS_WITH_DK, REL_OPTIONS,
    INVERSE_REL, DIAGONAL_SET, SEMANTICS_BLOCK,
    prompt_hard_language_scores,
    prompt_q1_baseline_and_support_sentences,
    prompt_answer_only_allow_dontknow,
    prompt_q2_answer_from_text,
    prompt_q2_unified_relations_and_variants,
    flip_question_once,
    build_symbolic_QA_prompt,
)

# ============================================================
# 0-A) LLM CALL WRAPPERS
# ============================================================
DEFAULT_SEED = 42


def call_ollama_llama(
    prompt: str,
    model: str = "gpt-5.1",
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
) -> Tuple[str, Dict[str, Any]]:
    """Returns (response_text, token_info_dict).

    Calls GPT-5.1 via OpenAI Responses API with medium reasoning effort.
    """
    return _call_gpt_reasoning(prompt, model="gpt-5.1", effort="medium")


# ============================================================
# 0-B) PARALLEL EXECUTION HELPER (now from utils/switch_helpers)
# ============================================================


# ============================================================
# 2) NORMALIZATION / PARSING  (unchanged)
# ============================================================
def normalize_relation(rel: str) -> str:
    rel = (rel or "").strip().lower()
    rel = rel.replace("_", " ").replace("-", " ")
    rel = re.sub(r"\s+", " ", rel).strip()
    if rel in ["dontknow", "don't know", "dont-know", "don't-know"]:
        rel = "dont know"
    rel = rel.replace("upper left", "upper-left").replace("upper right", "upper-right")
    rel = rel.replace("lower left", "lower-left").replace("lower right", "lower-right")
    return rel

def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
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
    items = []
    ok = True
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
    chunks = re.split(r"\n\s*\n", text)
    items = []
    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        items.append(json.loads(ch))
    return items


# ============================================================
# 3) SIMPLE HELPERS  (unchanged)
# ============================================================
def _extract_selected_option_any(raw_or_dict: Any) -> str:
    if isinstance(raw_or_dict, dict):
        for k in ["selected_option", "answer"]:
            v = raw_or_dict.get(k, None)
            if isinstance(v, str) and v.strip():
                return normalize_relation(v)
        return ""
    if isinstance(raw_or_dict, str) and raw_or_dict.strip():
        obj = extract_json_object(raw_or_dict)
        if isinstance(obj, dict):
            for k in ["selected_option", "answer"]:
                v = obj.get(k, None)
                if isinstance(v, str) and v.strip():
                    return normalize_relation(v)
    return ""

def _compute_lm_correct(selected: str, gt: str) -> Optional[bool]:
    gt_n = normalize_relation(gt)
    sel_n = normalize_relation(selected)
    if not gt_n:
        return None
    if sel_n not in REL_OPTIONS_NO_DK:
        return False
    return bool(sel_n == gt_n)

def find_entities(text: str) -> List[str]:
    return sorted(set(re.findall(r"\b[A-Z]\b", text or "")))

def relations_list_to_text(relations: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for r in relations or []:
        h = str(r.get("head", "")).strip()
        t = str(r.get("tail", "")).strip()
        rel = normalize_relation(str(r.get("relation", "")).strip())
        if h and t and rel:
            lines.append(f"{h} is {rel} of {t}")
    return lines


# relations_to_sentences_verbatim imported from utils/switch_helpers
# _bool_to_float imported from utils/switch_helpers


# ============================================================
# 3B) Hard-language scoring wrapper (prompt builder is in switch_prompts.py)
# ============================================================
def extract_hard_language_scores(text: str, gpt_model: str = "gpt-5-mini") -> Dict[str, Any]:
    prompt = prompt_hard_language_scores(text)
    raw, dt = _timed_call(call_gpt_5_1, prompt, model=gpt_model)
    parsed = extract_json_object(raw)
    items = parsed.get("items", [])
    if not isinstance(items, list):
        items = []
    clean = []
    for it in items:
        if not isinstance(it, dict):
            continue
        span = str(it.get("span", "")).strip()
        typ = str(it.get("type", "")).strip().lower()
        try:
            diff = float(it.get("difficulty", 0.0))
        except Exception:
            diff = 0.0
        if not span:
            continue
        if typ not in {"clock", "direction", "ambiguity"}:
            typ = "direction"
        clean.append({"span": span, "type": typ, "difficulty": _clip01(diff)})
    return {"prompt": prompt, "raw": raw, "parsed": parsed, "items": clean, "timing_sec": dt}


def q2_unified_relations_and_variants_via_gpt(
    support_sentences: List[str],
    question: str,
    gpt_model: str = "gpt-5-mini"
) -> Dict[str, Any]:
    prompt = prompt_q2_unified_relations_and_variants(support_sentences, question)
    raw, dt = _timed_call(call_gpt_5_1, prompt, model=gpt_model)
    parsed = extract_json_object(raw)

    rels = parsed.get("relations_direct", [])
    if not isinstance(rels, list):
        rels = []
    clean_rels = []
    for r in rels:
        if not isinstance(r, dict):
            continue
        h = str(r.get("head", "")).strip()
        t = str(r.get("tail", "")).strip()
        rel = normalize_relation(str(r.get("relation", "")).strip())
        src = str(r.get("source_sentence", "")).strip()
        if re.fullmatch(r"[A-Z]", h) and re.fullmatch(r"[A-Z]", t) and (rel in REL_OPTIONS_NO_DK) and src:
            clean_rels.append({"head": h, "tail": t, "relation": rel, "source_sentence": src})

    variants = parsed.get("variants", {})
    if not isinstance(variants, dict):
        variants = {}

    support_fallback = " ".join([s.strip() for s in (support_sentences or []) if isinstance(s, str) and s.strip()]).strip()
    canon_lines = "\n".join([f"{r['head']} is {r['relation']} of {r['tail']}" for r in clean_rels]) if clean_rels else support_fallback

    def _v(level: str, field: str, fallback: str) -> str:
        obj = variants.get(level, {})
        if isinstance(obj, dict):
            v = obj.get(field, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return fallback

    clean_variants = {
        "simple": {
            "support_text": _v("simple", "support_text", support_fallback),
            "question": _v("simple", "question", question),
        },
        "hinted": {
            "support_text": _v("hinted", "support_text", support_fallback),
            "question": _v("hinted", "question", question),
        },
        "canonical": {
            "support_text": _v("canonical", "support_text", canon_lines),
            "question": _v("canonical", "question", question),
        },
    }

    flip_consistency = parsed.get("flip_consistency", {})
    if not isinstance(flip_consistency, dict):
        flip_consistency = {}

    flipped_q = flip_consistency.get("flipped_question", question)
    if not isinstance(flipped_q, str) or not flipped_q.strip():
        flipped_q = question

    flip_applied = bool(flip_consistency.get("flip_applied", False))

    ref_rel = flip_consistency.get("reference_relation", None)
    if not isinstance(ref_rel, dict):
        ref_rel = {}
    ref_h = str(ref_rel.get("head", "")).strip()
    ref_t = str(ref_rel.get("tail", "")).strip()
    ref_r = normalize_relation(str(ref_rel.get("relation", "")).strip())
    if not (re.fullmatch(r"[A-Z]", ref_h or "") and re.fullmatch(r"[A-Z]", ref_t or "") and (ref_r in REL_OPTIONS_NO_DK)):
        if clean_rels:
            ref_h, ref_r, ref_t = clean_rels[0]["head"], clean_rels[0]["relation"], clean_rels[0]["tail"]
        else:
            ref_h, ref_r, ref_t = "", "", ""

    flip_reason = flip_consistency.get("reason", "")
    if not isinstance(flip_reason, str):
        flip_reason = ""
    used_order_fallback = bool(parsed.get("used_order_fallback", False))

    bundle = {
        "used_order_fallback": used_order_fallback,
        "relations_direct": clean_rels,
        "variants": clean_variants,
        "flip_consistency": {
            "flipped_question": flipped_q.strip(),
            "flip_applied": flip_applied,
            "reference_relation": {"head": ref_h, "relation": ref_r, "tail": ref_t} if (ref_h and ref_t and ref_r) else {},
            "reason": flip_reason.strip(),
        }
    }

    return {"prompt": prompt, "raw": raw, "parsed": parsed, "bundle": bundle, "timing_sec": dt}



# ============================================================
# 5) SUPPORT SENTENCES  (now from utils/switch_helpers)
# 6) ABLATION            (now from utils/switch_helpers)
# ============================================================


def get_story_text_and_sentences(inst: Dict[str, Any]) -> Tuple[str, List[str]]:
    ss = inst.get("story_sentences", None)
    if isinstance(ss, list) and any(isinstance(x, str) and x.strip() for x in ss):
        story_sents = [str(x).strip() for x in ss if isinstance(x, str) and x.strip()]
        fixed = []
        for s in story_sents:
            if s and s[-1] not in ".!?":
                fixed.append(s + ".")
            else:
                fixed.append(s)
        story_text = " ".join(fixed).strip()
        return story_text, story_sents
    story_text = str(inst.get("story", "") or "").strip()
    story_sents = split_sentences(story_text)
    return story_text, story_sents


# ============================================================
# 9) STABILITY  (unchanged)
# ============================================================
def stability_maxfreq(answers: List[str]) -> float:
    cleaned = [normalize_relation(a) for a in answers if isinstance(a, str) and a.strip()]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return 0.0
    counts: Dict[str, int] = {}
    for a in cleaned:
        counts[a] = counts.get(a, 0) + 1
    maxf = max(counts.values())
    return float(maxf / len(cleaned))


# ============================================================
# 10) COMPLEXITY  (unchanged logic, receives pre-computed HL)
# ============================================================
def compute_complexity(
    story: str,
    question: str,
    support_sentences: List[str],
    k_hop: Optional[int],
    structured_relations_used: List[Dict[str, Any]],
    dataset_name: str,
    precomputed_hl: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    C3 v2 -- redesigned complexity metric.

    Components (6 signals):
      SB  (Support Burden, 0.20)       -- saturating on k_support count.
            How many support sentences the model must chain together.
            1 sent is easy, 5+ is hard.  Formula: saturating(k_support, c=5)
      SD  (Selection Difficulty, 0.15) -- distractor ratio.
            How hard it is to find the right sentences among noise.
            Formula: (n_story - k_support) / n_story  (0 when all are support)
      CL  (Chain Length, 0.20)         -- ground-truth chain length.
            Anchors complexity to actual problem size regardless of how
            many sentences the model chose to look at.
            Formula: saturating(k_hop, c=5)
      HL  (Hard Language, 0.25)        -- GPT-scored; captures clock/compass
            directions AND ambiguous phrasing ("both there", "presents right
            to", "parallel + on top") that confuses small models.
      DB  (Diagonal Burden, 0.10)      -- fraction of diagonal relations.
      EL  (Entity Load, 0.10)          -- saturating on entity count.

    SB + SD capture the model's *perception* of chain difficulty.
    CL anchors to the *actual* chain length (since k_support is often
    under-reported by small models -- median k_support=2 even at k_hop=10).

    If precomputed_hl is provided (from a parallel GPT call), uses it directly
    instead of making a fresh GPT call.  This is the key C3 speed-up.
    """
    def saturating_ratio(x: float, c: float) -> float:
        x = float(max(0.0, x))
        c = float(max(1e-9, c))
        return float(x / (x + c))

    # ---- Story sentences ----
    story_sents = split_sentences(story) if story else []
    n_story = len(story_sents)

    # ---- Support sentences ----
    support_sents = [s.strip() for s in (support_sentences or []) if isinstance(s, str) and s.strip()]
    k_support = len(support_sents)

    # SB (Reasoning Chain Burden): how many support sentences the model must
    # chain together. 1 sentence is trivial, 5+ is hard.
    # saturating(k_support, c=5): 1->0.167, 2->0.286, 3->0.375, 5->0.500
    support_burden = saturating_ratio(k_support, c=5.0)

    # SD (Selection Difficulty): how hard it is to find the right sentences
    # among distractors. If 2 out of 10, SD=0.8 (lots of noise to sift).
    # If 1 out of 1, SD=0 (nothing to search for).
    selection_difficulty = float((n_story - k_support) / max(1, n_story)) if n_story > 0 else 0.0

    # CL (Chain Length): ground-truth anchor. Small models under-report
    # k_support (median=2 even at k=10), so SB+SD alone plateau.
    # CL = saturating(k_hop, c=5) rises monotonically with actual chain len.
    k_hop_val = int(k_hop) if k_hop is not None else k_support
    chain_length = saturating_ratio(k_hop_val, c=5.0)

    # ---- Entities ----
    ents_all = sorted(set(find_entities(f"{story or ''} {question or ''}")))
    num_entities = len(ents_all)
    entity_load = saturating_ratio(num_entities, c=6.0)

    # ---- Relations ----
    rels = structured_relations_used or []
    num_relations = len(rels)
    diag_ct = 0
    for r in rels:
        if isinstance(r, dict):
            rel = normalize_relation(str(r.get("relation", "")))
        else:
            rel = normalize_relation(str(r))
        if rel in DIAGONAL_SET:
            diag_ct += 1
    diagonal_burden = float(diag_ct / max(1, num_relations))

    # ---- Hard Language (v2: now includes ambiguity) ----
    support_text = " ".join(support_sents).strip()
    hardlang_note = "missing"
    hardlang_scores = None
    hardlang_timing_sec = 0.0

    try:
        if precomputed_hl is not None:
            hl = precomputed_hl
        else:
            hl = extract_hard_language_scores(support_text, gpt_model="gpt-5-mini")

        hardlang_timing_sec = float(hl.get("timing_sec", 0.0)) if isinstance(hl, dict) else 0.0
        items = hl.get("items", []) if isinstance(hl, dict) else []
        ds = []
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                ds.append(_clip01(float(it.get("difficulty", 0.0))))
            except Exception:
                continue
        hardlang_scores = items
        if ds:
            hard_language = float(0.6 * max(ds) + 0.4 * (sum(ds) / len(ds)))
            hardlang_note = "ok"
        else:
            hard_language = 0.0
            hardlang_note = "no_items"
    except Exception as e:
        hard_language = 0.0
        hardlang_note = f"error:{type(e).__name__}"

    hard_language = _clip01(hard_language)

    # ---- Weights (C3 v2) ----
    # SB+SD = model's perception of chain difficulty (0.20+0.15=0.35)
    # CL    = ground-truth chain length anchor       (0.20)
    # HL    = linguistic difficulty + ambiguity       (0.25)
    # DB+EL = structural features                    (0.10+0.10=0.20)
    w_SB = 0.20   # reasoning chain burden (how many links model selected)
    w_SD = 0.15   # selection difficulty   (how hard to find the right links)
    w_CL = 0.20   # chain length           (actual problem size, monotonic)
    w_HL = 0.25   # hard language + ambiguity
    w_DB = 0.10   # diagonal burden
    w_EL = 0.10   # entity load

    complexity_score = (
        w_SB * support_burden
        + w_SD * selection_difficulty
        + w_CL * chain_length
        + w_HL * hard_language
        + w_DB * diagonal_burden
        + w_EL * entity_load
    )
    complexity_score = _clip01(complexity_score)

    return {
        "dataset_name": dataset_name,
        "num_entities": num_entities,
        "k_support": k_support,
        "n_story_sentences": n_story,
        "num_relations_used": num_relations,
        "diagonal_relations": diag_ct,
        "k_hop_used": k_hop_val,
        "components": {
            "support_burden_SB": float(support_burden),
            "selection_difficulty_SD": float(selection_difficulty),
            "chain_length_CL": float(chain_length),
            "hard_language_HL": float(hard_language),
            "diagonal_burden_DB": float(diagonal_burden),
            "entity_load_EL": float(entity_load),
        },
        "weights": {"SB": w_SB, "SD": w_SD, "CL": w_CL, "HL": w_HL, "DB": w_DB, "EL": w_EL},
        "complexity_score_0_1": float(complexity_score),
        "feature_coverage": 1.0,
        "details": {
            "support_text_used_for_HL": support_text,
            "hard_language_items": hardlang_scores,
            "hard_language_status": hardlang_note,
            "hard_language_timing_sec": float(hardlang_timing_sec),
            "notes": {
                "v2_formula": True,
                "SB_is_saturating_k_support_c5": True,
                "SD_is_distractor_ratio": True,
                "CL_is_saturating_k_hop_c5": True,
                "CL_anchors_for_k_support_underreporting": True,
                "HL_includes_ambiguity": True,
                "HL_uses_0.6*max+0.4*mean": True,
            }
        }
    }

def build_text_only_with_relations_prompt_from_inst(sample: Dict[str, Any]) -> str:
    candidate_options = REL_OPTIONS_NO_DK[:]
    options_list = ", ".join(candidate_options)
    options_json = json.dumps(candidate_options, ensure_ascii=False)

    pred_rels = (
        sample.get("relation_extraction", {}).get("predicted_relations")
        or sample.get("predicted_relations")
        or sample.get("relations")
        or []
    )
    rels_block = relations_to_sentences_verbatim(pred_rels)
    story, story_sents = get_story_text_and_sentences(sample)
    question = (sample.get("question", "") or "").strip()

    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<reason using ONLY relations (story only if tie-break)>",
  "selected_option": "<one of: {options_list}>"
}}

Source of truth:
- USE ONLY the RELATIONS list below to answer. Ignore the STORY unless a tie-break is needed.
- If the relations are insufficient to uniquely determine the answer, choose the best-supported option from {options_json} and explain why.

RELATIONS:
{rels_block}

STORY (ignore unless needed):
{story}

Question:
{question}

{SEMANTICS_BLOCK}

How to compute BEFORE you output (brief internal steps, do not include in JSON):
1) Build constraints from each relation edge.
2) Compose constraints (transitivity, merging overlaps).
3) Determine asked relation between the two queried entities.
4) Map to exactly one option in {options_json}.
5) Output ONLY the JSON.
""".strip()


def _unwrap_inst(inst: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(inst, dict):
        return {}
    for k in ["record", "example", "ex", "item", "data"]:
        v = inst.get(k, None)
        if isinstance(v, dict) and ("question" in v or "story" in v or "solver" in v):
            return v
    return inst


def get_ground_truth_label(inst: Dict[str, Any]) -> str:
    inst = _unwrap_inst(inst)
    for k in ["ground_truth", "label", "gold","gold_relation","answer", "oracle_answer", "target", "gt"]:
        v = inst.get(k, None)
        if isinstance(v, str) and v.strip():
            return normalize_relation(v)
    solver = inst.get("solver", None)
    if isinstance(solver, dict):
        fvg = solver.get("final_prediction_vs_gold", None)
        if isinstance(fvg, dict):
            gold = fvg.get("gold", None)
            if isinstance(gold, str) and gold.strip():
                return normalize_relation(gold)
    return ""


def get_grid_correct(inst: Dict[str, Any]) -> Optional[bool]:
    inst = _unwrap_inst(inst)
    solver = inst.get("solver", None)
    if not isinstance(solver, dict):
        return None
    fvg = solver.get("grid_predicted", None)
    if not isinstance(fvg, dict):
        return None
    if "correct" in fvg:
        return bool(fvg["correct"])
    pred = fvg.get("predicted", None)
    gold = fvg.get("gold", None)
    if isinstance(pred, str) and isinstance(gold, str) and pred.strip() and gold.strip():
        return normalize_relation(pred) == normalize_relation(gold)
    return None


# ============================================================
# 13) FAST INSTANCE RUNNER
# ============================================================
def run_instance_one_record(
    inst: Dict[str, Any],
    llama_model: str,
    temp: float,
    dataset_name: str,
    trust_noswitch_threshold: float = 0.70,
    complexity_noswitch_threshold: float = 0.55,
    skip_eval_baselines: bool = False,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """
    Same output schema as the original, but with three speed-ups:
      1) Parallel LLM calls (ThreadPool)
      2) Short-circuit: skip Q2 when decision is already determined
      3) Cached LLM clients

    If skip_eval_baselines=True, the two expensive evaluation-only Ollama calls
    (baseline_symbolic_text_only and text_only_with_relations) are skipped entirely.
    These are NOT used by the switching decision; they only measure accuracy for
    comparison.  This typically saves 60-75% of per-instance wall time.

    NOTE on Ollama parallelism:
      Ollama queues requests behind OLLAMA_NUM_PARALLEL (default=4 on recent
      versions, 1 on older).  If your Ollama is set to 1, ThreadPool parallelism
      won't help for Ollama calls.  Set the env var before starting Ollama:
        $env:OLLAMA_NUM_PARALLEL = 4   # PowerShell
        export OLLAMA_NUM_PARALLEL=4   # bash
    """
    t_inst0 = _now()
    meta = inst.get("meta", {}) or {}
    dataset_id = str(meta.get("dataset_id", inst.get("dataset_id", "")))
    index = int(meta.get("index", inst.get("index", -1)))
    k_hop = meta.get("k_hop", inst.get("k_hop", None))
    print("index", index)

    story, story_sents = get_story_text_and_sentences(inst)
    question = str(inst.get("question", "")).strip()
    gt = get_ground_truth_label(inst)
    grid_correct = str(inst.get("grid_correct", "")).strip()

    print("story", story, "question", question, "gt", gt, grid_correct)

    prompts_and_outputs = inst.get("prompts_and_outputs", {}) or {}
    timing: Dict[str, Any] = {
        "baseline_symbolic_text_only_sec": None,
        "text_only_with_relations_sec": None,
        "Q1_total_sec": None,
        "Q2_total_sec": None,
        "Q3_total_sec": None,
        "switch_policy_eval_sec": None,
        "instance_total_sec": None,
        "short_circuit": None,
        "details": {"Q1": {}, "Q2": {}, "Q3": {}},
    }

    # Token accumulator for the whole instance
    tokens: Dict[str, Any] = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "per_call": [],
    }

    def _acc_tokens(label: str, tok: Dict[str, Any]) -> None:
        """Accumulate token counts from a single Ollama call."""
        tokens["total_prompt_tokens"]     += int(tok.get("prompt_tokens", 0) or 0)
        tokens["total_completion_tokens"] += int(tok.get("completion_tokens", 0) or 0)
        tokens["per_call"].append({"label": label, **tok})

    # ===========================================================
    # ROUND 1 (parallel): Q1A always; Baseline + Text+Rels only if needed
    # ===========================================================
    t_round1_start = _now()

    q1a_prompt = prompt_q1_baseline_and_support_sentences(story, question)
    fut_q1a = _submit(call_ollama_llama, q1a_prompt, model=llama_model, temperature=temp, seed=seed)

    baseline_prompt = None
    rels_prompt = None
    fut_base = None
    fut_rels = None
    baseline_correct = None
    rels_correct = None

    if not skip_eval_baselines:
        baseline_prompt = build_symbolic_QA_prompt(
            story_text=story, question_text=question, candidate_answers=REL_OPTIONS_NO_DK[:]
        )
        rels_prompt = build_text_only_with_relations_prompt_from_inst(inst)
        fut_base = _submit(call_ollama_llama, baseline_prompt, model=llama_model, temperature=temp, seed=seed)
        fut_rels = _submit(call_ollama_llama, rels_prompt, model=llama_model, temperature=temp, seed=seed)

    # Collect Q1A first (needed by everything downstream)
    (q1a_raw, q1a_tok), dt_q1a = fut_q1a.result()
    _acc_tokens("Q1A_baseline_support", q1a_tok)

    # Collect baselines if they were submitted
    if fut_base is not None:
        (baseline_raw, base_tok), dt_base = fut_base.result()
        _acc_tokens("baseline_symbolic_text_only", base_tok)
        baseline_parsed   = extract_json_object(baseline_raw)
        baseline_selected = _extract_selected_option_any(baseline_parsed)
        baseline_correct  = _compute_lm_correct(baseline_selected, gt)
        timing["baseline_symbolic_text_only_sec"] = float(dt_base)
        prompts_and_outputs["baseline_symbolic_text_only"] = {
            "prompt": baseline_prompt, "lm_answer_raw": baseline_raw,
            "parsed": baseline_parsed, "selected_option": baseline_selected,
            "lm_correct": baseline_correct,
            "tokens": base_tok,
        }
    else:
        timing["baseline_symbolic_text_only_sec"] = 0.0

    if fut_rels is not None:
        (rels_raw, rels_tok), dt_rels = fut_rels.result()
        _acc_tokens("text_only_with_relations", rels_tok)
        rels_parsed   = extract_json_object(rels_raw)
        rels_selected = _extract_selected_option_any(rels_parsed)
        rels_correct  = _compute_lm_correct(rels_selected, gt)
        timing["text_only_with_relations_sec"] = float(dt_rels)
        prompts_and_outputs["text_only_with_relations"] = {
            "prompt": rels_prompt, "lm_answer_raw": rels_raw,
            "parsed": rels_parsed, "selected_option": rels_selected,
            "lm_correct": rels_correct,
            "tokens": rels_tok,
        }
    else:
        timing["text_only_with_relations_sec"] = 0.0

    t_round1_end = _now()
    timing["round1_wall_sec"] = float(_secs(t_round1_start, t_round1_end))

    # Parse Q1A → support sentences
    q1a_obj = extract_json_object(q1a_raw)
    q1_answer = normalize_relation(q1a_obj.get("answer", ""))
    support_sentences = q1a_obj.get("support_sentences", [])
    if not isinstance(support_sentences, list):
        support_sentences = []
    support_sentences = match_support_sentences_to_story(story, support_sentences, story_sentences=story_sents)
    if not support_sentences:
        q_ents = set(find_entities(question))
        story_sents2 = split_sentences(story)
        cand = [s for s in story_sents2 if any(e in s for e in q_ents)]
        support_sentences = cand[:2] if cand else story_sents2[:2]

    q1_just = str(q1a_obj.get("justification", "")).strip()
    support_only_text = " ".join([s.strip() for s in support_sentences if isinstance(s, str) and s.strip()]).strip()

    # ===========================================================
    # ROUND 2 (parallel): Q1S + Q1C + GPT-unified + GPT-hardlang
    #   All depend only on support_sentences from Q1A.
    # ===========================================================
    tQ10 = _now()

    q1s_prompt = prompt_answer_only_allow_dontknow(support_only_text, question)
    ablated_story, ablation_info = build_ablated_story_remove_sentences(story, support_sentences)
    q1c_prompt = prompt_answer_only_allow_dontknow(ablated_story, question)
    support_text_for_hl = " ".join([s.strip() for s in support_sentences if isinstance(s, str) and s.strip()]).strip()
    hl_prompt_text = prompt_hard_language_scores(support_text_for_hl)

    fut_q1s = _submit(call_ollama_llama, q1s_prompt, model=llama_model, temperature=temp, seed=seed)
    fut_q1c = _submit(call_ollama_llama, q1c_prompt, model=llama_model, temperature=temp, seed=seed)

    # GPT calls (unified + hard language) run in parallel with Ollama
    gpt_unified_prompt = prompt_q2_unified_relations_and_variants(support_sentences, question)
    fut_gpt_unified = _submit(call_gpt_5_1, gpt_unified_prompt, model="gpt-5-mini")
    fut_gpt_hl      = _submit(call_gpt_5_1, hl_prompt_text, model="gpt-5-mini")

    # Collect Q1S
    (q1s_raw, q1s_tok), dt_q1s = fut_q1s.result()
    _acc_tokens("Q1S_support_only", q1s_tok)
    q1s_obj = extract_json_object(q1s_raw)
    q1s_answer = normalize_relation(q1s_obj.get("answer", ""))
    q1s_just = str(q1s_obj.get("justification", "")).strip()
    q1_support_only_score = 1.0 if (q1s_answer and q1_answer and q1s_answer == q1_answer) else 0.0

    # Collect Q1C
    (q1c_raw, q1c_tok), dt_q1c = fut_q1c.result()
    _acc_tokens("Q1C_story_minus_support", q1c_tok)
    q1c_obj = extract_json_object(q1c_raw)
    ablated_answer = normalize_relation(q1c_obj.get("answer", ""))
    ablated_just = str(q1c_obj.get("justification", "")).strip()
    q1_minus_support_score = 1.0 if (ablated_answer == "dont know") else 0.0

    q1_score = float(np.mean([q1_support_only_score, q1_minus_support_score]))

    tQ11 = _now()
    timing["details"]["Q1"] = {
        "Q1A_sec": float(dt_q1a), "Q1S_sec": float(dt_q1s),
        "Q1C_sec": float(dt_q1c), "Q1_total_sec": float(_secs(tQ10, tQ11)),
    }
    timing["Q1_total_sec"] = float(_secs(tQ10, tQ11))

    # Collect GPT hard language (for C3)
    gpt_hl_raw, dt_gpt_hl = fut_gpt_hl.result()
    gpt_hl_parsed = extract_json_object(gpt_hl_raw)
    hl_items = gpt_hl_parsed.get("items", [])
    if not isinstance(hl_items, list):
        hl_items = []
    clean_hl_items = []
    for it in hl_items:
        if not isinstance(it, dict):
            continue
        span = str(it.get("span", "")).strip()
        typ = str(it.get("type", "")).strip().lower()
        try:
            diff = float(it.get("difficulty", 0.0))
        except Exception:
            diff = 0.0
        if not span:
            continue
        if typ not in {"clock", "direction", "ambiguity"}:
            typ = "direction"
        clean_hl_items.append({"span": span, "type": typ, "difficulty": _clip01(diff)})
    precomputed_hl = {
        "prompt": hl_prompt_text, "raw": gpt_hl_raw,
        "parsed": gpt_hl_parsed, "items": clean_hl_items, "timing_sec": dt_gpt_hl,
    }

    # Collect GPT unified (for Q2 availability check)
    gpt_unified_raw, dt_gpt_unified = fut_gpt_unified.result()
    gpt_unified_parsed = extract_json_object(gpt_unified_raw)

    # Reconstruct the same bundle that q2_unified_relations_and_variants_via_gpt returns
    _rels = gpt_unified_parsed.get("relations_direct", [])
    if not isinstance(_rels, list):
        _rels = []
    direct_rels = []
    for r in _rels:
        if not isinstance(r, dict):
            continue
        h = str(r.get("head", "")).strip()
        t = str(r.get("tail", "")).strip()
        rel = normalize_relation(str(r.get("relation", "")).strip())
        src = str(r.get("source_sentence", "")).strip()
        if re.fullmatch(r"[A-Z]", h) and re.fullmatch(r"[A-Z]", t) and (rel in REL_OPTIONS_NO_DK) and src:
            direct_rels.append({"head": h, "tail": t, "relation": rel, "source_sentence": src})

    _variants_raw = gpt_unified_parsed.get("variants", {})
    if not isinstance(_variants_raw, dict):
        _variants_raw = {}
    support_fallback = " ".join([s.strip() for s in (support_sentences or []) if isinstance(s, str) and s.strip()]).strip()
    canon_lines = "\n".join([f"{r['head']} is {r['relation']} of {r['tail']}" for r in direct_rels]) if direct_rels else support_fallback

    def _v(level, field, fallback):
        obj = _variants_raw.get(level, {})
        if isinstance(obj, dict):
            v = obj.get(field, "")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return fallback

    variants = {
        "simple":    {"support_text": _v("simple", "support_text", support_fallback),    "question": _v("simple", "question", question)},
        "hinted":    {"support_text": _v("hinted", "support_text", support_fallback),    "question": _v("hinted", "question", question)},
        "canonical": {"support_text": _v("canonical", "support_text", canon_lines),      "question": _v("canonical", "question", question)},
    }

    _flip_cons = gpt_unified_parsed.get("flip_consistency", {})
    if not isinstance(_flip_cons, dict):
        _flip_cons = {}
    flipped_q_from_gpt = str(_flip_cons.get("flipped_question", question) or question).strip()
    flip_applied = bool(_flip_cons.get("flip_applied", False))
    _ref_rel = _flip_cons.get("reference_relation", None)
    if not isinstance(_ref_rel, dict):
        _ref_rel = {}
    ref_h = str(_ref_rel.get("head", "")).strip()
    ref_t = str(_ref_rel.get("tail", "")).strip()
    ref_r = normalize_relation(str(_ref_rel.get("relation", "")).strip())
    if not (re.fullmatch(r"[A-Z]", ref_h or "") and re.fullmatch(r"[A-Z]", ref_t or "") and (ref_r in REL_OPTIONS_NO_DK)):
        if direct_rels:
            ref_h, ref_r, ref_t = direct_rels[0]["head"], direct_rels[0]["relation"], direct_rels[0]["tail"]
        else:
            ref_h, ref_r, ref_t = "", "", ""
    flip_reason = str(_flip_cons.get("reason", "") or "").strip()
    used_order_fallback = bool(gpt_unified_parsed.get("used_order_fallback", False))

    flip_ref_rel = {"head": ref_h, "relation": ref_r, "tail": ref_t} if (ref_h and ref_t and ref_r) else {}

    uq = {
        "prompt": gpt_unified_prompt, "raw": gpt_unified_raw,
        "parsed": gpt_unified_parsed,
        "bundle": {
            "used_order_fallback": used_order_fallback,
            "relations_direct": direct_rels,
            "variants": variants,
            "flip_consistency": {
                "flipped_question": flipped_q_from_gpt,
                "flip_applied": flip_applied,
                "reference_relation": flip_ref_rel,
                "reason": flip_reason,
            },
        },
        "timing_sec": dt_gpt_unified,
    }

    q2_rel_lines = [f"{r['head']} is {r['relation']} of {r['tail']}" for r in direct_rels]
    q2_available = bool(direct_rels)

    # ===========================================================
    # COMPUTE C3 (uses precomputed HL — no extra GPT call)
    # ===========================================================
    tQ30 = _now()
    complexity = compute_complexity(
        story=story, question=question,
        support_sentences=support_sentences,
        k_hop=k_hop,
        structured_relations_used=(direct_rels if q2_available else []),
        dataset_name=dataset_name,
        precomputed_hl=precomputed_hl,
    )
    q3 = float(complexity["complexity_score_0_1"])
    tQ31 = _now()
    timing["details"]["Q3"] = {
        "Q3_total_sec": float(_secs(tQ30, tQ31)),
        "hard_language_gpt_sec": float(dt_gpt_hl),
    }
    timing["Q3_total_sec"] = float(_secs(tQ30, tQ31))

    # ===========================================================
    # SHORT-CIRCUIT CHECK: Can we skip Q2?
    #
    # Decision rule: stay text-only iff (C3 < τ_c AND T > τ_t)
    #
    # Three cases where Q2 is unnecessary:
    #   A) Q2 unavailable (no direct relations from GPT)
    #   B) C3 >= τ_c  → will switch regardless of trust
    #   C) C1 low enough that T_max = 0.6*C1 + 0.4*1.0 <= τ_t → will switch
    #   D) C3 < τ_c AND C1 alone > τ_t → won't switch even with worst Q2
    # ===========================================================
    tQ20 = _now()

    skip_q2 = False
    skip_reason = ""

    if not q2_available:
        skip_q2 = True
        skip_reason = "no_direct_relations"
    elif q3 >= complexity_noswitch_threshold:
        # Case B: high complexity -> always switch, trust irrelevant
        skip_q2 = True
        skip_reason = f"C3={q3:.3f} >= tau_c={complexity_noswitch_threshold} -> always switch"
    else:
        # C3 < tau_c, so the decision depends on trust.
        # T = 0.6*C1 + 0.4*C2 when Q2 available
        # Rule: no-switch iff C3 < tau_c AND trust >= tau_t  (note: >= not >)
        t_max = 0.6 * q1_score + 0.4 * 1.0   # best possible Q2
        t_min = 0.6 * q1_score + 0.4 * 0.0   # worst possible Q2

        if t_max < trust_noswitch_threshold:
            # Case C: even perfect Q2 can't reach threshold -> switch
            skip_q2 = True
            skip_reason = f"T_max={t_max:.3f} < tau_t={trust_noswitch_threshold} -> always switch"
        elif t_min >= trust_noswitch_threshold:
            # Case D: even zero Q2 meets threshold -> don't switch
            skip_q2 = True
            skip_reason = f"T_min={t_min:.3f} >= tau_t={trust_noswitch_threshold} -> never switch"

    q2_timing_detail: Dict[str, Any] = {
        "unified_relations_variants_gpt_sec": float(dt_gpt_unified),
        "q2_3_ollama_calls_sec_total": 0.0,
        "q2_3_ollama_calls_sec_each": [],
        "flip_ollama_call_sec": None,
        "Q2_total_sec": None,
        "short_circuited": skip_q2,
        "short_circuit_reason": skip_reason,
    }

    q2_block: Dict[str, Any] = {
        "available": False,
        "rule": "Q2 is computed ONLY if unified GPT returns at least one DIRECT relation from support sentences.",
        "relations_source": "support sentences only (direct-only; no inverse/transitive inference)",
        "relations_used": [],
    }

    q2_score = None

    if skip_q2:
        # ---- Q2 SKIPPED ----
        q2_block.update({
            "available": q2_available,
            "short_circuited": True,
            "short_circuit_reason": skip_reason,
            "q2_unified_relations_variants_via_gpt": uq,
            "relations_direct_used": direct_rels,
            "relations_used": q2_rel_lines,
        })
    else:
        # ---- ROUND 3 (parallel): 3 paraphrase variants + flip ----
        q2_runs = []
        q2_answers = []

        # Fire all 4 calls in parallel
        flip_text = variants["canonical"]["support_text"]
        flip_q = flipped_q_from_gpt
        flip_prompt = prompt_q2_answer_from_text(flip_text, flip_q)

        variant_futures = []
        for level in ["simple", "hinted", "canonical"]:
            s_text = variants[level]["support_text"]
            q_text = variants[level]["question"]
            p = prompt_q2_answer_from_text(s_text, q_text)
            variant_futures.append((level, s_text, q_text, p, _submit(call_ollama_llama, p, model=llama_model, temperature=temp, seed=seed)))

        fut_flip = _submit(call_ollama_llama, flip_prompt, model=llama_model, temperature=temp, seed=seed)

        # Collect variants
        for level, s_text, q_text, p, fut in variant_futures:
            (rraw, rtok), dt_call = fut.result()
            _acc_tokens(f"Q2_{level}", rtok)
            q2_timing_detail["q2_3_ollama_calls_sec_each"].append(float(dt_call))
            q2_timing_detail["q2_3_ollama_calls_sec_total"] += float(dt_call)

            robj = extract_json_object(rraw)
            ans = normalize_relation(robj.get("answer", ""))
            just = str(robj.get("justification", "")).strip()
            q2_answers.append(ans)
            q2_runs.append({
                "variant_level": level, "text_used": s_text, "question_used": q_text,
                "prompt": p, "raw": rraw, "parsed": robj,
                "extracted": {"answer": ans, "justification": just},
                "timing_sec": float(dt_call),
                "tokens": rtok,
            })

        paraphrase_stability = stability_maxfreq(q2_answers)

        # Collect flip
        (flip_raw, flip_tok), dt_flip = fut_flip.result()
        _acc_tokens("Q2_flip", flip_tok)
        q2_timing_detail["flip_ollama_call_sec"] = float(dt_flip)
        flip_obj = extract_json_object(flip_raw)
        flip_ans = normalize_relation(flip_obj.get("answer", ""))

        expected_flip = q1_answer if (not flip_applied) else INVERSE_REL.get(q1_answer, None)
        flip_score = 1.0 if (expected_flip is not None and flip_ans == expected_flip) else 0.0
        q2_score = float(np.mean([paraphrase_stability, flip_score]))

        q2_block.update({
            "available": True,
            "support_sentences_used": support_sentences,
            "support_only_text_used": support_only_text,
            "q2_unified_relations_variants_via_gpt": uq,
            "relations_direct_used": direct_rels,
            "relations_used": q2_rel_lines,
            "paraphrase_runs_3": q2_runs,
            "answers_across_3": q2_answers,
            "paraphrase_stability": paraphrase_stability,
            "flip_test": {
                "flipped_question": flip_q,
                "flip_applied": flip_applied,
                "flip_reason": flip_reason,
                "reference_relation": flip_ref_rel,
                "prompt": flip_prompt,
                "raw": flip_raw,
                "parsed": flip_obj,
                "extracted": {"answer": flip_ans},
                "expected_answer_given_flip_applied_rule": expected_flip,
                "rule_used": {
                    "if_flip_applied_false_expected": "q1_answer",
                    "if_flip_applied_true_expected": "INVERSE_REL[q1_answer]"
                },
                "score": flip_score,
                "tokens": flip_tok,
            },
            "q2_score": q2_score,
            "components": {"paraphrase_stability": paraphrase_stability, "flip_score": flip_score},
        })

    tQ21 = _now()
    q2_timing_detail["Q2_total_sec"] = float(_secs(tQ20, tQ21))
    timing["details"]["Q2"] = q2_timing_detail
    timing["Q2_total_sec"] = float(_secs(tQ20, tQ21))

    # ---------- Trust ----------
    if (not skip_q2) and q2_available and (q2_score is not None):
        trust = 0.60 * q1_score + 0.40 * float(q2_score)
        trust_weights_effective = {"Q1": 0.60, "Q2": 0.40}
    else:
        trust = float(q1_score)
        trust_weights_effective = {"Q1": 1.00, "Q2": 0.00}

    # ---------- Switch policy ----------
    tSW0 = _now()
    would_switch = not (q3 < complexity_noswitch_threshold and trust >= trust_noswitch_threshold)
    chosen_correct = rels_correct if would_switch else baseline_correct
    tSW1 = _now()
    timing["switch_policy_eval_sec"] = float(_secs(tSW0, tSW1))
    timing["short_circuit"] = skip_reason if skip_q2 else None

    t_inst1 = _now()
    timing["instance_total_sec"] = float(_secs(t_inst0, t_inst1))
    timing["decision_pipeline_sec"] = float(
        (timing["Q1_total_sec"] or 0.0) + (timing["Q2_total_sec"] or 0.0) + (timing["Q3_total_sec"] or 0.0)
    )

    # Finalize token summary
    tokens["total_tokens"] = tokens["total_prompt_tokens"] + tokens["total_completion_tokens"]

    return {
        "meta": {"dataset_id": dataset_id, "index": index, "k_hop": k_hop, "dataset_name": dataset_name,
                 "seed": seed},
        "ground_truth": gt,
        "grid_correct": grid_correct,
        "inputs": {"story": story, "question": question},
        "timing": timing,
        "lm_correct": {
            "baseline_symbolic_text_only": baseline_correct,
            "text_only_with_relations": rels_correct,
            "grid_correct": grid_correct,
        },
        "prompts_and_outputs": prompts_and_outputs,
        "Q1": {
            "A_baseline_support": {
                "prompt": q1a_prompt, "raw": q1a_raw, "parsed": q1a_obj,
                "extracted": {"answer": q1_answer, "support_sentences": support_sentences, "justification": q1_just},
                "tokens": q1a_tok,
            },
            "S_support_only": {
                "support_only_text": support_only_text,
                "prompt": q1s_prompt, "raw": q1s_raw, "parsed": q1s_obj,
                "extracted": {"answer": q1s_answer, "justification": q1s_just},
                "score": q1_support_only_score,
                "tokens": q1s_tok,
            },
            "C_story_minus_support": {
                "ablation_info": ablation_info, "ablated_story": ablated_story,
                "prompt": q1c_prompt, "raw": q1c_raw, "parsed": q1c_obj,
                "extracted": {"answer": ablated_answer, "justification": ablated_just},
                "score": q1_minus_support_score,
                "strict_rule": "score=1 only if answer == 'dont know'",
                "tokens": q1c_tok,
            },
            "q1_score": q1_score,
        },
        "Q2": q2_block,
        "Q3": {"complexity": complexity, "q3_score": q3},
        "tokens": tokens,
        "scores": {
            "trustworthiness_score": float(trust),
            "complexity_score": q3,
            "would_switch": would_switch,
            "switch_policy_correct": chosen_correct,
            "switch_policy_rule": {
                "no_switch_if": f"(complexity < {complexity_noswitch_threshold}) AND (trust >= {trust_noswitch_threshold})",
                "switch_otherwise": True
            },
            "weights_effective": {"trust": trust_weights_effective}
        }
    }


# ============================================================
# 14) RUN DATASET  (unchanged structure, uses fast runner)
# ============================================================
# _safe_mean imported from utils/switch_helpers

def compute_timing_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    recs = [r for r in records if isinstance(r, dict) and "timing" in r and "scores" in r]
    if not recs:
        return {"n": 0, "error": "no timing found"}

    q1, q2, q3, inst, decision, base, rels, switch_eval = [], [], [], [], [], [], [], []
    switch_only_inst_times = []
    short_circuited_count = 0

    for r in recs:
        t = r.get("timing", {}) or {}
        q1.append(float(t.get("Q1_total_sec") or 0.0))
        q2.append(float(t.get("Q2_total_sec") or 0.0))
        q3.append(float(t.get("Q3_total_sec") or 0.0))
        inst.append(float(t.get("instance_total_sec") or 0.0))
        decision.append(float(t.get("decision_pipeline_sec") or 0.0))
        base.append(float(t.get("baseline_symbolic_text_only_sec") or 0.0))
        rels.append(float(t.get("text_only_with_relations_sec") or 0.0))
        switch_eval.append(float(t.get("switch_policy_eval_sec") or 0.0))
        if t.get("short_circuit"):
            short_circuited_count += 1
        if bool((r.get("scores", {}) or {}).get("would_switch", False)):
            switch_only_inst_times.append(float(t.get("instance_total_sec") or 0.0))

    # Aggregate token counts
    total_prompt_tok = 0
    total_completion_tok = 0
    for r in recs:
        tok = r.get("tokens", {}) or {}
        total_prompt_tok     += int(tok.get("total_prompt_tokens", 0) or 0)
        total_completion_tok += int(tok.get("total_completion_tokens", 0) or 0)

    return {
        "n": len(recs),
        "short_circuited_q2_count": short_circuited_count,
        "totals_sec": {
            "Q1_total": float(sum(q1)), "Q2_total": float(sum(q2)),
            "Q3_total": float(sum(q3)), "decision_pipeline_total": float(sum(decision)),
            "instance_total": float(sum(inst)),
        },
        "means_sec": {
            "Q1": _safe_mean(q1), "Q2": _safe_mean(q2), "Q3": _safe_mean(q3),
            "decision_pipeline": _safe_mean(decision),
            "instance_total": _safe_mean(inst),
            "baseline_symbolic_text_only": _safe_mean(base),
            "text_only_with_relations": _safe_mean(rels),
            "switch_policy_eval": _safe_mean(switch_eval),
        },
        "avg_instance_time_when_would_switch_sec": _safe_mean(switch_only_inst_times) if switch_only_inst_times else None,
        "switch_count": int(len(switch_only_inst_times)),
        "tokens": {
            "total_prompt_tokens": total_prompt_tok,
            "total_completion_tokens": total_completion_tok,
            "total_tokens": total_prompt_tok + total_completion_tok,
            "mean_prompt_tokens_per_instance": float(total_prompt_tok / max(1, len(recs))),
            "mean_completion_tokens_per_instance": float(total_completion_tok / max(1, len(recs))),
        },
    }


def run_dataset_one_pretty_json(
    input_path: str,
    output_json: str,
    dataset_name: str,
    llama_model: str = "llama3.1:8b",
    temp: float = 0.0,
    limit: Optional[int] = 20,
    start_after_index: int = 10,
    trust_noswitch_threshold: float = 0.70,
    complexity_noswitch_threshold: float = 0.55,
    skip_eval_baselines: bool = False,
    seed: int = DEFAULT_SEED,
) -> List[Dict[str, Any]]:

    t0 = _now()
    data = read_json_or_jsonl(input_path)

    filtered = []
    for ex in data:
        meta = ex.get("meta", {}) or {}
        ex_index = meta.get("index", ex.get("index", None))
        try:
            ex_index_int = int(ex_index)
        except Exception:
            ex_index_int = -1
        if ex_index_int > start_after_index:
            filtered.append(ex)

    if isinstance(limit, int):
        filtered = filtered[:limit]

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        f.write("")

    records: List[Dict[str, Any]] = []

    for i, ex in enumerate(filtered):
        try:
            rec = run_instance_one_record(
                ex,
                llama_model=llama_model,
                temp=temp,
                dataset_name=dataset_name,
                trust_noswitch_threshold=trust_noswitch_threshold,
                complexity_noswitch_threshold=complexity_noswitch_threshold,
                skip_eval_baselines=skip_eval_baselines,
                seed=seed,
            )
        except Exception as e:
            rec = {"meta": ex.get("meta", {}), "error": str(e)}

        records.append(rec)
        with open(output_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=2))
            f.write("\n\n")

    timing_summary = compute_timing_summary(records)
    timing_summary_path = output_json.replace(".json", "_timing_summary.json")
    with open(timing_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "input_path": input_path,
                "output_json": output_json,
                "runtime_total_sec": float(_secs(t0, _now())),
                "timing_summary": timing_summary,
            },
            f, ensure_ascii=False, indent=2,
        )

    print(f"Saved STREAMING pretty JSON objects: {output_json} (n={len(records)})")
    print(f"Saved timing summary: {timing_summary_path}")
    return records


# ============================================================
# 15) SUMMARY STATS + TIMING PRINTS
# ============================================================
def print_summary_stats(records: List[Dict[str, Any]]) -> None:
    recs = [r for r in records if isinstance(r, dict) and "scores" in r]
    n = len(recs)
    if n == 0:
        print("No valid records to summarize.")
        return

    B = [r.get("lm_correct", {}).get("baseline_symbolic_text_only", None) for r in recs]
    R = [r.get("lm_correct", {}).get("text_only_with_relations", None) for r in recs]
    Q1 = [float(r["Q1"]["q1_score"]) for r in recs]

    Q2 = []
    Q2_avail = []
    for r in recs:
        q2 = r.get("Q2", {}) or {}
        avail = bool(q2.get("available", False)) and not bool(q2.get("short_circuited", False))
        Q2_avail.append(avail)
        if avail and ("q2_score" in q2) and (q2["q2_score"] is not None):
            Q2.append(float(q2["q2_score"]))
        else:
            Q2.append(float("nan"))

    Q3 = [float(r["scores"]["complexity_score"]) for r in recs]
    T  = [float(r["scores"]["trustworthiness_score"]) for r in recs]
    SW  = [bool(r["scores"]["would_switch"]) for r in recs]
    SWC = [r["scores"]["switch_policy_correct"] for r in recs]

    base_known = [b for b in B if b is not None]
    rels_known = [b for b in R if b is not None]
    swc_known  = [b for b in SWC if b is not None]

    base_acc = np.mean([1.0 if b else 0.0 for b in base_known]) if base_known else float("nan")
    rels_acc = np.mean([1.0 if b else 0.0 for b in rels_known]) if rels_known else float("nan")
    sw_acc   = np.mean([1.0 if b else 0.0 for b in swc_known])  if swc_known else float("nan")

    q2_rate = 100.0 * (sum(1 for x in Q2_avail if x) / max(1, n))

    # Count short-circuits
    sc_count = sum(1 for r in recs if (r.get("timing", {}) or {}).get("short_circuit"))

    print("\n================ SUMMARY (FAST) ================")
    print(f"N instances (valid): {n}")
    print(f"Q2 short-circuited: {sc_count}/{n} ({100*sc_count/max(1,n):.0f}%)")

    if base_known:
        print(f"Text-only (B) accuracy: {100*base_acc:.1f}% | error: {100*(1-base_acc):.1f}% (known={len(base_known)})")
    else:
        print("Text-only (B) accuracy: NA")
    if rels_known:
        print(f"Text+relations (R) accuracy: {100*rels_acc:.1f}% | error: {100*(1-rels_acc):.1f}% (known={len(rels_known)})")
    else:
        print("Text+relations (R) accuracy: NA")
    if swc_known:
        print(f"Switch-policy accuracy: {100*sw_acc:.1f}% | error: {100*(1-sw_acc):.1f}% (known={len(swc_known)})")
    else:
        print("Switch-policy accuracy: NA")

    print(f"Q2 available rate: {q2_rate:.1f}%")
    print(f"Mean Q1: {np.nanmean(Q1):.3f} | Mean Q2: {np.nanmean(Q2):.3f} | Mean Q3: {np.nanmean(Q3):.3f} | Mean T: {np.nanmean(T):.3f}")
    print(f"Switch rate: {100*np.mean([1.0 if x else 0.0 for x in SW]):.1f}%")

    ts = compute_timing_summary(records)
    if ts.get("n", 0) > 0:
        ms = ts["means_sec"]
        print("\n---------------- TIMING (seconds) ----------------")
        print(f"Mean Q1 total: {ms['Q1']:.3f}")
        print(f"Mean Q2 total: {ms['Q2']:.3f}")
        print(f"Mean Q3 total: {ms['Q3']:.3f}")
        print(f"Mean decision pipeline (Q1+Q2+Q3): {ms['decision_pipeline']:.3f}")
        print(f"Mean instance total: {ms['instance_total']:.3f}")
        print(f"Mean baseline_symbolic_text_only call: {ms['baseline_symbolic_text_only']:.3f}")
        print(f"Mean text_only_with_relations call: {ms['text_only_with_relations']:.3f}")
        print(f"Mean switch policy eval: {ms['switch_policy_eval']:.6f}")
        print(f"Q2 short-circuited count: {ts.get('short_circuited_q2_count', 0)}")
        if ts.get("avg_instance_time_when_would_switch_sec", None) is not None:
            print(f"Avg instance time WHEN would_switch=True: {ts['avg_instance_time_when_would_switch_sec']:.3f} (n={ts['switch_count']})")
        else:
            print("Avg instance time WHEN would_switch=True: NA (no switches)")

        tok = ts.get("tokens", {}) or {}
        if tok.get("total_tokens", 0) > 0:
            print(f"\n---------------- TOKENS (Ollama) ----------------")
            print(f"Total prompt tokens:     {tok['total_prompt_tokens']:,}")
            print(f"Total completion tokens:  {tok['total_completion_tokens']:,}")
            print(f"Total tokens:            {tok['total_tokens']:,}")
            print(f"Mean prompt tok/instance: {tok['mean_prompt_tokens_per_instance']:,.0f}")
            print(f"Mean compl. tok/instance: {tok['mean_completion_tokens_per_instance']:,.0f}")

        print("-------------------------------------------------\n")
    print("======================================\n")


# ============================================================
# 16) PLOTS  (unchanged)
# ============================================================
def save_correlation_plots(records: List[Dict[str, Any]], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    recs = [r for r in records if isinstance(r, dict) and "scores" in r]
    x = list(range(len(recs)))

    B   = [_bool_to_float(r.get("lm_correct", {}).get("baseline_symbolic_text_only", None)) for r in recs]
    R   = [_bool_to_float(r.get("lm_correct", {}).get("text_only_with_relations", None)) for r in recs]
    Q1  = [float(r["Q1"]["q1_score"]) for r in recs]

    Q2 = []
    for r in recs:
        q2 = r.get("Q2", {}) or {}
        if bool(q2.get("available", False)) and (q2.get("q2_score", None) is not None):
            Q2.append(float(q2["q2_score"]))
        else:
            Q2.append(float("nan"))

    Q3  = [float(r["scores"]["complexity_score"]) for r in recs]
    T   = [float(r["scores"]["trustworthiness_score"]) for r in recs]

    def _newfig(title):
        plt.figure(figsize=(10, 4))
        plt.title(title, fontsize=11)
        plt.xlabel("instance", fontsize=9)
        plt.ylabel("score", fontsize=9)
        plt.ylim(-0.05, 1.05)

    def _legend_outside():
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0, fontsize=8, frameon=False)
        plt.tight_layout(rect=[0, 0, 0.82, 1])

    paths = {}

    _newfig("Scores: Q1, Q2, Complexity")
    plt.plot(x, Q1, label="Q1", linewidth=1.5)
    plt.plot(x, Q2, label="Q2", linewidth=1.5)
    plt.plot(x, Q3, label="Q3", linewidth=1.5)
    _legend_outside()
    p1 = os.path.join(out_dir, "plot_scores_q1_q2_q3.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight"); plt.close()
    paths["q1_q2_q3"] = p1

    _newfig("Trust vs Complexity")
    plt.plot(x, T,  label="T", linewidth=1.5)
    plt.plot(x, Q3, label="Q3", linewidth=1.5)
    _legend_outside()
    p2 = os.path.join(out_dir, "plot_trust_vs_complexity.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight"); plt.close()
    paths["trust_vs_complexity"] = p2

    _newfig("Accuracy vs Trust")
    plt.plot(x, B,  label="B", linewidth=1.5)
    plt.plot(x, R,  label="R", linewidth=1.5)
    plt.plot(x, T,  label="T", linewidth=1.5)
    _legend_outside()
    p3 = os.path.join(out_dir, "plot_accuracy_vs_trust.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight"); plt.close()
    paths["accuracy_vs_trust"] = p3

    return paths


# ============================================================
# 16B) COMPLEXITY-ONLY RERUN
#   Re-scores complexity on existing results WITHOUT re-running
#   any Ollama calls.  Only uses GPT for hard-language re-scoring.
#   Parallelised: all GPT HL calls run concurrently via ThreadPool.
# ============================================================
def rerun_complexity_only(
    input_json: str,
    output_json: Optional[str] = None,
    dataset_name: str = "stepgame",
    trust_noswitch_threshold: float = 0.70,
    complexity_noswitch_threshold: float = 0.55,
    rescore_hl: bool = True,
    rerun_q2: bool = True,
    llama_model: str = "qwen3:32b",
    temp: float = 0.7,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Read previously-saved evaluation records, recompute the complexity
    (Q3) score using the updated C3 v2 formula, optionally re-run Q2
    for previously short-circuited records, and re-evaluate the switch
    decision with (optionally different) thresholds.

    Parameters
    ----------
    input_json : str
        Path to the existing evaluation JSON (concatenated objects format).
    output_json : str or None
        Where to write updated records.  If None, writes to
        ``<input_json stem>_rescore.json`` in the same directory.
    rescore_hl : bool
        If True  -> call GPT again with the NEW HL prompt (captures ambiguity).
                     This is the recommended mode; GPT calls are cheap & parallel.
        If False -> reuse the precomputed HL items from the original run.
                     Faster (no GPT calls at all) but won't pick up ambiguity.
    rerun_q2 : bool
        If True  -> re-evaluate Q2 short-circuit with new thresholds and
                     run the 4 Ollama calls (3 paraphrase + 1 flip) for any
                     record that is no longer short-circuitable.
    llama_model : str
        Ollama model name for Q2 calls.
    temp : float
        Temperature for Ollama calls.
    seed : int
        Seed for reproducibility.
    """
    import time as _time

    t0 = _time.perf_counter()

    # ── Read records ──
    dec = json.JSONDecoder()
    raw_text = open(input_json, "r", encoding="utf-8").read().strip()
    records: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(raw_text):
        try:
            obj, end = dec.raw_decode(raw_text, pos)
            records.append(obj)
            pos = end
            while pos < len(raw_text) and raw_text[pos] in " \t\n\r":
                pos += 1
        except json.JSONDecodeError:
            break

    if not records:
        raise RuntimeError(
            f"Loaded 0 records from {input_json} "
            f"(file size: {len(raw_text)} chars). "
            f"Check the file path and contents."
        )
    print(f"Loaded {len(records)} records from {input_json}")

    # ── Step 1: re-score HL in parallel (GPT-only, fast) ──
    from concurrent.futures import ThreadPoolExecutor, as_completed
    hl_results: Dict[int, Dict[str, Any]] = {}

    if rescore_hl:
        # Verify API key is available before launching 250 GPT calls
        if not os.getenv("OPENAI_API_KEY", ""):
            raise RuntimeError(
                "rescore_hl=True but OPENAI_API_KEY is not set!\n"
                "Set it before calling: os.environ['OPENAI_API_KEY'] = 'sk-...'"
            )
        print("Re-scoring Hard Language with updated prompt (ambiguity-aware) ...")
        import time as _hl_time
        _hl_t0 = _hl_time.perf_counter()
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {}
            skipped = 0
            for i, rec in enumerate(records):
                q3_old = (rec.get("Q3", {}) or {}).get("complexity", {}) or {}
                details = q3_old.get("details", {}) or {}
                support_text = details.get("support_text_used_for_HL", "")
                if not support_text:
                    # Reconstruct from Q1 support sentences
                    q1 = rec.get("Q1", {}) or {}
                    # Try both key names (A_baseline_support or Q1A_baseline_support)
                    q1a = q1.get("A_baseline_support", {}) or q1.get("Q1A_baseline_support", {}) or {}
                    sups = q1a.get("support_sentences_matched", [])
                    if not sups:
                        parsed = q1a.get("parsed", {}) or {}
                        sups = parsed.get("support_sentences", [])
                    if not sups:
                        extracted = q1a.get("extracted", {}) or {}
                        sups = extracted.get("support_sentences", [])
                    support_text = " ".join(
                        s.strip() for s in (sups or []) if isinstance(s, str) and s.strip()
                    ).strip()
                if not support_text:
                    # Last resort: use the full story as support text
                    support_text = (rec.get("inputs", {}) or {}).get("story", "").strip()
                if support_text:
                    futures[pool.submit(extract_hard_language_scores, support_text, "gpt-5-mini")] = i
                else:
                    hl_results[i] = {"items": [], "timing_sec": 0.0}
                    skipped += 1

            print(f"  Submitted {len(futures)} GPT calls, skipped {skipped} (no support text)")

            done_count = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    hl_results[idx] = fut.result()
                except Exception as e:
                    hl_results[idx] = {"items": [], "timing_sec": 0.0}
                done_count += 1
                if done_count % 50 == 0:
                    print(f"  HL scored {done_count}/{len(futures)} ...")

        _hl_elapsed = _hl_time.perf_counter() - _hl_t0
        print(f"  HL re-scoring done ({len(hl_results)} records) in {_hl_elapsed:.1f}s")
    else:
        print("Reusing existing HL scores (no GPT calls)")

    # ── Step 2: recompute complexity for each record ──
    updated = 0
    for i, rec in enumerate(records):
        meta = rec.get("meta", {}) or {}
        k_hop = meta.get("k_hop", None)
        inputs = rec.get("inputs", {}) or {}
        story = inputs.get("story", "")
        question = inputs.get("question", "")

        # Get support sentences from Q1 (try both key names)
        q1 = rec.get("Q1", {}) or {}
        q1a = q1.get("A_baseline_support", {}) or q1.get("Q1A_baseline_support", {}) or {}
        support_sentences = q1a.get("support_sentences_matched", [])
        if not support_sentences:
            parsed = q1a.get("parsed", {}) or {}
            support_sentences = parsed.get("support_sentences", [])
        if not support_sentences:
            extracted = q1a.get("extracted", {}) or {}
            support_sentences = extracted.get("support_sentences", [])

        # Get structured relations from Q2/GPT unified
        q2 = rec.get("Q2", {}) or {}
        structured_rels = q2.get("relations_used", [])
        if not structured_rels:
            # Try from GPT unified bundle
            uq = q2.get("gpt_unified", {}) or {}
            bundle = uq.get("bundle", {}) or {}
            structured_rels = bundle.get("relations_direct", [])

        # Get precomputed HL
        if rescore_hl:
            precomputed_hl = hl_results.get(i, {"items": [], "timing_sec": 0.0})
        else:
            q3_old = (rec.get("Q3", {}) or {}).get("complexity", {}) or {}
            details = q3_old.get("details", {}) or {}
            old_items = details.get("hard_language_items", [])
            precomputed_hl = {"items": old_items or [], "timing_sec": 0.0}

        # Recompute
        new_complexity = compute_complexity(
            story=story,
            question=question,
            support_sentences=support_sentences,
            k_hop=k_hop,
            structured_relations_used=structured_rels,
            dataset_name=dataset_name,
            precomputed_hl=precomputed_hl,
        )

        new_q3 = float(new_complexity["complexity_score_0_1"])

        # Update record
        rec["Q3"] = {"complexity": new_complexity, "q3_score": new_q3}

        # Re-evaluate switch decision
        trust = float((rec.get("scores", {}) or {}).get("trustworthiness_score", 0.0))
        would_switch = not (new_q3 < complexity_noswitch_threshold and trust >= trust_noswitch_threshold)

        # Determine if switch decision is correct
        text_correct = (rec.get("lm_correct", {}) or {}).get("baseline_symbolic_text_only")
        grid_correct = rec.get("grid_correct")
        if would_switch:
            chosen_correct = grid_correct
        else:
            chosen_correct = text_correct

        rec["scores"]["complexity_score"] = new_q3
        rec["scores"]["would_switch"] = would_switch
        rec["scores"]["switch_policy_correct"] = chosen_correct
        rec["scores"]["switch_policy_rule"] = {
            "no_switch_if": f"(complexity < {complexity_noswitch_threshold}) AND (trust >= {trust_noswitch_threshold})",
            "switch_otherwise": True,
        }

        updated += 1

    # ── Step 2.5: re-run Q2 for previously short-circuited records ──
    q2_rerun_count = 0
    q2_rerun_success = 0
    if rerun_q2:
        print("\n── Q2 Rerun: checking which short-circuited records now need Q2 ──")
        for i, rec in enumerate(records):
            q2 = rec.get("Q2", {}) or {}
            was_sc = q2.get("short_circuited", False)
            if not was_sc:
                continue  # Q2 already ran during original pipeline

            # Check if Q2 data (variants, flip) is available
            uq = q2.get("q2_unified_relations_variants_via_gpt", {}) or {}
            bundle = uq.get("bundle", {}) or {}
            variants = bundle.get("variants", {})
            flip_cons = bundle.get("flip_consistency", {}) or {}
            direct_rels = q2.get("relations_direct_used", []) or []

            q2_available = bool(direct_rels)
            if not q2_available:
                continue  # Case A: no direct relations, Q2 stays unavailable

            # Re-evaluate short-circuit with NEW thresholds and NEW complexity
            new_q3 = float((rec.get("scores", {}) or {}).get("complexity_score", 0.0))
            q1 = rec.get("Q1", {}) or {}
            q1_score = float(q1.get("q1_score", 0.0))

            skip_q2 = False
            skip_reason = ""

            if new_q3 >= complexity_noswitch_threshold:
                skip_q2 = True
                skip_reason = f"C3={new_q3:.3f} >= tau_c={complexity_noswitch_threshold} -> always switch"
            else:
                t_max = 0.6 * q1_score + 0.4 * 1.0
                t_min = 0.6 * q1_score + 0.4 * 0.0

                if t_max < trust_noswitch_threshold:
                    skip_q2 = True
                    skip_reason = f"T_max={t_max:.3f} < tau_t={trust_noswitch_threshold} -> always switch"
                elif t_min >= trust_noswitch_threshold:
                    skip_q2 = True
                    skip_reason = f"T_min={t_min:.3f} >= tau_t={trust_noswitch_threshold} -> never switch"

            if skip_q2:
                # Still short-circuited under new thresholds; update reason
                q2["short_circuit_reason"] = skip_reason
                # Trust stays as Q1-only since Q2 is still skipped
                trust = q1_score
                rec["scores"]["trustworthiness_score"] = trust
                would_switch = not (new_q3 < complexity_noswitch_threshold and trust >= trust_noswitch_threshold)
                text_correct = (rec.get("lm_correct", {}) or {}).get("baseline_symbolic_text_only")
                grid_correct = rec.get("grid_correct")
                chosen_correct = grid_correct if would_switch else text_correct
                rec["scores"]["would_switch"] = would_switch
                rec["scores"]["switch_policy_correct"] = chosen_correct
                continue

            # ── This record is NO LONGER short-circuited → run Q2! ──
            q2_rerun_count += 1
            rid = (rec.get("meta", {}) or {}).get("dataset_id", f"idx_{i}")

            # Get Q1 answer for flip test
            q1a_answer = (q1.get("A_baseline_support", {}) or {}).get("extracted", {}).get("answer", "")
            if not q1a_answer:
                # Fallback to baseline selected_option
                q1a_answer = (rec.get("prompts_and_outputs", {}) or {}).get(
                    "baseline_symbolic_text_only", {}
                ).get("selected_option", "")
            q1a_answer = normalize_relation(q1a_answer or "")

            flipped_q = str(flip_cons.get("flipped_question", "") or "").strip()
            flip_applied = bool(flip_cons.get("flip_applied", False))

            # Get question from inputs
            question = (rec.get("inputs", {}) or {}).get("question", "")
            if not flipped_q:
                flipped_q = question

            # Run 3 paraphrase variants + 1 flip via Ollama (parallel)
            try:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                q2_answers = []
                q2_runs = []

                # Prepare all 4 prompts
                variant_jobs = []
                for level in ["simple", "hinted", "canonical"]:
                    v = variants.get(level, {})
                    s_text = v.get("support_text", "")
                    q_text = v.get("question", question)
                    if not s_text:
                        continue
                    p = prompt_q2_answer_from_text(s_text, q_text)
                    variant_jobs.append((level, s_text, q_text, p))

                flip_text = (variants.get("canonical", {}) or {}).get("support_text", "")
                if not flip_text:
                    flip_text = (variants.get("simple", {}) or {}).get("support_text", "")
                flip_prompt = prompt_q2_answer_from_text(flip_text, flipped_q)

                # Fire all calls in parallel
                with _TPE(max_workers=4) as _pool:
                    variant_futures = []
                    for level, s_text, q_text, p in variant_jobs:
                        fut = _pool.submit(_timed_call, call_ollama_llama, p, model=llama_model, temperature=temp, seed=seed)
                        variant_futures.append((level, s_text, q_text, p, fut))
                    fut_flip = _pool.submit(_timed_call, call_ollama_llama, flip_prompt, model=llama_model, temperature=temp, seed=seed)

                    # Collect variant results
                    for level, s_text, q_text, p, fut in variant_futures:
                        (rraw, rtok), dt_call = fut.result()
                        robj = extract_json_object(rraw)
                        ans = normalize_relation(robj.get("answer", ""))
                        just = str(robj.get("justification", "")).strip()
                        q2_answers.append(ans)
                        q2_runs.append({
                            "variant_level": level, "text_used": s_text, "question_used": q_text,
                            "prompt": p, "raw": rraw, "parsed": robj,
                            "extracted": {"answer": ans, "justification": just},
                            "timing_sec": float(dt_call),
                            "tokens": rtok,
                        })

                    # Collect flip result
                    (flip_raw, flip_tok), dt_flip = fut_flip.result()

                paraphrase_stability = stability_maxfreq(q2_answers) if q2_answers else 0.0

                # Flip test
                flip_obj = extract_json_object(flip_raw)
                flip_ans = normalize_relation(flip_obj.get("answer", ""))

                expected_flip = q1a_answer if (not flip_applied) else INVERSE_REL.get(q1a_answer, None)
                flip_score = 1.0 if (expected_flip is not None and flip_ans == expected_flip) else 0.0

                q2_score = float(np.mean([paraphrase_stability, flip_score]))

                # Update Q2 block
                q2.update({
                    "available": True,
                    "short_circuited": False,
                    "short_circuit_reason": "",
                    "q2_rerun": True,
                    "paraphrase_runs_3": q2_runs,
                    "answers_across_3": q2_answers,
                    "paraphrase_stability": paraphrase_stability,
                    "flip_test": {
                        "flipped_question": flipped_q,
                        "flip_applied": flip_applied,
                        "prompt": flip_prompt,
                        "raw": flip_raw,
                        "parsed": flip_obj,
                        "extracted": {"answer": flip_ans},
                        "expected_answer_given_flip_applied_rule": expected_flip,
                        "score": flip_score,
                    },
                    "q2_score": q2_score,
                    "components": {
                        "paraphrase_stability": paraphrase_stability,
                        "flip_score": flip_score,
                    },
                })

                # Update trust with blended formula: T = 0.6*Q1 + 0.4*Q2
                trust = 0.6 * q1_score + 0.4 * q2_score
                rec["scores"]["trustworthiness_score"] = trust
                rec["scores"]["q2_score"] = q2_score

                # Re-evaluate switch decision with new trust
                would_switch = not (new_q3 < complexity_noswitch_threshold and trust >= trust_noswitch_threshold)
                text_correct = (rec.get("lm_correct", {}) or {}).get("baseline_symbolic_text_only")
                grid_correct = rec.get("grid_correct")
                chosen_correct = grid_correct if would_switch else text_correct
                rec["scores"]["would_switch"] = would_switch
                rec["scores"]["switch_policy_correct"] = chosen_correct

                q2_rerun_success += 1
                print(f"  [{q2_rerun_success}/{q2_rerun_count}] {rid}: Q2={q2_score:.3f} "
                      f"(para={paraphrase_stability:.2f}, flip={flip_score:.1f}) "
                      f"trust={trust:.3f} switch={would_switch}")

                # Periodic save every 25 records
                if q2_rerun_success % 25 == 0:
                    _partial = output_json or f"{os.path.splitext(input_json)[0]}_rescore{os.path.splitext(input_json)[1]}"
                    with open(_partial, "w", encoding="utf-8") as _pf:
                        for _r in records:
                            _pf.write(json.dumps(_r, ensure_ascii=False, indent=2))
                            _pf.write("\n\n")
                    print(f"    [checkpoint saved to {_partial}]")

            except Exception as e:
                print(f"  ERROR re-running Q2 for {rid}: {e}")
                # Keep original trust (Q1-only)

        print(f"\nQ2 rerun complete: {q2_rerun_success}/{q2_rerun_count} records updated")

    # ── Step 3: write output ──
    if output_json is None:
        base, ext = os.path.splitext(input_json)
        output_json = f"{base}_rescore{ext}"

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        f.write("")
    for rec in records:
        with open(output_json, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=2))
            f.write("\n\n")

    elapsed = _time.perf_counter() - t0
    print(f"\nComplexity rescore complete: {updated} records in {elapsed:.1f}s")
    print(f"Saved to: {output_json}")

    # Quick summary
    n = len(records)
    sw = sum(1 for r in records if (r.get("scores", {}) or {}).get("would_switch"))
    correct = sum(1 for r in records if (r.get("scores", {}) or {}).get("switch_policy_correct"))
    c3_vals = [float((r.get("scores", {}) or {}).get("complexity_score", 0)) for r in records]
    print(f"Switch rate: {sw}/{n} ({100*sw/max(1,n):.1f}%)")
    print(f"Switch-policy accuracy: {correct}/{n} ({100*correct/max(1,n):.1f}%)")
    print(f"Mean C3: {sum(c3_vals)/max(1,len(c3_vals)):.3f}")
    print(f"Thresholds used: trust>={trust_noswitch_threshold}, complexity<{complexity_noswitch_threshold}")

    return records


# ============================================================
# 17) MAIN
# ============================================================
if __name__ == "__main__":

    # ── MODE SWITCH ──
    # Set to "full" for the full pipeline, or "rescore" to only
    # recompute complexity (no Ollama calls, only GPT for HL).
    MODE = "full"   # <--- change to "full" to run the entire pipeline

    if MODE == "rescore":
        # ── COMPLEXITY-ONLY RERUN ──
        # Reads existing results, re-scores C3 with the updated formula
        # (v2: CL + SD + ambiguity-aware HL), and re-evaluates switching.
        # Only makes GPT calls for HL (parallel, fast).
        # Set rescore_hl=False to skip even GPT calls (reuse old HL scores).
        INPUT_FILE = r"./metric/t1rust_eval_finalqwen3_32b (1).json"
        OUTPUT_FILE = r"./metric/t1rust_eval_finalqwen3_32b_v2.json"
        records = rerun_complexity_only(
            input_json=INPUT_FILE,
            output_json=OUTPUT_FILE,         # write to separate file, never overwrite input
            dataset_name="stepgame",
            trust_noswitch_threshold=0.70,
            complexity_noswitch_threshold=0.55,
            rescore_hl=False,   # True = call GPT with new ambiguity-aware prompt
            rerun_q2=True,     # True = re-run Q2 for records no longer short-circuited
            llama_model="qwen3:32b",
            temp=0.7,
            seed=42,
        )
        print_summary_stats(records)

    else:
        # ── FULL PIPELINE ──
        input_path = "./stepgame_switch_qwen8b_250_next.json"
        SKIP_EVAL = False
        SEED = 42

        records = run_dataset_one_pretty_json(
            input_path=input_path,
            output_json="./metric/trust_eval_finaqwen38bpart2.json",
            dataset_name="stepgame",
            llama_model="qwen3:8b",
            temp=0.7,
            limit=None,
            start_after_index=-1,
            trust_noswitch_threshold=0.70,
            complexity_noswitch_threshold=0.55,
            skip_eval_baselines=SKIP_EVAL,
            seed=SEED,
        )

        print_summary_stats(records)
        plots = save_correlation_plots(records, out_dir="shreya/metric/plots4qwen38pt2b")
        print("Saved plots:", plots)
