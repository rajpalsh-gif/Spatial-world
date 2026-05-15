
import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from utils.timing import _now, _secs, _timed_call
from utils.parsing import _clip01, extract_json_object, split_sentences, _norm_for_match
from utils.llm_clients import (
    call_gpt_nlp as call_gpt_5_1,
    call_ollama_logged as call_ollama_llama,
)
from utils.switch_helpers import (
    _submit,
    _bool_to_float,
    _safe_mean,
    relations_to_sentences_verbatim,
    match_support_sentences_to_story,
    build_ablated_story_remove_sentences,
)
from spartun.switch_prompts import (
    SEMANTICS_BLOCK,
    prompt_q1_baseline_and_support_sentences_spartun,
    prompt_answer_only_allow_dontknow_spartun,
    prompt_answer_forced_spartun,
    build_symbolic_QA_prompt_spartun,
    build_text_only_with_relations_prompt_from_inst_spartun,
    prompt_unified_paraphrase_and_flip,
    prompt_q2_answer_from_text_spartun,
    prompt_q2_answer_no_dontknow_spartun,
    prompt_flip_question_rewrite_spartun,
    prompt_negate_yesno_question_spartun,
    prompt_complexity_spartun_via_gpt,
)

# _submit / _bool_to_float / _safe_mean / relations_to_sentences_verbatim
# match_support_sentences_to_story / build_ablated_story_remove_sentences
# → all imported from utils/switch_helpers


# ============================================================
# 1) GENERAL NORMALIZATION / PARSING
# ============================================================

def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def normalize_answer_any(ans: str) -> str:
    a = _norm_space(ans).lower()
    a = a.replace("_", " ").replace("-", " ")
    a = re.sub(r"\s+", " ", a).strip()

    if a in {"yes", "y"}:
        return "yes"
    if a in {"no", "n"}:
        return "no"

    a = a.replace("dontknow", "dont know").replace("don't know", "dont know")
    a = re.sub(r"\s+", " ", a).strip()
    return a

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 2) SPaRTUN HELPERS (story/questions/objects/relations)
# ============================================================

def story_to_text(story_field: Any) -> str:
    if isinstance(story_field, list):
        return " ".join([str(s).strip() for s in story_field if str(s).strip()]).strip()
    return str(story_field or "").strip()

def objects_id_to_name(objects_info: Dict[str, Any], obj_id: str) -> str:
    oi = (objects_info or {}).get(str(obj_id), None)
    if isinstance(oi, dict):
        fn = str(oi.get("full_name", "")).strip()
        if fn:
            return fn
        gn = str(oi.get("general_name", "")).strip()
        if gn:
            return gn
    return str(obj_id)

def convert_relations_to_readable(
    relations: List[Dict[str, Any]],
    objects_info: Dict[str, Any]
) -> List[Dict[str, Any]]:
    out = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        hid = str(r.get("head_id", "")).strip()
        tid = str(r.get("tail_id", "")).strip()
        rel = normalize_answer_any(str(r.get("relation", "")).strip())
        if not hid or not tid or not rel:
            continue
        out.append({
            "head": objects_id_to_name(objects_info, hid),
            "tail": objects_id_to_name(objects_info, tid),
            "relation": rel,
            "head_id": hid,
            "tail_id": tid,
        })
    return out

# relations_to_sentences_verbatim → imported from utils/switch_helpers


# ============================================================
# 3) MATCH SUPPORT SENTENCES ROBUSTLY
# match_support_sentences_to_story    → from utils/switch_helpers
# build_ablated_story_remove_sentences → from utils/switch_helpers
# ============================================================

def match_selected_support_sentences(
    candidate_sentences: List[str],
    selected_sentences: List[str],
    max_sentences: int = 2,
) -> List[str]:
    cleaned_candidates = [s.strip() for s in (candidate_sentences or []) if isinstance(s, str) and s.strip()]
    cleaned_selected = [s.strip() for s in (selected_sentences or []) if isinstance(s, str) and s.strip()]

    if not cleaned_candidates:
        return []
    if not cleaned_selected:
        return cleaned_candidates[:max_sentences]

    candidate_norm = [_norm_for_match(s) for s in cleaned_candidates]
    matched = []
    for sentence in cleaned_selected:
        sent_norm = _norm_for_match(sentence)
        if not sent_norm:
            continue
        if sent_norm in candidate_norm:
            matched.append(cleaned_candidates[candidate_norm.index(sent_norm)])
            continue
        for idx, cand_norm in enumerate(candidate_norm):
            if sent_norm in cand_norm or cand_norm in sent_norm:
                matched.append(cleaned_candidates[idx])
                break

    out, seen = [], set()
    for sentence in matched:
        if sentence not in seen:
            out.append(sentence)
            seen.add(sentence)
        if len(out) >= max_sentences:
            break

    if not out:
        return cleaned_candidates[:max_sentences]
    return out[:max_sentences]



# ============================================================
# 6) Q2 FLIP EXPECTATION LOGIC (inverse vs same)
# ============================================================

INVERSE_REL = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "front": "behind", "behind": "front",
    "tpp": "tppi", "tppi": "tpp",
    "ntpp": "ntppi", "ntppi": "ntpp",
}
SAME_ON_FLIP = {"dc", "ec", "po", "near", "far"}  # symmetric under swap

def _parse_multi_answer(expected: str) -> set:
    """Parse expected answer which may be single or list like '["dc", "far"]'."""
    if not expected or not isinstance(expected, str):
        return set()
    e = expected.strip()
    # Try parsing as JSON list
    if e.startswith("["):
        try:
            items = json.loads(e)
            if isinstance(items, list):
                return {normalize_answer_any(str(x)) for x in items if str(x).strip()}
        except Exception:
            pass
    # Try comma-separated
    if "," in e:
        parts = [normalize_answer_any(p.strip()) for p in e.split(",") if p.strip()]
        if len(parts) > 1:
            return {p for p in parts if p}
    # Single answer
    n = normalize_answer_any(e)
    return {n} if n else set()

def expected_flipped_answer(original: str) -> Tuple[Optional[str], str]:
    o = normalize_answer_any(original)
    if o in INVERSE_REL:
        return INVERSE_REL[o], "inverse"
    if o in SAME_ON_FLIP:
        return o, "same"
    return None, "unsupported"


# ============================================================
# 7) SCORING HELPERS
# ============================================================

def _extract_selected_option_any(raw_or_dict: Any) -> str:
    if isinstance(raw_or_dict, dict):
        for k in ["selected_option", "answer"]:
            v = raw_or_dict.get(k, None)
            if isinstance(v, str) and v.strip():
                return normalize_answer_any(v)
        return ""

    if isinstance(raw_or_dict, str) and raw_or_dict.strip():
        obj = extract_json_object(raw_or_dict)
        if isinstance(obj, dict):
            for k in ["selected_option", "answer"]:
                v = obj.get(k, None)
                if isinstance(v, str) and v.strip():
                    return normalize_answer_any(v)
    return ""

def compute_correct_spartun(selected: str, gt: str, candidate_answers: List[str]) -> Optional[bool]:
    gt_n = normalize_answer_any(gt)
    sel_n = normalize_answer_any(selected)
    if not gt_n:
        return None

    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    cand_set = set(cand + ["dont know"])

    if sel_n not in cand_set:
        return False
    return bool(sel_n == gt_n)

def stability_maxfreq(answers: List[str]) -> float:
    cleaned = [normalize_answer_any(a) for a in answers if isinstance(a, str) and a.strip()]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return 0.0
    counts: Dict[str, int] = {}
    for a in cleaned:
        counts[a] = counts.get(a, 0) + 1
    maxf = max(counts.values())
    return float(maxf / len(cleaned))

def _bool_to_float(x: Optional[bool]) -> float:
    if x is True:
        return 1.0
    if x is False:
        return 0.0
    return 0.0

# ---------- NEW: YES/NO detection + inversion ----------
def _is_yesno_candidate_set(candidate_answers: List[str]) -> bool:
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    s = set(cand)
    return s.issubset({"yes", "no"}) and (("yes" in s) or ("no" in s))

def _invert_yesno(ans: str) -> Optional[str]:
    a = normalize_answer_any(ans)
    if a == "yes":
        return "no"
    if a == "no":
        return "yes"
    return None

# ---------- NEW: deterministic entity swap (no “didn’t really flip”) ----------
def deterministic_swap_entities_in_question(question: str, a_name: str, b_name: str) -> Optional[str]:
    """
    Swaps occurrences of a_name and b_name in question using placeholders (case-insensitive).
    Returns swapped question if both appear; else None.
    """
    if not question or not a_name or not b_name:
        return None

    q = question
    # find both (case-insensitive)
    if re.search(re.escape(a_name), q, flags=re.IGNORECASE) is None:
        return None
    if re.search(re.escape(b_name), q, flags=re.IGNORECASE) is None:
        return None

    A = "__ENT_A__"
    B = "__ENT_B__"

    q1 = re.sub(re.escape(a_name), A, q, flags=re.IGNORECASE)
    q2 = re.sub(re.escape(b_name), B, q1, flags=re.IGNORECASE)
    q3 = q2.replace(A, b_name).replace(B, a_name)
    return q3


# ============================================================
# 8) COMPLEXITY VIA GPT-5-mini (entities + hard language + coref burden)
#    FIX: calibrate difficulty ranges + post-process caps
# ============================================================

def _cap_difficulty(span: str, typ: str, diff: float) -> float:
    """
    Post-process to avoid runaway 0.9-ish scores unless truly warranted.
    This matches your preference: medium-range by default.
    """
    diff = _clip01(diff)
    s = (span or "").lower()

    is_clock = ("o'clock" in s) or re.search(r"\b\d+\s*o\s*clock\b", s) is not None

    if typ == "coref":
        # keep coref mostly <= ~0.55 unless it’s clearly chained (heuristic)
        chained = any(k in s for k in ["above", "below", "left", "right", "front", "behind"]) and ("box" in s)
        cap = 0.80 if chained else 0.55
        return float(min(diff, cap))

    if typ == "direction":
        # simple directions should not be huge
        cap = 0.70 if is_clock else 0.45
        return float(min(diff, cap))

    if typ == "clock":
        return float(min(diff, 0.80))

    return diff

def extract_complexity_features_spartun(
    story_text: str,
    question: str,
    support_sentences: List[str],
    gpt_model: str = "gpt-5-mini"
) -> Dict[str, Any]:
    prompt = prompt_complexity_spartun_via_gpt(story_text, question, support_sentences)
    raw, dt = _timed_call(call_gpt_5_1, prompt, model=gpt_model)
    parsed = extract_json_object(raw)

    try:
        num_entities = int(parsed.get("num_entities", 0))
    except Exception:
        num_entities = 0

    entities = parsed.get("entities", [])
    if not isinstance(entities, list):
        entities = []

    try:
        coref = float(parsed.get("coref_difficulty", 0.0))
    except Exception:
        coref = 0.0
    coref = _clip01(coref)

    hard = (parsed.get("hard_language", {}) or {})
    items = hard.get("items", [])
    if not isinstance(items, list):
        items = []

    clean_items = []
    ds = []
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
        if typ not in {"clock", "direction", "coref"}:
            typ = "direction"

        # cap/squash
        diff = _cap_difficulty(span, typ, diff)

        clean_items.append({"span": span, "type": typ, "difficulty": diff})
        ds.append(diff)

    # aggregate hard_language score (already capped at item-level)
    if ds:
        hard_score = _clip01(0.6 * max(ds) + 0.4 * (sum(ds) / len(ds)))
        hard_note = "ok"
    else:
        hard_score = 0.0
        hard_note = "no_items"

    # cap overall coref difficulty to avoid 0.9 spikes unless the model REALLY insists
    coref = float(min(coref, 0.70))

    return {
        "prompt": prompt,
        "raw": raw,
        "parsed": parsed,
        "timing_sec": float(dt),
        "features": {
            "num_entities": int(max(0, num_entities)),
            "entities": entities,
            "coref_difficulty": float(coref),
            "hard_language_items": clean_items,
            "hard_language_score": float(hard_score),
            "hard_language_status": hard_note,
        }
    }

def compute_complexity_spartun(
    story_text: str,
    question: str,
    support_sentences: List[str],
    gpt_model: str = "gpt-5-mini",
    reasoning_model_name: str = ""
) -> Dict[str, Any]:
    def saturating_ratio(x: float, c: float) -> float:
        x = float(max(0.0, x))
        c = float(max(1e-9, c))
        return float(x / (x + c))

    k_support = len([s for s in (support_sentences or []) if isinstance(s, str) and s.strip()])
    support_burden = saturating_ratio(k_support, c=5.0)

    g = extract_complexity_features_spartun(
        story_text=story_text,
        question=question,
        support_sentences=support_sentences,
        gpt_model=gpt_model
    )
    f = g["features"]
    num_entities = int(f.get("num_entities", 0))
    entity_load = saturating_ratio(float(num_entities), c=5.0)

    coref_raw = float(f.get("coref_difficulty", 0.0))
    hard_lang = float(f.get("hard_language_score", 0.0))

    coref_effective = _clip01(coref_raw)

    w_SB = 0.35
    w_EL = 0.10
    w_HL = 0.30
    w_CF = 0.25

    complexity_score = (
        w_SB * support_burden
        + w_EL * entity_load
        + w_HL * hard_lang
        + w_CF * coref_effective
    )
    complexity_score = _clip01(complexity_score)

    return {
        "components": {
            "support_burden_SB": float(support_burden),
            "entity_load_EL": float(entity_load),
            "hard_language_HL": float(hard_lang),
            "coref_difficulty_CF": float(coref_effective),
        },
        "weights": {"SB": w_SB, "EL": w_EL, "HL": w_HL, "CF": w_CF},
        "num_entities": int(num_entities),
        "k_support": int(k_support),
        "complexity_score_0_1": float(complexity_score),
        "gpt_details": {
            **g,
            "features": {
                **(g.get("features", {}) or {}),
                "coref_difficulty_raw": float(coref_raw),
                "coref_difficulty_effective": float(coref_effective),
                "reasoning_model_name": reasoning_model_name,
            }
        },
    }

def compute_flip_score(flip_ans: str, flip_expected: str,
                       gpt_expected: str = None,
                       computed_expected: str = None) -> float:
    """Soft flip score: accepts multiple expected answers (computed + GPT),
    handles list-form expected like '["dc", "far"]'."""
    if not flip_ans:
        return 0.0

    ans = normalize_answer_any(flip_ans)

    # Detect "don't know" variants
    is_dk = (
        "dont know" in ans or
        "don't know" in flip_ans.lower() or
        "do not know" in flip_ans.lower()
    )

    # Build set of all valid expected answers from all sources
    valid_expected = set()
    for exp in [flip_expected, gpt_expected, computed_expected]:
        if exp:
            valid_expected.update(_parse_multi_answer(str(exp)))
    valid_expected.discard("")
    valid_expected.discard("dont know")

    if not valid_expected:
        return 0.0

    if ans in valid_expected:
        return 1.0
    elif is_dk:
        return 0.5
    else:
        return 0.0

# ============================================================
# 9) ONE (SPaRTUN story,question) -> ONE RECORD
# ============================================================

def run_instance_one_record_spartun(
    inst: Dict[str, Any],
    llama_model: str,
    temp: float,
    dataset_name: str,
    trust_noswitch_threshold: float = 0.8,
    complexity_noswitch_threshold: float = 0.45,
    gpt_model: str = "gpt-5-mini",
) -> Dict[str, Any]:

    COMPLEXITY_CALIBRATION_SCALE = 1.0

    t_inst0 = _now()

    meta = inst.get("meta", {}) or {}
    dataset_id = str(meta.get("dataset_id", inst.get("dataset_id", "")))
    index = int(meta.get("index", inst.get("index", -1)))
    q_type = str(meta.get("q_type", inst.get("q_type", ""))).strip()
    asked_relation = normalize_answer_any(str(meta.get("asked_relation", inst.get("asked_relation", ""))).strip())

    story = str(inst.get("story", "")).strip()
    question = str(inst.get("question", "")).strip()
    # Accept BOTH naming conventions:
    # - instances produced by spartun_to_instances => label_ground_truth
    # - your "other json file" style              => ground_truth
    gt = str(inst.get("label_ground_truth", inst.get("ground_truth", ""))).strip()
    candidate_answers = inst.get("candidate_answers", []) or []

    prompts_and_outputs = inst.get("prompts_and_outputs", {}) or {}

    timing: Dict[str, Any] = {
        "baseline_symbolic_text_only_sec": None,
        "text_only_with_relations_sec": None,
        "Q1_total_sec": None,
        "Q2_total_sec": None,
        "Q3_total_sec": None,
        "switch_policy_eval_sec": None,
        "instance_total_sec": None,
        "decision_pipeline_sec": None,
        "details": {"Q1": {}, "Q2": {}, "Q3": {}}
    }

    # ================================================================
    # WAVE 1: Fire independent calls in parallel
    #   - Baseline (Ollama), Rels (Ollama), Q1-A (Ollama)
    #   Q1-A must finish before wave 2 (provides support_sentences)
    # ================================================================

    baseline_prompt = build_symbolic_QA_prompt_spartun(story, question, candidate_answers)
    rels_prompt = build_text_only_with_relations_prompt_from_inst_spartun(inst)
    q1a_prompt = prompt_q1_baseline_and_support_sentences_spartun(story, question, candidate_answers)

    fut_baseline = _submit(call_ollama_llama, baseline_prompt, model=llama_model, temperature=temp)
    fut_rels     = _submit(call_ollama_llama, rels_prompt, model=llama_model, temperature=temp)
    fut_q1a      = _submit(call_ollama_llama, q1a_prompt, model=llama_model, temperature=temp)

    # --- Collect baseline ---
    baseline_raw, dt_base = fut_baseline.result()
    baseline_parsed = extract_json_object(baseline_raw)
    baseline_selected = _extract_selected_option_any(baseline_parsed)

    # Retry once if baseline returned empty answer
    if not baseline_selected:
        print("  [RETRY] Baseline returned empty answer, re-querying...")
        baseline_raw_retry, dt_retry = _timed_call(call_ollama_llama, baseline_prompt, model=llama_model, temperature=temp)
        baseline_parsed_retry = extract_json_object(baseline_raw_retry)
        baseline_selected_retry = _extract_selected_option_any(baseline_parsed_retry)
        if baseline_selected_retry:
            baseline_raw = baseline_raw_retry
            baseline_parsed = baseline_parsed_retry
            baseline_selected = baseline_selected_retry
            dt_base += dt_retry

    baseline_correct = compute_correct_spartun(baseline_selected, gt, candidate_answers)
    timing["baseline_symbolic_text_only_sec"] = float(dt_base)

    prompts_and_outputs["baseline_symbolic_text_only"] = {
        "prompt": baseline_prompt,
        "lm_answer_raw": baseline_raw,
        "parsed": baseline_parsed,
        "selected_option": baseline_selected,
        "lm_correct": baseline_correct,
    }

    # --- Collect rels ---
    rels_raw, dt_rels = fut_rels.result()
    rels_parsed = extract_json_object(rels_raw)
    rels_selected = _extract_selected_option_any(rels_parsed)

    # Retry once if rels returned empty answer
    if not rels_selected:
        print("  [RETRY] Rels returned empty answer, re-querying...")
        rels_raw_retry, dt_retry = _timed_call(call_ollama_llama, rels_prompt, model=llama_model, temperature=temp)
        rels_parsed_retry = extract_json_object(rels_raw_retry)
        rels_selected_retry = _extract_selected_option_any(rels_parsed_retry)
        if rels_selected_retry:
            rels_raw = rels_raw_retry
            rels_parsed = rels_parsed_retry
            rels_selected = rels_selected_retry
            dt_rels += dt_retry

    rels_correct = compute_correct_spartun(rels_selected, gt, candidate_answers)
    timing["text_only_with_relations_sec"] = float(dt_rels)

    prompts_and_outputs["text_only_with_relations"] = {
        "prompt": rels_prompt,
        "lm_answer_raw": rels_raw,
        "parsed": rels_parsed,
        "selected_option": rels_selected,
        "lm_correct": rels_correct,
    }

    # --- Collect Q1-A (needed before everything else) ---
    tQ10 = _now()

    q1a_raw, dt_q1a = fut_q1a.result()
    q1a_obj = extract_json_object(q1a_raw)

    q1_answer = normalize_answer_any(str(q1a_obj.get("answer", "")).strip())

    # Retry once if model returned empty answer
    if not q1_answer:
        print("  [RETRY] Q1-A returned empty answer, re-querying...")
        q1a_raw_retry, dt_retry = _timed_call(call_ollama_llama, q1a_prompt, model=llama_model, temperature=temp)
        q1a_obj_retry = extract_json_object(q1a_raw_retry)
        q1_answer_retry = normalize_answer_any(str(q1a_obj_retry.get("answer", "")).strip())
        if q1_answer_retry:
            q1a_raw = q1a_raw_retry
            q1a_obj = q1a_obj_retry
            q1_answer = q1_answer_retry
            dt_q1a += dt_retry
    support_sentences = q1a_obj.get("support_sentences", [])
    if not isinstance(support_sentences, list):
        support_sentences = []
    support_sentences = match_support_sentences_to_story(story, support_sentences)

    if not support_sentences:
        story_sents = split_sentences(story)
        support_sentences = story_sents[:2] if story_sents else [story]

    q1_just = str(q1a_obj.get("justification", "")).strip()

    support_only_text = " ".join([s.strip() for s in support_sentences if isinstance(s, str) and s.strip()]).strip()

    # ================================================================
    # WAVE 2: Fire Q1-S, Q1-C, Unified GPT, Q3 in parallel
    #   All depend on support_sentences from Q1-A
    # ================================================================

    q1s_prompt = prompt_answer_forced_spartun(support_only_text, question, candidate_answers)
    ablated_story, ablation_info = build_ablated_story_remove_sentences(story, support_sentences)
    q1c_prompt = prompt_answer_only_allow_dontknow_spartun(ablated_story, question, candidate_answers)
    unified_prompt = prompt_unified_paraphrase_and_flip(support_sentences, question, story_text=story)

    fut_q1s     = _submit(call_ollama_llama, q1s_prompt, model=llama_model, temperature=temp)
    fut_q1c     = _submit(call_ollama_llama, q1c_prompt, model=llama_model, temperature=temp)
    fut_unified = _submit(call_gpt_5_1, unified_prompt, model=gpt_model)
    fut_q3      = _submit(compute_complexity_spartun,
        story_text=story, question=question,
        support_sentences=support_sentences,
        gpt_model=gpt_model, reasoning_model_name=llama_model)

    # --- Collect Q1-S ---
    q1s_raw, dt_q1s = fut_q1s.result()
    q1s_obj = extract_json_object(q1s_raw)
    q1s_answer = normalize_answer_any(str(q1s_obj.get("answer", "")).strip())
    q1s_just = str(q1s_obj.get("justification", "")).strip()
    q1_support_only_score = 1.0 if (q1s_answer and q1_answer and q1s_answer == q1_answer) else 0.0

    # --- Collect Q1-C ---
    q1c_raw, dt_q1c = fut_q1c.result()
    q1c_obj = extract_json_object(q1c_raw)
    ablated_answer = normalize_answer_any(str(q1c_obj.get("answer", "")).strip())
    ablated_just = str(q1c_obj.get("justification", "")).strip()
    q1_minus_support_score = 1.0 if (ablated_answer == "dont know") else 0.0

    q1_score = float(np.mean([q1_support_only_score, q1_minus_support_score]))

    tQ11 = _now()
    timing["details"]["Q1"] = {
        "Q1A_sec": float(dt_q1a),
        "Q1S_sec": float(dt_q1s),
        "Q1C_sec": float(dt_q1c),
        "Q1_total_sec": float(_secs(tQ10, tQ11)),
    }
    timing["Q1_total_sec"] = float(_secs(tQ10, tQ11))

    # --- Collect Unified GPT from wave 2 (needed before Q2 Ollama calls) ---
    unified_raw, dt_unified = fut_unified.result()
    unified_obj = extract_json_object(unified_raw)

    def _get3(obj: Dict[str, Any], k: str, fallback: str) -> Dict[str, str]:
        block = obj.get(k, {})
        if not isinstance(block, dict):
            block = {}
        return {
            "simple": str(block.get("simple", "")).strip() or fallback,
            "hinted": str(block.get("hinted", "")).strip() or fallback,
            "canonical": str(block.get("canonical", "")).strip() or fallback,
        }

    support_3 = _get3(unified_obj, "support", support_only_text)
    question_3 = _get3(unified_obj, "question", question)
    unified_flip_question = str(unified_obj.get("flipped_question", "")).strip()
    unified_flip_type = str(unified_obj.get("flip_type", "none")).strip().lower()
    flip_support_sentences = match_selected_support_sentences(
        support_sentences,
        unified_obj.get("flip_support_sentences", []),
        max_sentences=2,
    )
    flip_support_text = " ".join(flip_support_sentences).strip() or support_3["canonical"]
    raw_flip_expected = unified_obj.get("flip_expected_answer", "")
    if isinstance(raw_flip_expected, list):
        gpt_flip_expected_answer = json.dumps([normalize_answer_any(str(x)) for x in raw_flip_expected])
    else:
        gpt_flip_expected_answer = normalize_answer_any(str(raw_flip_expected).strip())

    # ================================================================
    # WAVE 3: Fire Q2 paraphrase x3 + flip in parallel
    #   Depends on unified GPT output (paraphrases + flip question)
    # ================================================================

    tQ20 = _now()

    q2_timing_detail: Dict[str, Any] = {
        "unified_gpt_sec": float(dt_unified),
        "q2_3_ollama_calls_sec_total": 0.0,
        "q2_3_ollama_calls_sec_each": [],
        "flip_ollama_sec": None,
        "Q2_total_sec": None,
    }

    # Submit Q2 paraphrase x3
    q2_para_futures = []
    for level in ["simple", "hinted", "canonical"]:
        p = prompt_q2_answer_no_dontknow_spartun(support_3[level], question_3[level], candidate_answers)
        q2_para_futures.append((level, support_3[level], question_3[level], p,
            _submit(call_ollama_llama, p, model=llama_model, temperature=temp)))

    # Submit Q2 flip
    is_yesno = (_is_yesno_candidate_set(candidate_answers) or (q_type.lower() in {"yn", "yesno", "yes/no"}))
    flip_q = unified_flip_question
    fut_flip = None
    flip_prompt = None
    if flip_q:
        flip_prompt = prompt_q2_answer_from_text_spartun(story, flip_q, candidate_answers)
        fut_flip = _submit(call_ollama_llama, flip_prompt, model=llama_model, temperature=temp)

    # --- Collect Q2 paraphrase runs ---
    q2_runs = []
    q2_answers = []
    for level, s_text, q_text, p, fut in q2_para_futures:
        rraw, dt_call = fut.result()
        q2_timing_detail["q2_3_ollama_calls_sec_each"].append(float(dt_call))
        q2_timing_detail["q2_3_ollama_calls_sec_total"] += float(dt_call)

        robj = extract_json_object(rraw)
        ans = normalize_answer_any(str(robj.get("answer", "")).strip())
        just = str(robj.get("justification", "")).strip()

        q2_answers.append(ans)
        q2_runs.append({
            "level": level,
            "support_text_used": s_text,
            "question_used": q_text,
            "prompt": p,
            "raw": rraw,
            "parsed": robj,
            "extracted": {"answer": ans, "justification": just},
            "timing_sec": float(dt_call),
        })

    paraphrase_stability = stability_maxfreq(q2_answers)

    # --- Collect FLIP TEST ---
    flip_block = {"supported": False, "score": 0.0, "expectation_type": "unsupported"}
    canonical_ans = q2_answers[-1] if q2_answers else ""

    if fut_flip is not None:
        flip_raw, dt_flip_oll = fut_flip.result()
        q2_timing_detail["flip_ollama_sec"] = float(dt_flip_oll)
        flip_obj = extract_json_object(flip_raw)
        flip_ans = normalize_answer_any(str(flip_obj.get("answer", "")).strip())

        if is_yesno and canonical_ans in {"yes", "no"}:
            computed_expected = _invert_yesno(canonical_ans)
            expected = gpt_flip_expected_answer if gpt_flip_expected_answer and gpt_flip_expected_answer != "dont know" else computed_expected
            flip_score = compute_flip_score(flip_ans, expected,
                                            gpt_expected=gpt_flip_expected_answer,
                                            computed_expected=computed_expected)
            flip_block = {
                "supported": True,
                "flip_type": "yesno_negation",
                "expectation_type": "inverse",
                "expected_answer_after_flip": expected,
                "gpt_expected_answer": gpt_flip_expected_answer,
                "flip_question": flip_q,
                "support_sentences_used": flip_support_sentences,
                "support_text_used": flip_support_text,
                "answer_prompt": flip_prompt,
                "raw": flip_raw,
                "parsed": flip_obj,
                "extracted": {"answer": flip_ans},
                "score": flip_score,
            }
        else:
            computed_flip_ans, flip_expect_type = expected_flipped_answer(canonical_ans)
            expected_ans = gpt_flip_expected_answer if gpt_flip_expected_answer and gpt_flip_expected_answer != "dont know" else computed_flip_ans
            if expected_ans is not None or computed_flip_ans is not None or gpt_flip_expected_answer:
                flip_score = compute_flip_score(
                    flip_ans, expected_ans,
                    gpt_expected=gpt_flip_expected_answer,
                    computed_expected=computed_flip_ans
                )
                flip_block = {
                    "supported": True,
                    "flip_type": "entity_swap",
                    "expectation_type": flip_expect_type,
                    "expected_answer_after_flip": expected_ans,
                    "gpt_expected_answer": gpt_flip_expected_answer,
                    "computed_expected_answer": computed_flip_ans,
                    "flip_question": flip_q,
                    "support_sentences_used": flip_support_sentences,
                    "support_text_used": flip_support_text,
                    "answer_prompt": flip_prompt,
                    "raw": flip_raw,
                    "parsed": flip_obj,
                    "extracted": {"answer": flip_ans},
                    "score": flip_score,
                }

    # Q2 score rule
    if flip_block.get("supported", False):
        q2_score = float(np.mean([paraphrase_stability, float(flip_block.get("score", 0.0))]))
        q2_components = {"paraphrase_stability": paraphrase_stability, "flip_score": float(flip_block.get("score", 0.0))}
    else:
        q2_score = float(paraphrase_stability)
        q2_components = {"paraphrase_stability": paraphrase_stability, "flip_score": None}

    # Penalize Q2 if flip answer mirrors paraphrase majority when it shouldn't
    _flip_penalty_applied = 0.0
    if flip_block.get("supported") and q2_answers:
        flip_ans_val = normalize_answer_any(str((flip_block.get("extracted") or {}).get("answer", "")))
        para_majority = max(set(q2_answers), key=q2_answers.count) if q2_answers else ""
        flip_type_val = flip_block.get("flip_type", "")
        if flip_ans_val and para_majority and flip_ans_val == para_majority:
            should_differ = False
            if flip_type_val == "yesno_negation":
                should_differ = True
            elif flip_type_val == "entity_swap" and para_majority not in SAME_ON_FLIP:
                should_differ = True
            if should_differ:
                _flip_penalty_applied = 0.10
                q2_score = max(0.0, q2_score - _flip_penalty_applied)
    q2_components["flip_same_as_para_penalty"] = _flip_penalty_applied

    tQ21 = _now()
    q2_timing_detail["Q2_total_sec"] = float(_secs(tQ20, tQ21))
    timing["details"]["Q2"] = q2_timing_detail
    timing["Q2_total_sec"] = float(_secs(tQ20, tQ21))

    Q2_block = {
        "available": True,
        "rule": "Q2 computed with 3 paraphrase-level calls; flip question is generated in same GPT call as paraphrases (no separate flip GPT call).",
        "support_sentences_used": support_sentences,
        "unified_generation": {
            "prompt": unified_prompt,
            "raw": unified_raw,
            "parsed": unified_obj,
            "support_3": support_3,
            "question_3": question_3,
            "flip_question": unified_flip_question,
            "flip_type": unified_flip_type,
            "timing_sec": float(dt_unified),
        },
        "paraphrase_runs": q2_runs,
        "answers_across_3": q2_answers,
        "paraphrase_stability": paraphrase_stability,
        "flip_test": flip_block,
        "q2_score": q2_score,
        "components": q2_components,
    }

    # --- Collect Q3 complexity ---
    tQ30 = _now()
    complexity, dt_q3 = fut_q3.result()
    q3_raw = float(complexity["complexity_score_0_1"])
    q3 = _clip01(q3_raw * COMPLEXITY_CALIBRATION_SCALE)
    tQ31 = _now()

    timing["details"]["Q3"] = {
        "Q3_total_sec": float(_secs(tQ30, tQ31)),
        "complexity_gpt_sec": float((complexity.get("gpt_details", {}) or {}).get("timing_sec", 0.0)),
        "complexity_raw": q3_raw,
        "complexity_calibration_scale": COMPLEXITY_CALIBRATION_SCALE,
        "complexity_calibrated": q3,
    }
    timing["Q3_total_sec"] = float(_secs(tQ30, tQ31))

    trust = 0.60 * q1_score + 0.40 * float(q2_score)
    trust_weights_effective = {"Q1": 0.60, "Q2": 0.40}

    tSW0 = _now()
    would_switch = not (q3 < complexity_noswitch_threshold and trust >= trust_noswitch_threshold)
    chosen_correct = rels_correct if would_switch else baseline_correct
    tSW1 = _now()
    timing["switch_policy_eval_sec"] = float(_secs(tSW0, tSW1))

    t_inst1 = _now()
    timing["instance_total_sec"] = float(_secs(t_inst0, t_inst1))
    timing["decision_pipeline_sec"] = float(
        (timing["Q1_total_sec"] or 0.0) + (timing["Q2_total_sec"] or 0.0) + (timing["Q3_total_sec"] or 0.0)
    )

    return {
        "meta": {
            "dataset_id": dataset_id,
            "index": index,
            "dataset_name": dataset_name,
            "q_type": q_type,
            "asked_relation": asked_relation,
        },
        "ground_truth": normalize_answer_any(gt),
        "inputs": {"story": story, "question": question},
        "candidate_answers": [normalize_answer_any(x) for x in (candidate_answers or [])],

        "timing": timing,

        "lm_correct": {
            "baseline_symbolic_text_only": baseline_correct,
            "text_only_with_relations": rels_correct,
        },

        "prompts_and_outputs": prompts_and_outputs,

        "Q1": {
            "A_baseline_support": {
                "prompt": q1a_prompt,
                "raw": q1a_raw,
                "parsed": q1a_obj,
                "extracted": {"answer": q1_answer, "support_sentences": support_sentences, "justification": q1_just},
            },
            "S_support_only": {
                "support_only_text": support_only_text,
                "prompt": q1s_prompt,
                "raw": q1s_raw,
                "parsed": q1s_obj,
                "extracted": {"answer": q1s_answer, "justification": q1s_just},
                "score": q1_support_only_score,
            },
            "C_story_minus_support": {
                "ablation_info": ablation_info,
                "ablated_story": ablated_story,
                "prompt": q1c_prompt,
                "raw": q1c_raw,
                "parsed": q1c_obj,
                "extracted": {"answer": ablated_answer, "justification": ablated_just},
                "score": q1_minus_support_score,
                "strict_rule": "score=1 only if answer == 'dont know'",
            },
            "q1_score": q1_score,
        },

        "Q2": Q2_block,

        "Q3": {"complexity": complexity, "q3_score_raw": q3_raw, "q3_score": q3},

        "scores": {
            "trustworthiness_score": float(trust),
            "complexity_score": float(q3),
            "would_switch": bool(would_switch),
            "switch_policy_correct": chosen_correct,
            "switch_policy_rule": {
                "no_switch_if": f"(complexity < {complexity_noswitch_threshold}) AND (trust > {trust_noswitch_threshold})",
                "switch_otherwise": True
            },
            "weights_effective": {"trust": trust_weights_effective},
            "thresholds": {
                "trust_noswitch_threshold": float(trust_noswitch_threshold),
                "complexity_noswitch_threshold": float(complexity_noswitch_threshold),
                "complexity_calibration_scale": float(COMPLEXITY_CALIBRATION_SCALE),
            }
        }
    }

from collections import OrderedDict

def group_instances_by_dataset_id(instances: List[Dict[str, Any]]) -> "OrderedDict[str, List[Dict[str, Any]]]":
    grouped = OrderedDict()
    for inst in instances:
        dsid = str((inst.get("meta", {}) or {}).get("dataset_id", ""))
        if dsid not in grouped:
            grouped[dsid] = []
        grouped[dsid].append(inst)
    return grouped

# ============================================================
# 10) DATASET LOADER: SPaRTUN -> instances
# ============================================================

def spartun_to_instances(spartun_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    data = spartun_json.get("data", [])
    if not isinstance(data, list):
        return []

    instances: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        identifier = str(item.get("identifier", "")).strip()
        story_text = story_to_text(item.get("story", ""))

        objects_info = item.get("objects_info", {}) or {}
        readable_relations = convert_relations_to_readable(item.get("relations", []) or [], objects_info)

        questions = item.get("questions", []) or []
        for q in questions:
            if not isinstance(q, dict):
                continue
            q_id = int(q.get("q_id", -1))
            q_type = str(q.get("q_type", "")).strip()
            question_text = str(q.get("question", "")).strip()
            candidate_answers = q.get("candidate_answers", []) or []
            answer_list = q.get("answer", []) or []
            gt = str(answer_list[0]).strip() if isinstance(answer_list, list) and answer_list else str(q.get("answer", "")).strip()

            qi = q.get("question_info", {}) or {}
            asked = qi.get("asked_relation", "")
            if isinstance(asked, list) and asked:
                asked = asked[0]
            asked_relation = str(asked).strip()

            query_ids = q.get("query", None)
            a_name = ""
            b_name = ""
            if isinstance(query_ids, list) and len(query_ids) == 2:
                a_name = objects_id_to_name(objects_info, str(query_ids[0]))
                b_name = objects_id_to_name(objects_info, str(query_ids[1]))

            instances.append({
                "meta": {
                    "dataset_id": identifier,
                    "index": q_id,
                    "q_type": q_type,
                    "asked_relation": asked_relation,
                    "k_hop": qi.get("reasoning_steps", None),
                },
                "story": story_text,
                "question": question_text,
                "label_ground_truth": gt,
                "candidate_answers": candidate_answers,
                "relations": readable_relations,

                "query_ids": query_ids if isinstance(query_ids, list) else None,
                "query_entity_a_name": a_name,
                "query_entity_b_name": b_name,
            })
    return instances


# NEW: accept your "already-pretty" list-of-records JSON and re-run the pipeline on it
def pretty_records_to_instances(pretty_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    instances: List[Dict[str, Any]] = []
    for item in pretty_list or []:
        if not isinstance(item, dict):
            continue
        meta = item.get("meta", {}) or {}
        inputs = item.get("inputs", {}) or {}
        story = str(inputs.get("story", item.get("story", "")) or "").strip()
        question = str(inputs.get("question", item.get("question", "")) or "").strip()
        gt = str(item.get("ground_truth", item.get("label_ground_truth", "")) or "").strip()
        cand = item.get("candidate_answers", []) or []
        rels = (item.get("relations", None) or [])
        instances.append({
            "meta": {
                "dataset_id": str(meta.get("dataset_id", item.get("dataset_id", ""))),
                "index": int(meta.get("index", item.get("index", -1))),
                "q_type": str(meta.get("q_type", item.get("q_type", ""))).strip(),
                "asked_relation": str(meta.get("asked_relation", item.get("asked_relation", ""))).strip(),
                "k_hop": meta.get("k_hop", None),
            },
            "story": story,
            "question": question,
            "ground_truth": gt,                 # <- handled in run_instance_one_record_spartun()
            "label_ground_truth": gt,           # <- also set for compatibility
            "candidate_answers": cand,
            "relations": rels,
            "query_ids": item.get("query_ids", None),
            "query_entity_a_name": item.get("query_entity_a_name", ""),
            "query_entity_b_name": item.get("query_entity_b_name", ""),
        })
    return instances


# ============================================================
# 11) TIMING SUMMARY
# ============================================================
# _safe_mean imported from utils/switch_helpers

def compute_timing_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    recs = [r for r in records if isinstance(r, dict) and "timing" in r and "scores" in r]
    if not recs:
        return {"n": 0, "error": "no timing found"}

    q1, q2, q3, inst, decision, base, rels, switch_eval, switch_only_inst_times = ([] for _ in range(9))

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

        if bool((r.get("scores", {}) or {}).get("would_switch", False)):
            switch_only_inst_times.append(float(t.get("instance_total_sec") or 0.0))

    return {
        "n": len(recs),
        "totals_sec": {
            "Q1_total": float(sum(q1)),
            "Q2_total": float(sum(q2)),
            "Q3_total": float(sum(q3)),
            "decision_pipeline_total": float(sum(decision)),
            "instance_total": float(sum(inst)),
        },
        "means_sec": {
            "Q1": _safe_mean(q1),
            "Q2": _safe_mean(q2),
            "Q3": _safe_mean(q3),
            "decision_pipeline": _safe_mean(decision),
            "instance_total": _safe_mean(inst),
            "baseline_symbolic_text_only": _safe_mean(base),
            "text_only_with_relations": _safe_mean(rels),
            "switch_policy_eval": _safe_mean(switch_eval),
        },
        "avg_instance_time_when_would_switch_sec": _safe_mean(switch_only_inst_times) if switch_only_inst_times else None,
        "switch_count": int(len(switch_only_inst_times)),
    }


# ============================================================
# 12) RUN DATASET -> ONE PRETTY JSON FILE
# ============================================================

def run_spartun_dataset_one_pretty_json(
    input_path: str,
    output_json: str,
    dataset_name: str,
    llama_model: str = "qwen3:32b",
    temp: float = 0.7,
    limit: Optional[int] = None,
    limit_contexts: Optional[int] = None,
    skip_contexts: int = 0,
    start_after_qid: int = -1,
    trust_noswitch_threshold: float = 0.8,
    complexity_noswitch_threshold: float = 0.45,
    gpt_model: str = "gpt-5-mini",
) -> List[Dict[str, Any]]:

    t0 = _now()
    raw = read_json(input_path)

    if isinstance(raw, dict):
        instances = spartun_to_instances(raw)
    elif isinstance(raw, list):
        instances = pretty_records_to_instances(raw)
    else:
        raise ValueError(f"Unsupported input JSON type: {type(raw)}")

    # -------- NEW: limit by contexts (dataset_id) ----------
    grouped = group_instances_by_dataset_id(instances)
    dsids = list(grouped.keys())
    total_contexts = len(dsids)

    if skip_contexts > 0:
        dsids = dsids[skip_contexts:]
    if isinstance(limit_contexts, int):
        dsids = dsids[:max(0, limit_contexts)]

    print(f"  Contexts: {len(dsids)} selected (skipped {skip_contexts}, total {total_contexts})")

    filtered: List[Dict[str, Any]] = []
    for dsid in dsids:
        for inst in grouped[dsid]:
            idx = int((inst.get("meta", {}) or {}).get("index", -1))
            if idx > start_after_qid:
                filtered.append(inst)

    # -------- OLD behavior still available: limit by QUESTIONS ----------
    if isinstance(limit, int) and (limit_contexts is None):
        filtered = filtered[:limit]

    records: List[Dict[str, Any]] = []
    for inst in filtered:
        try:
            rec = run_instance_one_record_spartun(
                inst,
                llama_model=llama_model,
                temp=temp,
                dataset_name=dataset_name,
                trust_noswitch_threshold=trust_noswitch_threshold,
                complexity_noswitch_threshold=complexity_noswitch_threshold,
                gpt_model=gpt_model,
            )
            records.append(rec)
        except Exception as e:
            records.append({"meta": inst.get("meta", {}), "error": str(e)})

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

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
                "thresholds": {
                    "trust_noswitch_threshold": float(trust_noswitch_threshold),
                    "complexity_noswitch_threshold": float(complexity_noswitch_threshold),
                }
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved ONE pretty JSON: {output_json} (n={len(records)})")
    print(f"Saved timing summary: {timing_summary_path}")
    return records

        



# ============================================================
# 13) SUMMARY STATS
# ============================================================

def print_summary_stats(records: List[Dict[str, Any]]) -> None:
    recs = [r for r in records if isinstance(r, dict) and "scores" in r and "lm_correct" in r]
    n = len(recs)
    if n == 0:
        print("No valid records to summarize.")
        return

    B = [r.get("lm_correct", {}).get("baseline_symbolic_text_only", None) for r in recs]
    R = [r.get("lm_correct", {}).get("text_only_with_relations", None) for r in recs]

    Q1 = [float((r.get("Q1", {}) or {}).get("q1_score", 0.0)) for r in recs]
    Q2 = [float((r.get("Q2", {}) or {}).get("q2_score", 0.0)) for r in recs]
    Q3 = [float((r.get("scores", {}) or {}).get("complexity_score", 0.0)) for r in recs]
    T  = [float((r.get("scores", {}) or {}).get("trustworthiness_score", 0.0)) for r in recs]

    SW  = [bool((r.get("scores", {}) or {}).get("would_switch", False)) for r in recs]
    SWC = [r.get("scores", {}).get("switch_policy_correct", None) for r in recs]

    base_known = [b for b in B if b is not None]
    rels_known = [b for b in R if b is not None]
    swc_known  = [b for b in SWC if b is not None]

    base_acc = np.mean([1.0 if b else 0.0 for b in base_known]) if base_known else float("nan")
    rels_acc = np.mean([1.0 if b else 0.0 for b in rels_known]) if rels_known else float("nan")
    sw_acc   = np.mean([1.0 if b else 0.0 for b in swc_known])  if swc_known else float("nan")

    print("\n================ SUMMARY ================")
    print(f"N instances (valid): {n}")
    print(f"Text-only (B) accuracy: {100*base_acc:.1f}% (known={len(base_known)})" if base_known else "Text-only (B) accuracy: NA")
    print(f"Relations-only (R) accuracy: {100*rels_acc:.1f}% (known={len(rels_known)})" if rels_known else "Relations-only (R) accuracy: NA")
    print(f"Switch-policy accuracy: {100*sw_acc:.1f}% (known={len(swc_known)})" if swc_known else "Switch-policy accuracy: NA")
    print(f"Mean Q1: {np.mean(Q1):.3f} | Mean Q2: {np.mean(Q2):.3f} | Mean Q3: {np.mean(Q3):.3f} | Mean T: {np.mean(T):.3f}")
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
        print(f"Mean baseline_text call: {ms['baseline_symbolic_text_only']:.3f}")
        print(f"Mean relations call: {ms['text_only_with_relations']:.3f}")
        print(f"Mean switch policy eval: {ms['switch_policy_eval']:.6f}")
        if ts.get("avg_instance_time_when_would_switch_sec", None) is not None:
            print(f"Avg instance time WHEN would_switch=True: {ts['avg_instance_time_when_would_switch_sec']:.3f} (n={ts['switch_count']})")
        else:
            print("Avg instance time WHEN would_switch=True: NA (no switches)")
        print("-------------------------------------------------\n")

    print("======================================\n")


# ============================================================
# 14) PLOTS
# ============================================================

def save_correlation_plots(records: List[Dict[str, Any]], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    recs = [r for r in records if isinstance(r, dict) and "scores" in r]
    x = list(range(len(recs)))

    B = [_bool_to_float(r.get("lm_correct", {}).get("baseline_symbolic_text_only", None)) for r in recs]
    R = [_bool_to_float(r.get("lm_correct", {}).get("text_only_with_relations", None)) for r in recs]
    Q1 = [float((r.get("Q1", {}) or {}).get("q1_score", 0.0)) for r in recs]
    Q2 = [float((r.get("Q2", {}) or {}).get("q2_score", 0.0)) for r in recs]
    Q3 = [float((r.get("scores", {}) or {}).get("complexity_score", 0.0)) for r in recs]
    T  = [float((r.get("scores", {}) or {}).get("trustworthiness_score", 0.0)) for r in recs]

    def _newfig(title: str):
        plt.figure(figsize=(10, 4))
        plt.title(title, fontsize=11)
        plt.xlabel("instance", fontsize=9)
        plt.ylabel("score", fontsize=9)
        plt.ylim(-0.05, 1.05)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)

    def _legend_outside():
        plt.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
            fontsize=8,
            frameon=False
        )
        plt.tight_layout(rect=[0, 0, 0.82, 1])

    paths: Dict[str, str] = {}

    _newfig("Scores: Q1, Q2, Complexity (SPaRTUN)")
    plt.plot(x, Q1, label="Q1", linewidth=1.5)
    plt.plot(x, Q2, label="Q2", linewidth=1.5)
    plt.plot(x, Q3, label="Q3", linewidth=1.5)
    _legend_outside()
    p1 = os.path.join(out_dir, "plot_scores_q1_q2_q3.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()
    paths["q1_q2_q3"] = p1

    _newfig("Trust vs Complexity (SPaRTUN)")
    plt.plot(x, T,  label="T", linewidth=1.5)
    plt.plot(x, Q3, label="Q3", linewidth=1.5)
    _legend_outside()
    p2 = os.path.join(out_dir, "plot_trust_vs_complexity.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()
    paths["trust_vs_complexity"] = p2

    _newfig("Accuracy vs Trust (SPaRTUN)")
    plt.plot(x, B,  label="Text-only (B)", linewidth=1.5)
    plt.plot(x, R,  label="Relations (R)", linewidth=1.5)
    plt.plot(x, T,  label="Trust (T)", linewidth=1.5)
    _legend_outside()
    p3 = os.path.join(out_dir, "plot_accuracy_vs_trust.png")
    plt.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close()
    paths["accuracy_vs_trust"] = p3

    return paths


# ============================================================
# 15) MAIN
# ============================================================

if __name__ == "__main__":
    input_path = "./cor2ef_llm_only_test_with_grids_line_relationsqwen314b (1) (1).json"
    out_json = "./trustnew_eval_spartun_qwen314b.json"
    out_plots_dir = "./plotsnew_spartun_qwen314b"

    TRUST_NO_SWITCH = 0.8
    COMPLEXITY_NO_SWITCH = 0.45

    records = run_spartun_dataset_one_pretty_json(
    input_path=input_path,
    output_json=out_json,
    dataset_name="spartun",
    llama_model="qwen3:14b",
    temp=0.7,
    limit_contexts=3,     # process this many contexts
    skip_contexts=0,       # skip this many contexts first, then process limit_contexts
    limit=None,            # optional
    start_after_qid=-1,    # run all questions (0..)
    trust_noswitch_threshold=TRUST_NO_SWITCH,
    complexity_noswitch_threshold=COMPLEXITY_NO_SWITCH,
    gpt_model="gpt-5-mini")


    print_summary_stats(records)

    plots = save_correlation_plots(records, out_dir=out_plots_dir)
    print("Saved plots:", plots)
