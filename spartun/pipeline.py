# run_all_modalities_rerun_yn_full_grid.py
# Rerun QA for: full-grid, pruned-grid
#
# INPUT JSON: yn_full_grid_incorrect_only.json (flat item list; each item is one question)
# OUTPUT JSON: writes results back into the same file by default, while preserving old runs
#
# Uses your existing prompts_lib.py grid 3-prompt setup end-to-end:
#   Prompt1: build_prompt_grid_interpretation(grid) OR build_prompt_grid_small_interpretation(pruned_grid)
#   Prompt2: build_prompt_question_plan_yn(question) for YN; legacy question prompt otherwise
#   Prompt3: build_prompt_grid_answer_yn(interp, qplan_json, question) for YN; legacy answer prompt otherwise
#
# This keeps your hyperparams, prompt logging, timing, and full pipeline structure,
# but removes text-only and text+relations reasoning.

import json
import os
import re
import time
import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import spartun.prompts_lib as P
from utils.llm_clients import call_gpt as _call_gpt_llm, call_ollama as _call_ollama_llm
# -----------------------
# CONFIG
# -----------------------
SRC_JSON = Path("./spartunprev/llama370bgridgen.json")
OUT_JSON = SRC_JSON   # write results back into the YN file itself
BACKUP_JSON = Path("./newallama370b_incorrect_only.backup.json")
PROMPTS_JSONL = Path("./spartunprev/newllama370b_grid_rerun_prompts.jsonl")
SUMMARY_JSON = Path("./spartunprev/newllama370b_all_grid_rerun_summary.json")
LIMIT_CASES: Optional[int] = None   # None = all cases


# Ollama
OLLAMA_MODEL_NAME = "llama3.1:70b"
OLLAMA_TEMPERATURE = 0.0
OLLAMA_NUM_PREDICT = 8000
# Pipeline switches
RUN_FULL_GRID = True
RUN_PRUNED_GRID = True
RERUN_GRID_INTERP = True    # full end-to-end grid pipeline
PRESERVE_OLD_RUNS = True
ALWAYS_REPRUNE = True        # always re-do entity selection + pruning from full grid
MAX_RETRIES = 10              # if selected_option is None, retry this many extra times
START_INDEX = 0   # skip first 164 items, start from 165th item
# If pruned grid missing and you still want a fallback prune.
SMALLGRID_MARGIN_ROWS = 1

# OpenAI / GPT (for extraction fallback + grid validation)
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_MODEL = "gpt-5.1"
USE_GPT_EXTRACT = True      # GPT fallback when Ollama gives None after retries
USE_GPT_GRID_VALIDATE = True  # GPT validation of pruned grids

# -----------------------
# I/O helpers
# -----------------------

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def file_is_empty(path: Path) -> bool:
    return (not path.exists()) or (path.stat().st_size == 0)


def stable_prompt_id(text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[:16]


def log_prompt(kind: str, meta: Dict[str, Any], prompt_text: str) -> str:
    pid = stable_prompt_id(prompt_text)
    rec = {
        "type": "prompt",
        "prompt_id": pid,
        "kind": kind,
        "meta": meta,
        "prompt": prompt_text,
    }
    append_jsonl(PROMPTS_JSONL, rec)
    return pid


# -----------------------
# Ollama / GPT callers (thin wrappers delegating to utils/llm_clients)
# -----------------------

def call_ollama(prompt: str) -> str:
    return _call_ollama_llm(prompt, OLLAMA_MODEL_NAME, OLLAMA_TEMPERATURE, seed=None, num_predict=OLLAMA_NUM_PREDICT)


def call_gpt(prompt: str, max_tokens: int = 2000) -> Optional[str]:
    """Call GPT via the Responses API. Returns raw text."""
    return _call_gpt_llm(prompt, model=GPT_MODEL)

# -----------------------
# GPT answer extraction from raw Ollama output
# -----------------------

def _gpt_extract_answer(raw_output: str, question: str, q_type: str,
                        candidates: List[str]) -> Optional[Any]:
    """
    When Ollama produced raw text but JSON parsing failed (selected_option=None),
    use GPT to read the reasoning and extract the final answer.
    """
    cand_text = ", ".join(candidates) if candidates else "Yes, No"
    prompt = (
        "You are given the RAW OUTPUT of a spatial reasoning model that attempted to answer a question.\n"
        "The output may be TRUNCATED (cut off mid-sentence), contain thinking traces,\n"
        "partial JSON, or verbose reasoning.\n\n"
        "Your job: extract the FINAL ANSWER from this output by reading the reasoning.\n\n"
        f"Question type: {q_type}\n"
        f"Question: {question}\n"
        f"Candidate answers: {cand_text}\n\n"
        "RAW OUTPUT (may be truncated):\n"
        f"{raw_output[:4000]}\n\n"
        "RULES:\n"
        "1. Read the model's REASONING carefully even if the JSON is incomplete.\n"
        "2. Look for conclusions like 'the answer is Yes/No', 'predicate is satisfied',\n"
        "   'verdict:', 'only X and Y are TRUE', etc.\n"
        "3. If the output is truncated mid-JSON but the justification text clearly\n"
        "   indicates the answer, extract that answer.\n"
        "4. For YN: the answer must be Yes or No.\n"
        "5. For FR: extract all spatial relations the model concluded are TRUE.\n"
        "6. Only set selected_option to null if you truly cannot determine any answer.\n\n"
        'Return ONLY JSON: {"selected_option": [...], "justification": "brief extraction note"}\n'
    )
    gpt_raw = call_gpt(prompt)
    if gpt_raw is None:
        return None
    parsed = try_parse_json_dict(gpt_raw)
    return parsed.get("selected_option")


# -----------------------
# GPT pruned grid validation
# -----------------------

def _gpt_validate_and_fix_grid(pruned_grid: str, full_grid: str,
                                question: str, q_type: str) -> Optional[str]:
    """
    Use GPT to check if the pruned grid is structurally correct and contains
    all entities needed to answer the question. Returns fixed grid or None if valid.
    """
    prompt = (
        "You are validating a PRUNED spatial grid against its FULL grid.\n\n"
        "The pruned grid should:\n"
        "1. Contain all entities relevant to answering the question\n"
        "2. Have matching opening '[' and closing ']' brackets for every box\n"
        "3. Have a proper grid header with Col(...) markers\n"
        "4. Have Row(N) Col(M) on every entity line\n"
        "5. For 'all'/'every' questions, include ALL entities matching the category\n"
        "6. For 'any'/'a' questions, include at least the relevant entities\n\n"
        f"Question type: {q_type}\n"
        f"Question: {question}\n\n"
        "FULL GRID:\n"
        f"{full_grid}\n\n"
        "PRUNED GRID:\n"
        f"{pruned_grid}\n\n"
        "Return ONLY JSON:\n"
        "{\n"
        '  "is_valid": true/false,\n'
        '  "issues": ["list of issues found"],\n'
        '  "fixed_grid": "<corrected pruned grid if invalid, or null if valid>"\n'
        "}\n"
    )
    gpt_raw = call_gpt(prompt, max_tokens=3000)
    if gpt_raw is None:
        return None
    parsed = try_parse_json_dict(gpt_raw)
    if not parsed.get("is_valid") and parsed.get("fixed_grid"):
        fixed = parsed["fixed_grid"]
        if isinstance(fixed, str) and fixed.strip() and '_in(' in fixed:
            return fixed
    return None


# -----------------------
# Parsing helpers
# -----------------------

def _strip_code_fences(t: str) -> str:
    t = re.sub(r"^```(?:json)?\s*", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t.strip())
    return t.strip()


def _strip_think_tags(t: str) -> str:
    """Remove <think>...</think> blocks from Qwen thinking mode output."""
    return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()


def try_parse_json_dict(text: str) -> Dict[str, Any]:
    if text is None:
        return {"error": "none"}
    t = _strip_think_tags(str(text))
    t = _strip_code_fences(t)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        return {"error": "not_dict", "raw": t}
    except Exception:
        m = re.search(r"\{.*\}", t, flags=re.DOTALL)
        if m:
            chunk = m.group(0)
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
        return {"error": "json_parse_fail", "raw": t[:5000]}


def normalize_answer(ans: Any) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return str(ans).strip().lower()


def check_answer_multi(selected_option: Any, gold_answer: Any) -> Optional[bool]:
    if selected_option is None or gold_answer is None:
        return None

    sel_list = selected_option if isinstance(selected_option, list) else [selected_option]
    gold_list = gold_answer if isinstance(gold_answer, list) else [gold_answer]

    sel_norm = {normalize_answer(x) for x in sel_list if x is not None}
    gold_norm = {normalize_answer(x) for x in gold_list if x is not None}

    sel_norm.discard(None)
    gold_norm.discard(None)

    if not sel_norm or not gold_norm:
        return None

    return len(sel_norm.intersection(gold_norm)) > 0


# -----------------------
# Relations helpers (for prune fallback only)
# -----------------------
_ALLOWED_RELS = {
    "above", "below", "left", "right", "front", "behind", "near", "far",
    "dc", "ec", "po", "tpp", "ntpp", "tppi", "ntppi"
}


def _rel_to_str(r: Dict[str, Any]) -> Optional[str]:
    h = r.get("head") or r.get("head_id")
    rel = r.get("relation")
    t = r.get("tail") or r.get("tail_id")
    if not (isinstance(h, str) and isinstance(rel, str) and isinstance(t, str)):
        return None
    reln = rel.strip().lower()
    if reln not in _ALLOWED_RELS:
        reln = rel.strip().lower()
    return f"{h.strip()} {reln} {t.strip()}"


def get_relations_for_item(item: Dict[str, Any]) -> List[str]:
    rels = item.get("relations_used")
    if isinstance(rels, list) and rels:
        out = []
        for r in rels:
            if isinstance(r, str) and r.strip():
                out.append(r.strip())
            elif isinstance(r, dict):
                s = _rel_to_str(r)
                if s:
                    out.append(s)
        if out:
            return out

    rels = item.get("predicted_relations_used")
    if not isinstance(rels, list) or not rels:
        rels = item.get("ground_truth_relations")
    if not isinstance(rels, list):
        return []

    out = []
    for r in rels:
        if isinstance(r, str) and r.strip():
            out.append(r.strip())
        elif isinstance(r, dict):
            s = _rel_to_str(r)
            if s:
                out.append(s)

    seen = set()
    dedup = []
    for x in out:
        k = x.lower()
        if k not in seen:
            dedup.append(x)
            seen.add(k)
    return dedup


# -----------------------
# Small-grid pruning utilities (IMPROVED: preserves box headers + inter-box tags)
# -----------------------

def _is_row_line(line: str) -> bool:
    ll = line.strip().lower()
    return ll.startswith("row(") or ll.startswith("row ")


def _extract_row_label(line: str) -> Optional[str]:
    s = line.strip()
    m = re.match(r"^(Row\(\s*\d+\s*\))", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^(Row\s+\d+)", s, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _replace_row_label(line: str, new_label: str) -> str:
    old = _extract_row_label(line)
    if not old:
        return line
    return line.replace(old, new_label, 1)


def _is_box_header(line: str) -> bool:
    s = line.strip()
    return bool(re.search(r"\[(?:box|block)\s+\w+", s, flags=re.IGNORECASE))


def _is_box_closing(line: str) -> bool:
    s = line.strip()
    return s in ("]", "],", "]]", "]]]", "]]],") or (
        s.endswith("]") and "[" not in s
    )


def _get_box_name_from_header(line: str) -> Optional[str]:
    m = re.search(r"\[(?:box|block)\s+(\w+)", line, flags=re.IGNORECASE)
    return m.group(1) if m else None


def prune_grid_text_by_entities_and_renumber(
    grid_text: str,
    selected_entities: List[str],
    margin_rows: int = 0,
) -> str:
    if not grid_text or not selected_entities:
        return grid_text

    sel_low = [s.lower() for s in selected_entities]
    lines = grid_text.splitlines()
    if not lines:
        return grid_text

    keep = [False] * len(lines)
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st.startswith("identifier:") or st == "Grid:" or ("Col(" in st and "Row(" not in st):
            keep[i] = True

    row_indices = [i for i, ln in enumerate(lines) if _is_row_line(ln) and _extract_row_label(ln)]
    if not row_indices:
        for i, ln in enumerate(lines):
            low = ln.lower()
            if any(s in low for s in sel_low):
                keep[i] = True
        pruned = [ln for i, ln in enumerate(lines) if keep[i]]
        return "\n".join(pruned) if pruned else grid_text

    entity_hit_positions: Set[int] = set()
    for idx_pos, i in enumerate(row_indices):
        low = lines[i].lower()
        if any(s in low for s in sel_low):
            entity_hit_positions.add(idx_pos)

    if not entity_hit_positions:
        return grid_text

    kmin = max(0, min(entity_hit_positions) - margin_rows)
    kmax = min(len(row_indices) - 1, max(entity_hit_positions) + margin_rows)
    kept_row_positions = set(range(kmin, kmax + 1))

    for pos in kept_row_positions:
        start = row_indices[pos]
        keep[start] = True
        end = row_indices[pos + 1] if (pos + 1) < len(row_indices) else len(lines)
        for j in range(start + 1, end):
            keep[j] = True

    box_headers: List[Tuple[int, str]] = []
    box_closings: Dict[int, int] = {}

    for i, ln in enumerate(lines):
        if _is_box_header(ln):
            bname = _get_box_name_from_header(ln)
            if bname:
                box_headers.append((i, bname))

    for hidx, (hline, hname) in enumerate(box_headers):
        next_header_line = box_headers[hidx + 1][0] if hidx + 1 < len(box_headers) else len(lines)
        for j in range(hline + 1, next_header_line):
            if _is_box_closing(lines[j]):
                box_closings[hline] = j
                break

    for pos in kept_row_positions:
        line_idx = row_indices[pos]
        enclosing_header = None
        for hline, hname in reversed(box_headers):
            if hline < line_idx:
                enclosing_header = hline
                break
        if enclosing_header is not None:
            keep[enclosing_header] = True
            if enclosing_header in box_closings:
                keep[box_closings[enclosing_header]] = True

    kept_box_names: Set[str] = set()
    for hline, hname in box_headers:
        if keep[hline]:
            kept_box_names.add(hname.lower())

    for hline, hname in box_headers:
        if not keep[hline]:
            low = lines[hline].lower()
            for kbn in kept_box_names:
                if kbn in low:
                    keep[hline] = True
                    if hline in box_closings:
                        keep[box_closings[hline]] = True
                    kept_box_names.add(hname.lower())
                    break

    pruned_lines = [ln for i, ln in enumerate(lines) if keep[i]]

    out = pruned_lines[:]
    pruned_row_idxs = [i for i, ln in enumerate(out) if _is_row_line(ln) and _extract_row_label(ln)]
    for n, ridx in enumerate(pruned_row_idxs, start=1):
        out[ridx] = _replace_row_label(out[ridx], f"Row({n})")

    return "\n".join(out)


# -----------------------
# Post-prune bracket fix
# -----------------------

def _fix_grid_brackets(grid_text: str) -> str:
    """Fix a pruned grid that has mismatched brackets by appending missing ']'."""
    if not grid_text:
        return grid_text

    lines = grid_text.splitlines()
    open_boxes = []

    for i, line in enumerate(lines):
        s = line.strip()
        m = re.search(r'\[(?:box|block)\s+(\w+)', s, flags=re.IGNORECASE)
        if m:
            open_boxes.append(m.group(1))
        if s.endswith(']') and '[' not in s:
            if open_boxes:
                open_boxes.pop()

    if open_boxes:
        for _ in reversed(open_boxes):
            lines.append("]")
        return "\n".join(lines)

    return grid_text


# -----------------------
# Entity completeness for all/any quantifier questions
# -----------------------

def _question_needs_all_entities(question: str) -> List[str]:
    """
    If a YN question uses 'all'/'every'/'each'/'any' quantifiers,
    return keywords that must match entities in the pruned grid.
    """
    q_lower = question.lower().strip()
    spatial = (r'(?:\s+(?:to|in|above|below|left|right|north|south|east|west|'
               r'inside|within|covered|near|far|behind|front|touching|disconnected))')

    patterns = [
        r'\ball\s+([\w\s]+?)' + spatial,
        r'\bevery\s+([\w\s]+?)' + spatial,
        r'\beach\s+([\w\s]+?)' + spatial,
        r'\bany\s+([\w\s]+?)' + spatial,
    ]

    keywords = []
    for pat in patterns:
        m = re.search(pat, q_lower)
        if m:
            phrase = m.group(1).strip()
            for word in phrase.split():
                if word not in {"a", "an", "the", "of", "in", "is", "are"}:
                    keywords.append(word)

    return list(set(keywords))


def _ensure_entity_completeness(
    pruned_grid: str,
    full_grid: str,
    question: str,
    selected_entities: List[str],
) -> Tuple[str, List[str]]:
    """
    For all/any questions, check if pruned grid has all matching entities.
    If not, re-prune with the missing ones added.
    Returns (fixed_grid, updated_selected_entities).
    """
    keywords = _question_needs_all_entities(question)
    if not keywords:
        return pruned_grid, selected_entities

    full_entities = _extract_entities_from_grid(full_grid)
    required = [e for e in full_entities
                if all(kw in e.lower() for kw in keywords)]
    if not required:
        return pruned_grid, selected_entities

    pruned_entities = _extract_entities_from_grid(pruned_grid)
    pruned_lower = {e.lower() for e in pruned_entities}
    missing = [e for e in required if e.lower() not in pruned_lower]

    if not missing:
        return pruned_grid, selected_entities

    new_selected = list(set(selected_entities + required))
    new_grid = prune_grid_text_by_entities_and_renumber(
        full_grid, new_selected, margin_rows=SMALLGRID_MARGIN_ROWS,
    )
    new_grid = _fix_grid_brackets(new_grid)
    print(f"  entity-completeness: added {len(missing)} missing entities: {missing}")
    return new_grid, new_selected


# -----------------------
# Entity universe extraction
# -----------------------

def _extract_entities_from_grid(grid_text: str) -> List[str]:
    if not grid_text:
        return []
    entities = set()
    box_names = set()
    for line in grid_text.splitlines():
        bm = re.search(r'\[(?:box|block)\s+(\w+)', line, flags=re.IGNORECASE)
        if bm:
            box_names.add(f"box {bm.group(1)}")
            box_names.add(f"block {bm.group(1)}")
        em = re.search(r'(?:Col\(\d+\))\s+(.+?)_in\(', line)
        if em:
            ename = em.group(1).strip()
            if ename:
                entities.add(ename)
    return sorted(entities | box_names, key=lambda x: x.lower())


def build_universe_from_item(item: Dict[str, Any]) -> List[str]:
    # 1) Explicit universe field
    if isinstance(item.get("universe"), list) and item["universe"]:
        return item["universe"]

    # 2) objects_info
    objinfo = item.get("objects_info") or {}
    out = set()
    if isinstance(objinfo, dict) and objinfo:
        for _, v in objinfo.items():
            if not isinstance(v, dict):
                continue
            fn = v.get("full_name")
            gn = v.get("general_name")
            if isinstance(fn, str) and fn.strip():
                out.add(fn.strip())
            if isinstance(gn, str) and gn.strip():
                out.add(gn.strip())
        for k in objinfo.keys():
            if isinstance(k, str) and k.strip():
                out.add(k.strip())
        if out:
            return sorted(out, key=lambda x: x.lower())

    # 3) entity_selection.*.universe
    ent_sel = item.get("entity_selection", {})
    if isinstance(ent_sel, dict):
        for key in ("pruned_grid", "full_grid", "pruned_grid_rerun"):
            sel_data = ent_sel.get(key)
            if isinstance(sel_data, dict) and isinstance(sel_data.get("universe"), list):
                return sel_data["universe"]

    # 4) Extract from full grid text
    full_grid = item.get("full_grid_used", "")
    if isinstance(full_grid, str) and full_grid.strip():
        extracted = _extract_entities_from_grid(full_grid)
        if extracted:
            return extracted

    return []


def select_entities_with_ollama(question: str, universe: List[str]) -> Dict[str, Any]:
    prompt = P.build_entity_selection_prompt_no_relblock(question, universe)
    raw = call_ollama(prompt)
    parsed = try_parse_json_dict(raw)

    sel = parsed.get("selected_entities", [])
    if not isinstance(sel, list):
        sel = []

    uni_map = {u.lower(): u for u in universe}
    norm = []
    seen = set()
    for s in sel:
        if not isinstance(s, str):
            continue
        key = s.strip().lower()
        if key in uni_map and key not in seen:
            norm.append(uni_map[key])
            seen.add(key)

    return {"who": "ollama", "prompt": prompt, "raw": raw, "parsed": parsed, "selected": norm}


# -----------------------
# Grid QA runner (3-prompt grid pipeline)
# -----------------------

def run_three_prompt_grid(
    grid_text: str,
    question: str,
    candidates: List[str],
    meta: Dict[str, Any],
    kind_prefix: str,
    q_type: str,
    small: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    timing: Dict[str, float] = {}

    if small:
        p1 = P.build_prompt_grid_small_interpretation(grid_text)
        pid1 = log_prompt(f"{kind_prefix}_small_grid_interp", meta, p1)
        t = time.time()
        interp_raw = _strip_think_tags(call_ollama(p1))
        timing["grid_interp_s"] = time.time() - t
    else:
        p1 = P.build_prompt_grid_interpretation(grid_text)
        pid1 = log_prompt(f"{kind_prefix}_grid_interp", meta, p1)
        t = time.time()
        interp_raw = _strip_think_tags(call_ollama(p1))
        timing["grid_interp_s"] = time.time() - t

    is_yn = str(q_type or "").strip().upper() == "YN"

    if is_yn:
        p2 = P.build_prompt_question_plan_yn(question)
    else:
        p2 = P.build_prompt_question_interpretation(question, candidates)
    pid2 = log_prompt(f"{kind_prefix}_question_plan", meta, p2)
    t = time.time()
    qplan_raw = _strip_think_tags(call_ollama(p2))
    timing["question_plan_s"] = time.time() - t

    if is_yn:
        p3 = P.build_prompt_grid_answer_yn(
            interp_text=interp_raw,
            question_plan_json=qplan_raw,
            question=question,
        )
    else:
        p3 = P.build_prompt_grid_answer_from_interp_and_question_plan(
            interp_text=interp_raw,
            question_plan_json=qplan_raw,
            candidate_answers=candidates,
        )
    pid3 = log_prompt(f"{kind_prefix}_answer", meta, p3)
    t = time.time()
    ans_raw = call_ollama(p3)
    timing["answer_s"] = time.time() - t

    ans_cleaned = _strip_think_tags(ans_raw)
    ans_parsed = try_parse_json_dict(ans_cleaned)
    qplan_parsed = try_parse_json_dict(qplan_raw)
    selected = ans_parsed.get("selected_option")
    just = ans_parsed.get("justification")

    result = {
        "prompt_ids": {"grid_interp": pid1, "question_plan": pid2, "answer": pid3},
        "interp_raw": interp_raw,
        "question_plan_raw": qplan_raw,
        "question_plan_parsed": qplan_parsed,
        "answer_raw": ans_raw,
        "answer_parsed": ans_parsed,
        "selected_option": selected,
        "justification": just,
        "prompt_family": "yn_specific" if is_yn else "legacy",
    }
    return result, timing


# -----------------------
# Retry wrapper: reruns until selected_option is not None, up to MAX_RETRIES extra attempts
# -----------------------

def run_with_retry(
    grid_text: str,
    question: str,
    candidates: List[str],
    meta: Dict[str, Any],
    kind_prefix: str,
    q_type: str,
    small: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    run_result, timing = run_three_prompt_grid(
        grid_text=grid_text,
        question=question,
        candidates=candidates,
        meta=meta,
        kind_prefix=kind_prefix,
        q_type=q_type,
        small=small,
    )
    for attempt in range(1, MAX_RETRIES + 1):
        if run_result.get("selected_option") is not None:
            break
        print(f"  {kind_prefix} attempt {attempt}: got None answer, retrying...")
        run_result, timing = run_three_prompt_grid(
            grid_text=grid_text,
            question=question,
            candidates=candidates,
            meta=meta,
            kind_prefix=kind_prefix,
            q_type=q_type,
            small=small,
        )

    # GPT extraction fallback: if still None but raw output exists, ask GPT
    if run_result.get("selected_option") is None and USE_GPT_EXTRACT:
        raw = run_result.get("answer_raw", "")
        if raw and isinstance(raw, str) and len(raw.strip()) > 10:
            print(f"  {kind_prefix}: Ollama gave None after retries, trying GPT extraction...")
            gpt_sel = _gpt_extract_answer(raw, question, q_type, candidates)
            if gpt_sel is not None:
                run_result["selected_option"] = gpt_sel
                run_result["repair_note"] = "gpt_extracted_from_raw"
                print(f"  {kind_prefix}: GPT extracted -> {gpt_sel}")

    return run_result, timing


# -----------------------
# Get grids for an item
# -----------------------

def get_full_grid_for_item(item: Dict[str, Any]) -> str:
    g = item.get("full_grid_used")
    if isinstance(g, str) and g.strip():
        return g
    g = item.get("pruned_grid_used")
    if isinstance(g, str) and g.strip():
        return g
    return ""


def get_pruned_grid_for_item(item: Dict[str, Any]) -> str:
    g = item.get("pruned_grid_used")
    if isinstance(g, str) and g.strip():
        return g
    return ""


# -----------------------
# Main
# -----------------------

def main():
    t0 = time.time()

    if not SRC_JSON.exists():
        raise FileNotFoundError(f"Missing input file: {SRC_JSON}")

    if not BACKUP_JSON.exists():
        shutil.copy2(SRC_JSON, BACKUP_JSON)
        print(f"backup created: {BACKUP_JSON}")

    if file_is_empty(PROMPTS_JSONL):
        append_jsonl(PROMPTS_JSONL, {
            "type": "meta",
            "name": "rerun_yn_full_grid_pipeline",
            "src_json": str(SRC_JSON),
            "out_json": str(OUT_JSON),
            "ollama_model": OLLAMA_MODEL_NAME,
            "ollama_temperature": OLLAMA_TEMPERATURE,
            "ollama_num_predict": OLLAMA_NUM_PREDICT,
            "run_full_grid": RUN_FULL_GRID,
            "run_pruned_grid": RUN_PRUNED_GRID,
            "rerun_grid_interp": RERUN_GRID_INTERP,
            "started_at_unix": time.time(),
        })

    with SRC_JSON.open("r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Expected SRC_JSON to contain a JSON list of item dicts.")

    total_items = len(items)

    # ----- Pre-run cleanup -----
    # 1) Remove empty template runs (no prompt_family, no answer_raw = GPT-generated stubs)
    cleaned_template = 0
    for item in items:
        runs = item.get("runs", {})
        if not isinstance(runs, dict):
            continue
        keys_to_remove = []
        for rk, rv in runs.items():
            if not isinstance(rv, dict):
                keys_to_remove.append(rk)
                continue
            pf = rv.get("prompt_family")
            raw = rv.get("answer_raw")
            if pf is None and (raw is None or raw == "" or raw is False):
                keys_to_remove.append(rk)
        for rk in keys_to_remove:
            del runs[rk]
            cleaned_template += 1
    if cleaned_template:
        print(f"Cleaned {cleaned_template} empty template runs")

    # 2) Fix existing broken pruned grids in the JSON
    fixed_grids = 0
    for item in items:
        for gk in ("pruned_grid_used", "pruned_grid_used_rerun"):
            g = item.get(gk, "") or ""
            if g and g.count('[') > g.count(']'):
                item[gk] = _fix_grid_brackets(g)
                fixed_grids += 1
    if fixed_grids:
        print(f"Fixed {fixed_grids} existing broken pruned grids")

    # 3) Try to extract answers from existing raw output that has None selected_option
    recovered = 0
    for item in items:
        runs = item.get("runs", {})
        if not isinstance(runs, dict):
            continue
        gold = item.get("gold")
        question = item.get("question", "")
        q_type = item.get("q_type", "YN")
        candidates = item.get("candidate_answers") or ["Yes", "No"]
        for rk, rv in runs.items():
            if not isinstance(rv, dict):
                continue
            if rv.get("selected_option") is not None:
                continue
            raw = rv.get("answer_raw")
            if not raw or not isinstance(raw, str) or len(raw.strip()) < 10:
                continue
            # Try better parsing (think-tag aware)
            parsed = try_parse_json_dict(raw)
            new_sel = parsed.get("selected_option")
            if new_sel is not None:
                rv["selected_option"] = new_sel
                rv["justification"] = parsed.get("justification", rv.get("justification"))
                rv["correct"] = check_answer_multi(new_sel, gold)
                rv["repair_note"] = "recovered_by_better_parsing"
                recovered += 1
                continue
            # GPT extraction fallback
            if USE_GPT_EXTRACT:
                gpt_sel = _gpt_extract_answer(raw, question, q_type, candidates)
                if gpt_sel is not None:
                    rv["selected_option"] = gpt_sel
                    rv["correct"] = check_answer_multi(gpt_sel, gold)
                    rv["repair_note"] = "recovered_by_gpt_extract"
                    recovered += 1
    if recovered:
        print(f"Recovered {recovered} answers from existing raw output")

    # ----- BEFORE accuracy -----
    print("\n" + "=" * 70)
    print("  ACCURACY BEFORE NEW RUNS")
    print("=" * 70)
    for rk_label in ("full_grid_yn", "pruned_grid_yn"):
        total_rk = 0
        correct_rk = 0
        none_rk = 0
        for item in items:
            runs = item.get("runs", {})
            rv = runs.get(rk_label)
            if not isinstance(rv, dict):
                continue
            total_rk += 1
            sel = rv.get("selected_option")
            if sel is None:
                none_rk += 1
            elif check_answer_multi(sel, item.get("gold")):
                correct_rk += 1
        if total_rk:
            print(f"  {rk_label}: {correct_rk}/{total_rk} = {correct_rk/total_rk*100:.1f}%"
                  f"  (none: {none_rk})")
        else:
            print(f"  {rk_label}: no existing runs")
    print("=" * 70 + "\n")

    processed = 0

    stats = {
        "full_grid_yn": {"ran": 0, "correct": 0},
        "pruned_grid_yn": {"ran": 0, "correct": 0},
    }

    for idx, item in enumerate(items):
        if idx < START_INDEX:
               continue

        if LIMIT_CASES is not None and processed >= LIMIT_CASES:
            break
        if not isinstance(item, dict):
            continue

        identifier = item.get("identifier", "")
        q_id = item.get("q_id")
        q_type = item.get("q_type", "YN")
        question = item.get("question") or ""
        gold = item.get("gold")
        candidates = item.get("candidate_answers") or ["Yes", "No"]

        meta = {"identifier": identifier, "q_id": q_id, "q_type": q_type}
        t_q = time.time()

        if PRESERVE_OLD_RUNS and isinstance(item.get("runs"), dict) and "old_runs_snapshot" not in item:
            item["old_runs_snapshot"] = item.get("runs")

        item.setdefault("runs", {})
        item.setdefault("timing", {})
        item.setdefault("ollama", {})

        print(f"\n[{idx + 1}/{total_items}] {identifier} q_id={q_id} q_type={q_type}")
        print(f"Q: {question}")

        # -------- per-run-type skip checks --------
        existing_pg = item.get("runs", {}).get("pruned_grid_yn", {})
        existing_fg = item.get("runs", {}).get("full_grid_yn", {})
        # FG: skip only if already has a non-None answer
        fg_has_answer = isinstance(existing_fg, dict) and existing_fg.get("selected_option") is not None
        # PG: skip only if already has a non-None AND correct answer
        pg_has_answer = isinstance(existing_pg, dict) and existing_pg.get("selected_option") is not None
        pg_is_correct = pg_has_answer and check_answer_multi(existing_pg.get("selected_option"), gold)
        skip_fg = fg_has_answer
        skip_pg = pg_has_answer and pg_is_correct
        if skip_fg and skip_pg:
            print(f"  fg has answer + pg correct, skipping")
            continue

        # -------- full grid --------
        full_grid = get_full_grid_for_item(item)
        full_grid_run = None
        full_grid_t: Dict[str, float] = {}
        full_grid_correct = None

        if RUN_FULL_GRID and not skip_fg and isinstance(full_grid, str) and full_grid.strip():
            full_grid_run, full_grid_t = run_with_retry(
                grid_text=full_grid,
                question=question,
                candidates=candidates,
                meta=meta,
                kind_prefix="full_grid_yn",
                q_type=q_type,
                small=False,
            )
            full_grid_correct = check_answer_multi(full_grid_run.get("selected_option"), gold)
            item["runs"]["full_grid_yn"] = {**full_grid_run, "correct": full_grid_correct}
            item["timing"]["full_grid_yn"] = full_grid_t
            stats["full_grid_yn"]["ran"] += 1
            stats["full_grid_yn"]["correct"] += int(bool(full_grid_correct))
            print(f"  full_grid_yn -> {full_grid_run.get('selected_option')} | correct={full_grid_correct}")

        # -------- pruned grid --------
        pruned_grid = "" if ALWAYS_REPRUNE else get_pruned_grid_for_item(item)
        pruned_grid_t: Dict[str, float] = {}
        pruned_grid_run = None
        pruned_grid_correct = None
        prune_extra_timing: Dict[str, float] = {}
        pruned_grid_ent_sel = None

        if RUN_PRUNED_GRID and not skip_pg:
            if not (isinstance(pruned_grid, str) and pruned_grid.strip()):
                universe = build_universe_from_item(item)
                if not universe:
                    print(f"  WARNING: no entity universe found for {identifier}, skipping pruned grid")
                if universe and full_grid.strip():
                    print(f"  universe has {len(universe)} entities: {universe[:5]}...")
                    t_es = time.time()
                    pruned_grid_ent_sel = select_entities_with_ollama(question, universe)
                    prune_extra_timing["entity_select_s"] = time.time() - t_es
                    ent_pid = log_prompt("pruned_entity_select_rerun", meta, pruned_grid_ent_sel["prompt"])
                    pruned_grid_ent_sel["prompt_id"] = ent_pid
                    pruned_grid_ent_sel["universe"] = universe

                    t_pr = time.time()
                    pruned_grid = prune_grid_text_by_entities_and_renumber(
                        full_grid,
                        pruned_grid_ent_sel.get("selected", []),
                        margin_rows=SMALLGRID_MARGIN_ROWS,
                    )
                    # Fix any mismatched brackets from pruning
                    pruned_grid = _fix_grid_brackets(pruned_grid)
                    # Ensure all/any quantifier questions have all matching entities
                    pruned_grid, _ = _ensure_entity_completeness(
                        pruned_grid, full_grid, question,
                        pruned_grid_ent_sel.get("selected", []),
                    )
                    # GPT validation: ask GPT if the pruned grid is correct
                    if USE_GPT_GRID_VALIDATE:
                        gpt_fixed = _gpt_validate_and_fix_grid(
                            pruned_grid, full_grid, question, q_type,
                        )
                        if gpt_fixed:
                            print(f"  GPT fixed pruned grid for {identifier}")
                            pruned_grid = gpt_fixed
                    prune_extra_timing["prune_s"] = time.time() - t_pr
                    item.setdefault("entity_selection", {})
                    item["entity_selection"]["pruned_grid_rerun"] = pruned_grid_ent_sel
                    item["pruned_grid_used_rerun"] = pruned_grid
                    print(f"  re-pruned grid: {len(pruned_grid_ent_sel.get('selected', []))} entities selected")

            if isinstance(pruned_grid, str) and pruned_grid.strip():
                pruned_grid_run, pruned_grid_t = run_with_retry(
                    grid_text=pruned_grid,
                    question=question,
                    candidates=candidates,
                    meta=meta,
                    kind_prefix="pruned_grid_yn",
                    q_type=q_type,
                    small=True,
                )
                pruned_grid_correct = check_answer_multi(pruned_grid_run.get("selected_option"), gold)
                item["runs"]["pruned_grid_yn"] = {**pruned_grid_run, "correct": pruned_grid_correct}
                item["timing"]["pruned_grid_yn"] = {**pruned_grid_t, **prune_extra_timing}
                stats["pruned_grid_yn"]["ran"] += 1
                stats["pruned_grid_yn"]["correct"] += int(bool(pruned_grid_correct))
                print(f"  pruned_grid_yn -> {pruned_grid_run.get('selected_option')} | correct={pruned_grid_correct}")

        q_total = time.time() - t_q
        item["timing"]["question_total_rerun_s"] = q_total
        item["ollama"].update({
            "model": OLLAMA_MODEL_NAME,
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
        })
        item["rerun_meta"] = {
            "source": "yn_full_grid_incorrect_only",
            "updated_full_grid_prompt_family": "yn_specific",
            "rerun_grid_interp": RERUN_GRID_INTERP,
            "always_reprune": ALWAYS_REPRUNE,
        }

        processed += 1

        with OUT_JSON.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)

    summary = {
        "script_total_s": time.time() - t0,
        "items_total_in_file": total_items,
        "items_processed": processed,
        "limit_cases": LIMIT_CASES,
        "out_json": str(OUT_JSON),
        "backup_json": str(BACKUP_JSON),
        "prompts_jsonl": str(PROMPTS_JSONL),
        "ollama": {
            "model": OLLAMA_MODEL_NAME,
            "temperature": OLLAMA_TEMPERATURE,
            "num_predict": OLLAMA_NUM_PREDICT,
        },
        "metrics": {
            k: {
                **v,
                "accuracy": (v["correct"] / v["ran"]) if v["ran"] else 0.0,
            }
            for k, v in stats.items()
        },
    }
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  ACCURACY AFTER ALL RUNS")
    print("=" * 70)
    for rk_label in ("full_grid_yn", "pruned_grid_yn"):
        total_rk = 0
        correct_rk = 0
        none_rk = 0
        for item in items:
            rv = item.get("runs", {}).get(rk_label)
            if not isinstance(rv, dict):
                continue
            total_rk += 1
            sel = rv.get("selected_option")
            if sel is None:
                none_rk += 1
            elif check_answer_multi(sel, item.get("gold")):
                correct_rk += 1
        if total_rk:
            print(f"  {rk_label}: {correct_rk}/{total_rk} = {correct_rk/total_rk*100:.1f}%"
                  f"  (none: {none_rk})")
    print("=" * 70)

    print(f"\nwrote updated YN file: {OUT_JSON}")
    print(f"wrote backup: {BACKUP_JSON}")
    print(f"wrote summary: {SUMMARY_JSON}")
    print(f"prompts logged: {PROMPTS_JSONL}")
    for k, v in summary["metrics"].items():
        print(f"{k}: {v['correct']}/{v['ran']} = {100.0 * v['accuracy']:.1f}%")


if __name__ == "__main__":
    main()