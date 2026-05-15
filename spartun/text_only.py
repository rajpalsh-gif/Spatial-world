# ============================================================
# TEXT-ONLY + TEXT+RELATIONS BASELINE RERUN
# Improved prompts: chain-of-thought, YN/FR-specific, real few-shot examples
#
# INPUT:  flat JSON list  (same format as fr_full_grid_hard_incorrect_same_structure*.json
#                          OR  yn_pruned_grid_incorrect_only.json)
#         Each item has: identifier, q_id, q_type, question, gold,
#                        story (list), relations_used (list), candidate_answers (list)
# OUTPUT: single JSON file – all items + both run results appended in-place
# ============================================================

import json
import re
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from langchain_ollama import OllamaLLM
except Exception as e:
    raise RuntimeError(f"langchain_ollama required. pip install langchain-ollama\n{e!r}")

# ============================================================
# CONFIG  (mirror hyperparams from the other cells)
# ============================================================
TEXT_SRC_JSON  = Path("./improved/cool.json")
# To run on YN instead, swap to:
# TEXT_SRC_JSON  = Path("yn_pruned_grid_incorrect_only.json")

TEXT_OUT_JSON  = Path("coolbetterq32b.json")
TEXT_BACKUP    = TEXT_OUT_JSON.with_suffix(".backup.json")

OLLAMA_MODEL_NAME   = "qwen3:32b"
OLLAMA_TEMPERATURE  = 0.7
OLLAMA_NUM_PREDICT  = 5000

LIMIT_CASES: Optional[int] = 1000   # None = all
MAX_RETRIES = 2                    # retries if selected_option is None
START_INDEX = 0
RUN_TEXT_ONLY = False
RUN_TEXT_WITH_RELATIONS = True


# ============================================================
# FEW-SHOT EXAMPLE BANK
# (drawn from real stories in your attached data files – nothing fabricated)
# ============================================================

_YN_FEWSHOT = """\
── EXAMPLE A ──
Story: "A medium red star is covered by a block called HHH. We have a block named LLL. \
South of and to the west of another block named KKK there is block HHH. Disconnected from and \
farther from this block is block KKK with a little grey pentagon and a large grey pentagon. \
To the north of the large grey pentagon there is a large purple pentagon. It is covered by block KKK. \
North of a large grey hexagon is the large grey pentagon. The large grey hexagon is inside and \
touching block KKK. Block KKK covers a little purple hexagon."
Question: "Are all medium red stars south of the little grey pentagon?"
Candidates: Yes, No
Answer: {"justification": "The medium red star is inside block HHH. Block HHH is south of block KKK. The little grey pentagon is inside block KKK. So the star is south of the pentagon.", "selected_option": "Yes"}

── EXAMPLE B ──
Story: "We have two blocks, named AAA and BBB. Block AAA contains a medium yellow square and \
covers a blue square. Another medium yellow square is over medium yellow square number one. \
This thing is inside block AAA. Medium yellow square number one is over the blue object. \
Above, disconnected from and in front of block BBB is block AAA. Block BBB covers a medium black \
square and contains another medium black square."
Question: "Are all blue squares above all medium black things?"
Candidates: Yes, No
Answer: {"justification": "The blue square is in block AAA. The medium black squares are in block BBB. Block AAA is above block BBB, so the blue square is above the black squares.", "selected_option": "Yes"}
"""

_YN_REL_FEWSHOT = """\
── EXAMPLE A ──
Story: "We have two blocks, named AAA and BBB. Block AAA contains a medium yellow square and \
covers a blue square. Above, disconnected from and in front of block BBB is block AAA. Block BBB \
covers a medium black square and contains another medium black square."
Extracted spatial relations:
    - block AAA above block BBB
    - block AAA dc block BBB
    - blue square in block AAA
    - medium black square number one in block BBB
Question: "Are all blue squares above all medium black things?"
Candidates: Yes, No
Answer: {"justification": "AAA is above BBB.", "selected_option": "Yes"}

── EXAMPLE B ──
Story: "Two boxes, called DDD and EEE exist in an image. Box DDD with a midsize white rectangle \
is to the north of and behind box EEE. Box DDD covers another midsize white rectangle. A midsize \
orange rectangle is in this box. Box EEE covers a midsize white rectangle and has a midsize orange rectangle."
Extracted spatial relations:
    - box DDD above box EEE
    - midsize white rectangle number one in box DDD
    - midsize white rectangle number two in box DDD
    - midsize orange rectangle in box DDD
    - midsize white rectangle number three in box EEE
Question: "Is any midsize white rectangle to the north of all objects?"
Candidates: Yes, No
Answer: {"justification": "Condition does not hold for all objects.", "selected_option": "No"}
"""

_FR_FEWSHOT = """\
── EXAMPLE A ──
Story: "There exist three boxes, named one, two and three. A medium yellow apple is inside box one. \
Box one covers another medium yellow apple. Below box three there is it. Disconnected from and to \
the right of this box there is box three with a medium orange apple. Another medium orange apple \
and another medium orange apple are in this box. Medium orange apple number two is above medium \
orange apple number one. Over medium orange apple number three there is medium orange apple number one."
Question: "Where is box three relative to medium yellow apple number one?"
Candidates: above, below, left, right, front, behind, near, far, dc, ec, po, tpp, ntpp, tppi, ntppi
Step-by-step reasoning:
  1. Medium yellow apple number one is inside box one (containment: tpp or ntpp with box one).
  2. Box three is ABOVE and to the RIGHT of box one (from "below box three there is [box one]"
     and "to the right of this box there is box three").
  3. Box three is DISCONNECTED from (dc) box one.
  4. Since box three does not contain the yellow apple (it contains orange apples) and does not
     overlap box one, box three is DC with respect to the apple.
  5. Because box three is above and to the right of box one, and the apple is inside box one,
     box three is also ABOVE and to the RIGHT of the apple.
  6. No near/far, ec, po, or containment between box three and the yellow apple.
Answer: {"justification": "Box three is above and right of box one and is DC from it; the yellow apple inside box one inherits these relations.", "selected_option": ["RIGHT", "ABOVE", "DC"]}

── EXAMPLE B ──
Story: "We have two blocks, named AAA and BBB. Block AAA contains a medium yellow square and \
covers a blue square. Another medium yellow square is over medium yellow square number one. \
This thing is inside block AAA. Medium yellow square number one is over the blue object. \
Above, disconnected from and in front of block BBB is block AAA. Block BBB covers a medium black \
square and contains another medium black square."
Question: "Where is AAA relative to medium black square number one?"
Candidates: above, below, left, right, front, behind, near, far, dc, ec, po, tpp, ntpp, tppi, ntppi
Step-by-step reasoning:
  1. Medium black square number one is inside block BBB (touch-edge → tpp relation with BBB).
  2. Block AAA is above, disconnected from, and in front of block BBB.
  3. AAA does not overlap BBB at all (dc) so AAA is also DC from the black square.
  4. AAA is above BBB → AAA is ABOVE the black square inside BBB.
  5. AAA is in front of BBB → AAA is FRONT relative to the black square.
  6. No left/right or near/far qualifiers stated.
Answer: {"justification": "Block AAA is above, in front of, and disconnected from block BBB, so the same relations hold to medium black square number one inside BBB.", "selected_option": ["ABOVE", "FRONT", "DC"]}
"""

_FR_REL_FEWSHOT = """\
── EXAMPLE A ──
Story: "A block named HHH contain a medium grey thing and a red hexagon. The medium purple hexagon \
is covered by block HHH."
Extracted spatial relations:
    - medium purple hexagon tpp block HHH
Question: "Where is block HHH regarding the purple object?"
Candidates: above, below, left, right, front, behind, near, far, dc, ec, po, tpp, ntpp, tppi, ntppi
Answer: {"justification": "The purple hexagon is tpp of HHH, so HHH is the inverse: tppi.", "selected_option": ["TPPI"]}
"""


# ============================================================
# PROMPT BUILDERS  (YN and FR separated, chain-of-thought)
# ============================================================

def _cand_str(candidates: List[str]) -> str:
    return ", ".join(candidates) if candidates else "(none)"


def build_yn_text_only_prompt(story: List[str], question: str, candidates: List[str]) -> str:
    story_text = " ".join(s.strip() for s in story if isinstance(s, str) and s.strip())
    return f"""\
You are a spatial reasoning expert. Answer YES/NO questions about a scene described in a story.

RULES:
- Read the story and answer the spatial question.
- Pay attention to quantifiers: "all" means every instance must satisfy the condition;
  "any" means at least one.
- Your final answer must be one of the provided candidates.
- Output ONLY valid JSON – no text outside the JSON.
- Put your reasoning in the "justification" field. Do NOT write reasoning outside the JSON.
- JSON schema:
  {{"justification": "your reasoning", "selected_option": "Yes or No"}}

WORKED EXAMPLES:
{_YN_FEWSHOT}
────────────────────────────────────────────────────────────
NOW ANSWER THE FOLLOWING:

Story: {story_text}

Question: {question}
Candidates: {_cand_str(candidates)}

Output ONLY the JSON object with your reasoning inside "justification".
"""


def build_yn_text_relations_prompt(story: List[str], relations: List[str], question: str, candidates: List[str]) -> str:
    story_text = " ".join(s.strip() for s in story if isinstance(s, str) and s.strip())
    rel_block = "\n".join(f"  - {r}" for r in relations) if relations else "  (none extracted)"
    return f"""\
You are a spatial reasoning expert. Answer YES/NO questions using a story AND a list of extracted
spatial relations.

RULES:
- Use the extracted relations as the main evidence.
- You may use the story only if the relations seem incomplete.
- Choose exactly one candidate.
- Output ONLY valid JSON with reasoning inside "justification".
- JSON schema:
  {{"justification": "your reasoning", "selected_option": "Yes or No"}}

WORKED EXAMPLES:
{_YN_REL_FEWSHOT}
────────────────────────────────────────────────────────────
NOW ANSWER THE FOLLOWING:

Story: {story_text}

Extracted spatial relations:
{rel_block}

Question: {question}
Candidates: {_cand_str(candidates)}

Output ONLY the JSON object with your reasoning inside "justification".
"""


def build_fr_text_only_prompt(story: List[str], question: str, candidates: List[str]) -> str:
    story_text = " ".join(s.strip() for s in story if isinstance(s, str) and s.strip())
    return f"""\
You are a spatial reasoning expert. Answer spatial fill-in questions.
The answer may require MULTIPLE labels from the candidates (e.g. a box can be ABOVE + RIGHT + DC
relative to another object simultaneously).

RULES – READ CAREFULLY:
1. Directional relations (above/below/left/right/front/behind) describe the position of the
   QUERY ENTITY relative to the REFERENCE ENTITY.
2. Topological relations:
   - dc  = completely disconnected (no shared border or interior)
   - ec  = externally connected (touching borders, no shared interior)
   - po  = partial overlap
   - tpp = tangential proper part (inside, touching inner boundary) – query IS INSIDE reference
   - ntpp= non-tangential proper part (fully inside, not touching boundary)
   - tppi= tpp inverse (reference is inside query)
   - ntppi= ntpp inverse (reference is fully inside query)
3. Inheritance rule: if entity A is inside container X, and container X is above container Y,
   then A is also above Y (and above everything inside Y).
4. Select ALL relations from the candidates that are true simultaneously.
- Output ONLY valid JSON. Schema (strict):
  {{ "justification": "describe your reasoning here", "selected_option": ["LABEL", ...]}}
  (For a single label, still use a list: ["LABEL"])

WORKED EXAMPLES:
{_FR_FEWSHOT}
────────────────────────────────────────────────────────────
NOW ANSWER THE FOLLOWING:

Story: {story_text}

Question: {question}
Candidates: {_cand_str(candidates)}

Use the available information and output JSON.
"""


def build_fr_text_relations_prompt(story: List[str], relations: List[str], question: str, candidates: List[str]) -> str:
    story_text = " ".join(s.strip() for s in story if isinstance(s, str) and s.strip())
    rel_block = "\n".join(f"  - {r}" for r in relations) if relations else "  (none extracted)"
    return f"""\
You are a spatial reasoning expert. Answer spatial fill-in questions using a story AND a list of
extracted spatial relations.

RULES – READ CAREFULLY:
1. Use the extracted relations as the main evidence.
2. Direction words describe the first entity relative to the second.
3. For topology: dc = disconnected, ec = touching only, po = overlap, tpp/ntpp = inside, tppi/ntppi = inverse.
4. Prefer directly stated relations over inherited or indirect ones.
5. If several labels seem possible, return only the most certain ones.
- Output ONLY valid JSON. Schema (strict):
  {{"justification": "describe your reasoning here", "selected_option": ["LABEL", ...]}}

WORKED EXAMPLES:
{_FR_REL_FEWSHOT}
────────────────────────────────────────────────────────────
NOW ANSWER THE FOLLOWING:

Story: {story_text}

Extracted spatial relations:
{rel_block}

Question: {question}
Candidates: {_cand_str(candidates)}

Use the available information and output JSON.
"""


# ============================================================
# HELPERS
# ============================================================

def _strip_fences(t: str) -> str:
    t = re.sub(r"^```(?:json)?\s*", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t.strip())
    return t.strip()

def try_parse_json_dict(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    t = _strip_fences(str(text))
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {"raw_parse_fail": text}

def normalize_answer(ans: Any) -> Optional[str]:
    if ans is None:
        return None
    if isinstance(ans, list):
        ans = ans[0] if ans else ""
    return str(ans).strip().lower()

def check_answer_exact(selected: Any, gold: Any) -> Optional[bool]:
    """Exact-set match (used by FR pipeline elsewhere). Also works for YN."""
    if selected is None or gold is None:
        return None
    def to_set(x):
        if isinstance(x, str):
            return {x.strip().lower()}
        if isinstance(x, list):
            return {str(i).strip().lower() for i in x if str(i).strip()}
        return set()
    s, g = to_set(selected), to_set(gold)
    return bool(s and g and s == g)

def check_answer_multi(selected: Any, gold: Any) -> Optional[bool]:
    """Partial-overlap match (more lenient, used for YN baseline)."""
    if selected is None or gold is None:
        return None
    sel = selected if isinstance(selected, list) else [selected]
    gld = gold if isinstance(gold, list) else [gold]
    s = {normalize_answer(x) for x in sel if x is not None} - {None}
    g = {normalize_answer(x) for x in gld if x is not None} - {None}
    if not s or not g:
        return None
    return bool(s & g)


# ============================================================
# OLLAMA
# ============================================================

_LM_TEXT: Optional[OllamaLLM] = None

def _get_lm() -> OllamaLLM:
    global _LM_TEXT
    if _LM_TEXT is None:
        _LM_TEXT = OllamaLLM(
            model=OLLAMA_MODEL_NAME,
            temperature=OLLAMA_TEMPERATURE,
            num_predict=OLLAMA_NUM_PREDICT,
        )
    return _LM_TEXT

def call_ollama_text(prompt: str) -> str:
    return _get_lm().invoke(prompt)

def run_prompt_with_retry(prompt: str, label: str) -> Tuple[str, Dict[str, Any]]:
    """Call Ollama, retry up to MAX_RETRIES if selected_option is None."""
    for attempt in range(MAX_RETRIES + 1):
        t0 = time.time()
        raw = call_ollama_text(prompt)
        elapsed = time.time() - t0
        parsed = try_parse_json_dict(raw)
        if parsed.get("selected_option") is not None:
            return raw, parsed
        if attempt < MAX_RETRIES:
            print(f"    [{label}] attempt {attempt+1}: selected_option=None, retrying...")
    return raw, parsed


# ============================================================
# MAIN
# ============================================================

def run_text_baselines():
    if not TEXT_SRC_JSON.exists():
        raise FileNotFoundError(f"Input not found: {TEXT_SRC_JSON}")

    # Load existing results for resume support
    _all: List[Dict] = []
    if TEXT_OUT_JSON.exists():
        try:
            prev = json.loads(TEXT_OUT_JSON.read_text(encoding="utf-8"))
            _all = prev if isinstance(prev, list) else []
        except Exception:
            _all = []
    done_keys = {(r.get("identifier"), r.get("q_id")) for r in _all if isinstance(r, dict)}

    if not TEXT_BACKUP.exists():
        shutil.copy2(TEXT_SRC_JSON, TEXT_BACKUP)
        print(f"backup -> {TEXT_BACKUP}")

    items: List[Dict] = json.loads(TEXT_SRC_JSON.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("SRC_JSON must be a JSON list")

    total = len(items)
    processed = skipped = 0
    stats = {
        "text_only_correct": 0, "text_only_ran": 0,
        "text_relations_correct": 0, "text_relations_ran": 0,
    }

    for idx, item in enumerate(items):
        if idx < START_INDEX:
            continue
        if LIMIT_CASES is not None and processed >= LIMIT_CASES:
            break
        if not isinstance(item, dict):
            continue

        identifier = item.get("identifier", "")
        q_id       = item.get("q_id")
        q_type     = str(item.get("q_type", "FR")).upper()
        question   = item.get("question") or ""
        gold       = item.get("gold")
        story      = item.get("story") or []
        if isinstance(story, str):
            story = [story]
        relations  = item.get("relations_used") or []
        if isinstance(relations, str):
            relations = [relations]
        candidates = item.get("candidate_answers") or (
            ["Yes", "No"] if q_type == "YN"
            else ["above","below","left","right","front","behind","near","far",
                  "dc","ec","po","tpp","ntpp","tppi","ntppi"]
        )

        key = (identifier, q_id)
        if key in done_keys:
            skipped += 1
            continue

        is_yn = (q_type == "YN")
        print(f"\n[{idx+1}/{total}] {identifier} q_id={q_id} q_type={q_type}")
        print(f"  Q: {question}")

        raw_text = None
        parsed_text: Dict[str, Any] = {}
        sel_text = None
        correct_text = None
        t_text = 0.0

        # ── TEXT ONLY ──
        if RUN_TEXT_ONLY:
            if is_yn:
                p_text = build_yn_text_only_prompt(story, question, candidates)
            else:
                p_text = build_fr_text_only_prompt(story, question, candidates)

            t0 = time.time()
            raw_text, parsed_text = run_prompt_with_retry(p_text, "text_only")
            t_text = time.time() - t0

            sel_text = parsed_text.get("selected_option")
            if is_yn:
                correct_text = check_answer_multi(sel_text, gold)
            else:
                correct_text = check_answer_exact(sel_text, gold)

            stats["text_only_ran"] += 1
            stats["text_only_correct"] += int(bool(correct_text))
            print(f"  text_only -> {sel_text} | correct={correct_text}")
        else:
            p_text = None
            print("  text_only -> skipped")

        raw_rels = None
        parsed_rels: Dict[str, Any] = {}
        sel_rels = None
        correct_rels = None
        t_rels = 0.0

        # ── TEXT + RELATIONS ──
        if RUN_TEXT_WITH_RELATIONS:
            if is_yn:
                p_rels = build_yn_text_relations_prompt(story, relations, question, candidates)
            else:
                p_rels = build_fr_text_relations_prompt(story, relations, question, candidates)

            t0 = time.time()
            raw_rels, parsed_rels = run_prompt_with_retry(p_rels, "text_relations")
            t_rels = time.time() - t0

            sel_rels = parsed_rels.get("selected_option")
            if is_yn:
                correct_rels = check_answer_multi(sel_rels, gold)
            else:
                correct_rels = check_answer_exact(sel_rels, gold)

            stats["text_relations_ran"] += 1
            stats["text_relations_correct"] += int(bool(correct_rels))
            print(f"  text_rels  -> {sel_rels} | correct={correct_rels}")
        else:
            p_rels = None
            print("  text_rels  -> skipped")

        # ── RECORD ──
        record = {
            "identifier": identifier,
            "q_id": q_id,
            "q_type": q_type,
            "question": question,
            "gold": gold,
            "candidate_answers": candidates,
            "story": story,
            "relations_used": relations,
            "runs": {
                "text_only": {
                    "prompt_used": p_text,
                    "raw": raw_text,
                    "parsed": parsed_text,
                    "selected_option": sel_text,
                    "chain_of_thought": parsed_text.get("chain_of_thought"),
                    "justification": parsed_text.get("justification"),
                    "correct": correct_text,
                    "time_s": round(t_text, 2),
                },
                "text_with_relations": {
                    "prompt_used": p_rels,
                    "raw": raw_rels,
                    "parsed": parsed_rels,
                    "selected_option": sel_rels,
                    "chain_of_thought": parsed_rels.get("chain_of_thought"),
                    "justification": parsed_rels.get("justification"),
                    "correct": correct_rels,
                    "time_s": round(t_rels, 2),
                },
            },
            "ollama": {
                "model": OLLAMA_MODEL_NAME,
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": OLLAMA_NUM_PREDICT,
            },
        }

        _all.append(record)
        done_keys.add(key)
        processed += 1

        # Save after every item (safe resume)
        TEXT_OUT_JSON.write_text(json.dumps(_all, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── SUMMARY ──
    def acc(c, n): return round(100.0 * c / n, 1) if n else 0.0
    print("\n" + "=" * 60)
    print(f"text_only     : {stats['text_only_correct']}/{stats['text_only_ran']} = {acc(stats['text_only_correct'], stats['text_only_ran'])}%")
    print(f"text_relations: {stats['text_relations_correct']}/{stats['text_relations_ran']} = {acc(stats['text_relations_correct'], stats['text_relations_ran'])}%")
    print(f"skipped (already done): {skipped}")
    print(f"Output -> {TEXT_OUT_JSON}")


run_text_baselines()

