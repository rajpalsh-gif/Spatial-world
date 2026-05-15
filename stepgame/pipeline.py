# -*- coding: utf-8 -*-
"""
Merged pipeline (no external judge):
1) Paragraph → (relations, full_grid, pruned_grid, normalized_pruned_grid)
2) Run EXACT grid prompts (coords → coarse → straight/diagonal) on the generated pruned grid
3) Print & return the reasoning/justifications from all 3 prompts
   - Coordinates step: "reason"
   - Coarse step: "justification"
   - Specialized step (straight/diagonal): "justification"

Requirements:
- Python 3.8+
- langchain-ollama (for Ollama Llama 3.1)
"""

import re, json, time
from typing import List, Dict, Tuple, Optional, Any

# =========================
# LLM backend (Ollama Llama 3.1)
# =========================
from langchain_ollama import OllamaLLM

# ---- Ollama (Llama 3.1), temp=0.7 as configured
lm2 = OllamaLLM(model="qwen3:14b", temperature=0.7)

# ---------------- Timing wrapper ----------------
def timed_invoke(model, prompt: str, timeout_s: int = 890) -> str:
    try:
        start = time.time()
        resp = model.invoke(prompt)
        if (time.time() - start) > timeout_s:
            return "TIMEOUT"
        return resp
    except Exception:
        return "TIMEOUT"

# ============================================================
# ============ PART A: Paragraph → Relations & Grids =========
# (same as before; stage1/2 prompts kept verbatim)
# ============================================================
def _parse_jsonish(output: str) -> Optional[Dict]:
    try:
        return json.loads(re.search(r"\{.*\}", output, re.S).group(0))
    except Exception:
        return None

def _recover_coarse_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'"(?:coarse|coarse_relation)"\s*:\s*"(straight|diagonal|overlap)"', text, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r'\b(straight|diagonal|overlap)\b', text, re.I)
    if m:
        return m.group(1).lower()
    return None

def _recover_relation_sentence_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r'\b([AB])\b\s+is\s+(?:the\s+)?([a-zA-Z\- ]+?)\s+(?:of|to)\s+\b([AB])\b\.?',
        r'\b([AB])\b\s+is\s+(?:the\s+)?([a-zA-Z\- ]+?)\s+\b([AB])\b\.?',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.I)
        if m:
            subj = m.group(1).upper()
            rel = m.group(2).strip()
            obj = m.group(3).upper()
            return f"{subj} is {rel} of {obj}."
    return None

def _try_call_llama_json(prompt: str) -> Optional[dict]:
    try:
        out = lm2.invoke(prompt)
        if isinstance(out, dict):
            return out
        return _parse_jsonish(out if isinstance(out, str) else str(out))
    except Exception:
        return None
def _call_llama_raw_and_parsed(prompt: str) -> Tuple[str, Optional[Dict]]:
    """
    Run lm2 with your exact prompt and return:
      raw_text (str), parsed_dict (dict|None)
    We never change the prompt; we just capture the raw output and a parsed dict if possible.
    """
    try:
        out = lm2.invoke(prompt)
        if isinstance(out, dict):
            raw = json.dumps(out, ensure_ascii=False)
            parsed = out
        else:
            raw = out if isinstance(out, str) else str(out)
            parsed = _parse_jsonish_robust(raw)
        return raw, parsed
    except Exception:
        return "TIMEOUT", None

# new (captures numbers but we only care about the names)
_COORDS_PAIR_RE = re.compile(
    r'([A-Za-z0-9_]+)\s+at\s+Row\((-?\d+)\),\s*Col\((-?\d+)\)\s*;\s*'
    r'([A-Za-z0-9_]+)\s+at\s+Row\((-?\d+)\),\s*Col\((-?\d+)\)\.?'
)

def parse_agents_from_coords(coords_line: str):
    m = _COORDS_PAIR_RE.search(coords_line or "")
    if m:
        # groups: 1=name1, 2=row1, 3=col1, 4=name2, 5=row2, 6=col2
        return m.group(1), m.group(4)
    return None, None


REL_NORMALIZE = {
    "top-right": "upper-right",
    "top-left": "upper-left",
    "bottom-right": "lower-right",
    "bottom-left": "lower-left",
}
ALLOWED = {
    "left","right","above","below",
    "upper-left","upper-right","lower-left","lower-right","overlap"
}

def _norm_rel_token(tok: str) -> str:
    t = (tok or "").strip().lower().replace(" ", "_")
    t = REL_NORMALIZE.get(t, t)
    return t

_SENT_RE = re.compile(
    r'^\s*([A-Za-z0-9_]+)\s+is\s+([a-zA-Z\- ]+?)\s+of\s+([A-Za-z0-9_]+)\s*\.?\s*$'
)

def parse_triplet_from_sentence_field(sentence_field: str) -> Optional[Tuple[str,str,str]]:
    """
    Parses: "<obj1> is <relation> of <obj2>"
    Returns (obj1_token, normalized_relation, obj2_token), or None if it doesn't match.
    """
    m = _SENT_RE.match(sentence_field or "")
    if not m:
        return None
    obj1, rel_raw, obj2 = m.group(1), m.group(2), m.group(3)
    rel = _norm_rel_token(rel_raw)
    if rel not in ALLOWED:
        return None
    return obj1, rel, obj2

def relation_A_to_B_from_sentence(sentence_field: str) -> Optional[str]:
    """
    The stage-2 prompt always uses A/B placeholders.
    We convert the sentence to the final label for A→B.
      - If "A is <rel> of B" → rel
      - If "B is <rel> of A" → inverse(rel)
    """
    parsed = parse_triplet_from_sentence_field(sentence_field)
    if not parsed:
        return None
    obj1, rel, obj2 = parsed
    if obj1.upper() == "A" and obj2.upper() == "B":
        return rel
    if obj1.upper() == "B" and obj2.upper() == "A":
        return INV_REL.get(rel, rel)
    # If someone outputs real symbols (rare), fall back to “A/B only” expectation:
    return None

def flip_relation(label: str) -> str:
    lab = (label or "").strip().lower()
    return INV_REL.get(lab, lab)

def need_flip_for_order(question: str, coords_line: str) -> bool:
    qx, qy = parse_question_agents(question or "")
    ax, ay = parse_agents_from_coords(coords_line or "")
    if not (qx and qy and ax and ay):
        return False
    # If model returned the two in reversed order, we must flip the relation
    if ax == qy and ay == qx:
        return True
    return False

def build_stage1_prompt(sentence: str) -> str:
    exam="""
Input: "B is at A’s 9 o’clock."
Output:
{"reason":"<think>9 o’clock → left only (one axis horizontal) → straight.</think>","coarse_relation":"straight"}

Input: "B is at A’s 12 o’clock."
Output:
{"reason":"<think>12 o’clock → above only (vertical axis) → straight.</think>","coarse_relation":"straight"}

Input: "B is parallel to A and at A’s 3 o’clock."
Output:
{"reason":"<think>Ignore 'parallel'; 3 o’clock means right only → straight.</think>","coarse_relation":"straight"}

Input: "A is at B’s 6 o’clock."
Output:
{"reason":"<think>6 o’clock means below → single vertical axis → straight.</think>","coarse_relation":"straight"}

Input: "B is at A’s 4 o’clock."
Output:
{"reason":"<think>4 o’clock means down and right (two axes) → diagonal.</think>","coarse_relation":"diagonal"}

Input: "B is at A’s 10 o’clock."
Output:
{"reason":"<think>10 o’clock means up and left (two axes) → diagonal.</think>","coarse_relation":"diagonal"}

Input: "B is below A and slightly to the left."
Output:
{"reason":"<think>'Below' + 'left' combine two axes → diagonal.</think>","coarse_relation":"diagonal"}

Input: "B is north-west of A."
Output:
{"reason":"<think>'North' (up) + 'west' (left) → two axes → diagonal.</think>","coarse_relation":"diagonal"}

"""
    return f"""
You are an expert in spatial reasoning.
Given the following sentence describing the spatial position between object A and object B:

"{sentence}"

Step 1: Decide whether this relation type is one of:
- straight  → left / right / above / below
- diagonal  → upper-left / upper-right / lower-left / lower-right
- overlap   → same position (overlap)

Before you output an answer think carefully
1. The sentence could be referring to 2 relations like A is lower and to the right of B that means you consider 2 relations ie. diagonal
2. Look out for word ambiguity like clock wise positions( A is 3:00 clock to B means A is right to B , 9:00 clock means A is left of B, 6:00 clock means A is below B, 12:00 clock means A is above B ) . This extends to directions liek northwest, here west would be left and east would be right, similarly north south is above below. if it says 45 degrees then it is diagonal for sure.
3. Think carefully and reason well before considering a relation type, does this sentence refer to 2 relations between A and B in the lower-upper or left-right both axes? so diagnol, does it mean only 1 relation? so straight, does it mean they re both on each other, overlap so overlap.
4. Important clarification: Think very carefully about these before answering:
Words such as “parallel”, “horizontal”, “vertical”, “side by side, “small gap between them”,“45 degrees”,
“on the same plane”,“center of the clock” or vague phrases like “over there”, “around”, “next to each other”,
do NOT tell you any direction. They are only background or filler words.
Ignore them completely when deciding direction.
Only terms that directly describe direction (left, right, above, below)
or exact clock positions (1–12 o’clock) or clear compass words
(east(right), west(left), north(above), south(below)) matter.So, south west is lower-left and northwest is upper-left. if "front" is mentioned, it means "upper".
Clock-position meanings: Please check these carefully if any clock position(only number between 1-12) mentioned in the sentence:
12 o’clock → above
3 o’clock → right
6 o’clock → below
9 o’clock → left
2 o’clock → upper-right
4 o’clock → lower-right
8 o’clock → lower-left
10 o’clock → upper-left
1 ≈ upper-right, 5 ≈ lower-right, 7 ≈ lower-left, 11 ≈ upper-left
Output your answer as JSON with the key "coarse" and value being one of:
"straight", "diagonal", "overlap".
Strict evidence rule:
• Do NOT invent clock hours unless the input mentions them.
• Ignore filler words: parallel, vertical, horizontal, front, back, corner, over there.
• If both 'upper/top' and 'right' appear, relation CANNOT be lower-right.
 CHECK: If only one direction is present and some vague word it is only that one direction otherwise if 2 directions are present (upper/lower + left/right, “top-right”, “north-west”, or “between hour1 and hour2”), only if there are 2 directions mentioned exactly not in vague words but exact 2 directions then output the diagonal .
these are some examples, use them to guide your reasoning:
{exam}

Respond ONLY in JSON format as:
{{"reason":"add your reason here under<think></think>",
"coarse_relation": "<straight|diagonal|overlap>"}}
"""

def build_stage2_prompt(sentence: str, coarse: str) -> Optional[str]:
    if coarse == "overlap":
        shift="overlap only no change"
        opts="overlap"
        exam="overlap"
        rule="overlap"
    elif coarse == "straight":
        opts = "left, right, above, below"
        shift='''left ↔ right
        above ↔ below'''
        exam="""
Input: "B is at A’s 9 o’clock."
Output:
{"reasoning":"<think>9 o’clock lies directly left of center. Sentence says 'B is at A’s 9 o’clock' → B is left of A.  Relation = left.</think>","sentence":"B is left of A."}

Input: "B is above A."
Output:
{"reasoning":"<think>Keyword ‘above’ directly mentioned in the sentence. So, B is above A.Relation = above.</think>","sentence":"B is above A."}

Input: "A and B are in a horizontal line with B on the right."
Output:
{"reasoning":"<think>Ignore ‘Horizontal line’ (filler) according to Rule 3, this will be ignored . ‘B on the right’ → B is right of A. So relation = right.</think>","sentence":"B is right of A."}

Input: "A and B are side by side with B at the bottom and A on the top."
Output:
{"reasoning":"<think>Ignore ‘side by side’ (filler) according to Rule 3. B is at bottom and A is at Top means B is below A or A is above B as bottom is below and Top is above. So we take B is below A. Relation = below.</think>","sentence":"B is below A."}

Input: "B is parallel to A and at A’s 12 o’clock."
Output:
{"reasoning":"<think>Ignore ‘parallel’ (filler) according to Rule 3. 12 o’clock = above. Since A is at 12 o clock means A is above B.Relation = above.</think>","sentence":"A is above B."}
 """
        rule="CHECK: If only one direction is present and some vague word it is only that one direction otherwise if 2 directions are present (upper/lower + left/right, “top-right”, “north-west”, or “between hour1 and hour2”), only if there are 2 directions mentioned exactly not in vague words but exact 2 directions then output the diagonal (UR/UL/LR/LL) and never drop an axis (2–3→UR, 10–11→UL, 4–5→LR, 7–8→LL)."
    elif coarse == "diagonal":
        opts = "upper-left, upper-right, lower-left, lower-right"
        shift='''upper-left ↔ lower-right
        upper-right ↔ lower-left'''
        rule="""
CHECK: If two directions are present (upper/lower + left/right, “top-right”, “north-west”, or “between h1 and h2”), always output the diagonal (UR/UL/LR/LL) and never drop an axis (2–3→UR, 10–11→UL, 4–5→LR, 7–8→LL).
"""
        exam="""
Input: "If A is the center of a clock face, B is located between 2 and 3."
Output:
{"reasoning":"<think>2 o’clock = up and right (upper-right diagonal). 3 o'clock= right(straight). So for getting between 2 and 3, we go between upper-right and right, since this refers to 2 directions, we go to the diagonal one, and answer is upper-right. So, B is upper-right of A. so relation = upper-right.</think>","sentence":"B is upper-right of A."}

Input: "A is diagonally below B to the right at a 45 degree angle."
Output:
{"reasoning":"<think>'below B to the right' means down and right → lower-right. Other words like "45 degrees" and "Diagonally" are filler words as per Rule 3 and need to be ignored. So, A is lower-right of B. Relation = lower-right.</think>","sentence":"A is lower-right of B."}
Input: "B is slightly off-center to the top left and A is slightly off-center to the bottom right."
Output:
{"reasoning":"<think>Sentence states B (top-left) and A (bottom-right). Thus B is above and to the left of A (upper-left). So, B is upper-left of A. Final relation = upper-left.</think>","sentence":"B is upper-left of A."}
 
Input: " A is above B and to the left of it."
Output:
{"reasoning":"<think>‘Above’ means up and ‘to the left’ means left. So, A is upper-left of B. Relation = upper-left.</think>","sentence":"A is upper-left of B."}

Input: "B is to the bottom-right of A."
Output:
{"reasoning":"<think>‘Bottom-right’ means below and right. No clock mentioned → do not invent one. So, B is lower-right of A. Relation = lower-right.</think>","sentence":"B is lower-right of A."}

Input: "B is there and A is at the 10 position of a clock face."
Output:
{"reasoning":"<think>10 o’clock = up and left (upper-left). Sentence states A’s position to B. So, A is upper-left of B. Relation = upper-left.</think>","sentence":"A is upper-left of B"}

Input: "A is south-west of B."
Output:
{"reasoning":"<think>‘south-west’ = down and left (lower-left). So, A is lower-left of B. Relation = lower-left.</think>","sentence":"A is lower-left of B."}

"""
    else:
        return None
    return f"""
You are given a sentence describing the spatial position of agent A relative to agent B.
That means what is the position of A if you are standing at B.
Now, based on that, decide the exact direction of A relative to B.

Sentence: "{sentence}"

Choose ONLY from: {opts}
Now, based on these rules choose the exact direction of A relative to B:

Rule 1 — Identify the relation:
Determine the fine-grained spatial relation implied by the sentence from the set: left, right, above, below, upper-left, upper-right, lower-left, lower-right, overlap.


Rule 2 — Recheck and output:
Think carefully, verify both the relation and directionality, then output your reasoning followed by the final relation between A and B.
Rule 3 — Important clarification:
Think very carefully about these before answering:
Words such as “parallel”, “horizontal”, “vertical”, “side by side, “small gap between them”,“45 degrees”,
“on the same plane”,“center of the clock” or vague phrases like “over there”, “around”, “next to each other”,
do NOT tell you any direction. They are only background or filler words.
Ignore them completely when deciding direction.

Only terms that directly describe direction (left, right, above, below)
or exact clock positions (1–12 o’clock) or clear compass words like (east(right), west(left), north(above), south(below)) matter.So, south west is lower-left and northwest is upper-left. if "front" is mentioned, it means "upper".
Clock-position meanings: Please check these carefully if any clock position(only number between 1-12) mentioned in the sentence:
12 o’clock → above
3 o’clock → right
6 o’clock → below
9 o’clock → left
2 o’clock → upper-right
4 o’clock → lower-right
8 o’clock → lower-left
10 o’clock → upper-left
1 ≈ upper-right, 5 ≈ lower-right, 7 ≈ lower-left, 11 ≈ upper-left
Strict evidence rule:
• Do NOT invent clock hours unless the input mentions them.
• Ignore filler words: parallel, vertical, horizontal, front, back, corner, over there.
• If both 'upper/top' and 'right' appear, relation CANNOT be lower-right.
{rule}

These examples show how to ignore filler words like “parallel”, “over there”, or “vertical”,
and focus only on the actual directional cue (left, right, above, below, or clock position).
{exam}
Output ONLY THESE 2 KEYS as JSON, give your reasoning and the final sentence in the format specified below:
{{
  "reasoning": "your reasoning under <think></think>",
  "sentence": "Output the sentence in the format "[object1] is [relation] of [object2]",
}}
"""
REL_ALIASES = {
    # normalize common variants to your canonical 9 labels
    "left": "left", "to the left of": "left",
    "right": "right", "to the right of": "right",
    "above": "above", "on top of": "above", "top": "above", "upper": "above",
    "below": "below", "bottom": "below", "under": "below",

    "upper-left": "upper-left", "upper left": "upper-left", "top-left": "upper-left", "topleft": "upper-left",
    "upper-right": "upper-right", "upper right": "upper-right", "top-right": "upper-right", "topright": "upper-right",
    "lower-left": "lower-left", "lower left": "lower-left", "bottom-left": "lower-left", "bottomleft": "lower-left",
    "lower-right": "lower-right","lower right":"lower-right","bottom-right":"lower-right","bottomright":"lower-right",

    "overlap": "overlap", "same cell": "overlap", "same position": "overlap",
}

def _norm_rel_label(s: str) -> Optional[str]:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" - ", "_").replace(" ", "_")  # coalesce to hyphen style
    # try exact; then try with spaces
    if s in REL_ALIASES:
        return REL_ALIASES[s]
    s2 = s.replace("-", " ")
    return REL_ALIASES.get(s2)

# Accepts: "A is above of B", "A is above to B", "A is above B"
_REL_SENT_RX_WITH_CONN = re.compile(
    r'\b([AB])\b\s+is\s+(?:the\s+)?(.+?)\s+(?:of|to)\s+\b([AB])\b\.?',
    flags=re.I
)
_REL_SENT_RX_NO_CONN = re.compile(
    r'\b([AB])\b\s+is\s+(?:the\s+)?(.+?)\s+\b([AB])\b\.?',
    flags=re.I
)

def parse_relation_sentence_to_triplet_with_mapping(sentence_text: str, a_symbol: str, b_symbol: str) -> Optional[Dict[str,str]]:
    """
    Parse a stage-2 'sentence' like 'B is upper-left of A.' or 'A is below B.'
    Then map A/B back to the original letters: A→a_symbol, B→b_symbol.
    """
    if not sentence_text:
        return None
    s = sentence_text.strip()

    m = _REL_SENT_RX_WITH_CONN.search(s) or _REL_SENT_RX_NO_CONN.search(s)
    if not m:
        return None

    subj_ab = m.group(1).upper()
    rel_txt = m.group(2).strip()
    obj_ab  = m.group(3).upper()

    rel = _norm_rel_label(rel_txt)
    if rel not in {
        "left","right","above","below",
        "upper-left","upper-right","lower-left","lower-right","overlap"
    }:
        return None

    head = a_symbol if subj_ab == "A" else b_symbol
    tail = a_symbol if obj_ab  == "A" else b_symbol
    return {"head": head, "relation": rel, "tail": tail}



# ---------- A/B rewriting + stage pipeline ----------
SENT_SPLIT_RE = re.compile(r'\s*\.\s*')
UPPER_LETTER = r'([A-Z])'

def _clean(s: str) -> str:
    return s.strip().rstrip('.').strip()

def _letters_in_sentence(s: str) -> List[str]:
    return re.findall(r'\b[A-Z]\b', s)

def _rewrite_pair_sentence(s: str) -> Optional[Dict]:
    s_clean = _clean(s)

    m = re.search(rf'\b{UPPER_LETTER}\b\s+is\s+at\s+\b{UPPER_LETTER}\b[’\']?s\s+((?:1[0-2]|[1-9]))', s_clean, re.I)
    if m:
        head, tail, hour = m.group(1), m.group(2), m.group(3)
        return {"ab_sentence": f"A is at B's {hour} o'clock.", "head": head, "tail": tail}

    m = re.search(rf'\b{UPPER_LETTER}\b\s+is\s+(?:positioned\s+)?(left|right|above|below)\s+(?:to|of)\s+\b{UPPER_LETTER}\b', s_clean, re.I)
    if m:
        head, rel, tail = m.group(1), m.group(2).lower(), m.group(3)
        return {"ab_sentence": f"A is {rel} of B.", "head": head, "tail": tail}

    m = re.search(rf'\b{UPPER_LETTER}\b\s+is\s+(?:placed\s+)?on\s+the\s+top\s+of\s+\b{UPPER_LETTER}\b', s_clean, re.I)
    if m:
        head, tail = m.group(1), m.group(2)
        return {"ab_sentence": "A is above B.", "head": head, "tail": tail}

    m = re.search(rf'\b{UPPER_LETTER}\b\s+is\s+(above|below|left|right)\s+\b{UPPER_LETTER}\b', s_clean, re.I)
    if m:
        head, rel, tail = m.group(1), m.group(2).lower(), m.group(3)
        return {"ab_sentence": f"A is {rel} of B.", "head": head, "tail": tail}

    # fallback: first two single letters become A/B placeholders
    letters = _letters_in_sentence(s_clean)
    letters = [x for x in letters if len(x) == 1]
    if len(letters) >= 2:
        head, tail = letters[0], letters[1]
        ab_sentence = s_clean.replace(head, "A").replace(tail, "B")
        return {"ab_sentence": ab_sentence, "head": head, "tail": tail}
    return None

def _run_stage_pipeline(ab_sentence: str, a_symbol: Optional[str] = None, b_symbol: Optional[str] = None) -> Optional[Dict]:
    # Stage 1 (coarse)
    p1 = build_stage1_prompt(ab_sentence)
    raw1, out1 = _call_llama_raw_and_parsed(p1)
    if not out1:
        coarse = _recover_coarse_from_text(raw1)
        if not coarse:
            return {"ok": False, "stage1": {"prompt": p1, "raw": raw1}, "stage2": {}}
        out1 = {"coarse": coarse}

    # Normalize key
    if "coarse" not in out1 and "coarse_relation" in out1:
        out1["coarse"] = out1["coarse_relation"]
    coarse = (out1.get("coarse") or out1.get("coarse_relation") or _recover_coarse_from_text(raw1) or "").strip().lower()
    if not coarse:
        return {"ok": False, "stage1": {"prompt": p1, "raw": raw1, "parsed": out1}, "stage2": {}}

    # Stage 2 (fine) — prompt depends on coarse
    p2 = build_stage2_prompt(ab_sentence, coarse)
    if not p2:
        return {"ok": False, "stage1": {"prompt": p1, "raw": raw1, "parsed": out1}, "stage2": {}}

    raw2, out2 = _call_llama_raw_and_parsed(p2)

    # recover the 'sentence' even if wrapped in prose
    sent2 = ""
    if out2 and isinstance(out2, dict):
        sent2 = (out2.get("sentence") or "").strip()
    if not sent2:
        sent2 = _extract_sentence_field_from_text(raw2) or ""
    if not sent2:
        sent2 = _recover_relation_sentence_from_text(raw2) or ""

    rec = {
        "stage1": {
            "prompt": p1,
            "raw": raw1,
            "parsed": out1,
            "coarse": coarse,
            "reason": out1.get("reason") or out1.get("reasoning"),
        },
        "stage2": {
            "prompt": p2,
            "raw": raw2,
            "parsed": out2 if out2 else None,
            "sentence": sent2 or None
        }
    }

    if not sent2:
        rec["ok"] = False
        return rec

    # Map A/B back to original symbols passed in
    a_sym = a_symbol or "A"
    b_sym = b_symbol or "B"
    trip = parse_relation_sentence_to_triplet_with_mapping(sent2, a_symbol=a_sym, b_symbol=b_sym)
    if not trip:
        rec["ok"] = False
        return rec

    rec["ok"] = True
    rec["final_relation"] = trip["relation"]
    rec["final_triplet"]  = trip
    return rec



def _balanced_json_object_from_text(text: str) -> Optional[str]:
    """Return the first balanced top-level JSON object substring {...} from messy text."""
    if not text:
        return None
    start_idx = None
    depth = 0
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx is not None:
                    return text[start_idx:i+1]
    return None

def _parse_jsonish_robust(text: str) -> Optional[dict]:
    """
    Robust JSON recovery:
    1) Try full parse
    2) If that fails, try a balanced-brace slice
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    blob = _balanced_json_object_from_text(text)
    if blob:
        try:
            return json.loads(blob)
        except Exception:
            return None
    return None

def _extract_sentence_field_from_text(text: str) -> Optional[str]:
    """
    If model returned extra chatter, try to recover the `"sentence": "..."` field directly.
    """
    if not text:
        return None
    m = re.search(r'"sentence"\s*:\s*"([^"]+)"', text, flags=re.S)
    if m:
        return m.group(1).strip()
    return None


def extract_relations_from_paragraph(text: str) -> Dict:
    raw_sents = [s for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    relations: List[Dict] = []
    for raw in raw_sents:
        parsed = _rewrite_pair_sentence(raw)
        if not parsed:
            continue
        pipe = _run_stage_pipeline(parsed["ab_sentence"], a_symbol=parsed["head"], b_symbol=parsed["tail"])
        if not pipe or not pipe.get("ok") or not pipe.get("final_triplet"):
            continue
        relations.append(pipe["final_triplet"])  # already mapped back to original symbols

    return {"relations": relations}


# ---------- Graph build + grids ----------
VEC = {
    "left":        (-1,  0),
    "right":       ( 1,  0),
    "above":       ( 0,  1),
    "below":       ( 0, -1),
    "upper-left":  (-1,  1),
    "upper-right": ( 1,  1),
    "lower-left":  (-1, -1),
    "lower-right": ( 1, -1),
    "overlap":     ( 0,  0),
}

def place_relations(relations: List[Dict], start_symbol: Optional[str]=None, start_coord=(0,0)) -> Dict[str, Tuple[int,int]]:
    coords: Dict[str, Tuple[int,int]] = {}
    if not relations:
        return coords
    if start_symbol is None:
        start_symbol = relations[0]["tail"]
    coords[start_symbol] = start_coord
    changed = True
    while changed:
        changed = False
        for r in relations:
            h, rel, t = r["head"], r["relation"], r["tail"]
            if rel not in VEC: continue
            dx, dy = VEC[rel]
            if t in coords and h not in coords:
                coords[h] = (coords[t][0] + dx, coords[t][1] + dy); changed = True
            elif h in coords and t not in coords:
                coords[t] = (coords[h][0] - dx, coords[h][1] - dy); changed = True
    return coords

def _bounds(coords: Dict[str, Tuple[int,int]]):
    xs = [x for x,_ in coords.values()] or [0]
    ys = [y for _,y in coords.values()] or [0]
    return min(xs), max(xs), min(ys), max(ys)

def _build_token_grid(coords: Dict[str, Tuple[int,int]], keep: set = None):
    if keep is not None:
        coords = {s:xy for s,xy in coords.items() if s in keep}
    if not coords:
        return [[]], []
    minx, maxx, miny, maxy = _bounds(coords)
    W = maxx - minx + 1; H = maxy - miny + 1
    cell: Dict[Tuple[int,int], List[str]] = {}
    for sym,(x,y) in coords.items():
        cell.setdefault((x,y), []).append(sym)
    for k in cell: cell[k].sort()
    grid = []
    for y in range(maxy, miny-1, -1):
        row = []
        for x in range(minx, maxx+1):
            toks = cell.get((x,y), [])
            row.append("/".join(toks) if toks else "_")
        grid.append(row)
    kept = sorted(coords.keys())
    return grid, kept
def _extract_sentence_field_from_text(text: str) -> Optional[str]:
    """
    Try to recover a JSON object from mixed text and return its 'sentence' field.
    Falls back to a regex for `"sentence": "..."` if JSON parsing fails.
    """
    if not text:
        return None
    # greedy JSON block recovery
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                s = obj.get("sentence")
                if isinstance(s, str) and s.strip():
                    return s.strip()
    except Exception:
        pass
    # fallback: direct regex capture
    m2 = re.search(r'"sentence"\s*:\s*"([^"]+)"', text)
    return m2.group(1).strip() if m2 else None

def _render_labeled(grid: List[List[str]]) -> str:
    if not grid or not grid[0]: return ""
    H = len(grid)
    header = "    " + "  ".join(f"C{j+1}" for j in range(len(grid[0])))
    lines = [header]
    for i in range(H):
        lines.append(f"R{i+1:<2}" + "  ".join(grid[i]))
    return "\n".join(lines)

def render_full_grid_str(coords: Dict[str, Tuple[int,int]]) -> str:
    grid, _ = _build_token_grid(coords, keep=None)
    return _render_labeled(grid)

def render_pruned_grid_str(coords: Dict[str, Tuple[int,int]], keep_syms: Tuple[str,str]) -> str:
    keep = {s for s in keep_syms if s}
    grid, kept = _build_token_grid(coords, keep=keep)
    if not kept or not grid or not grid[0]:
        return "    C1\nR1  -"
    H, W = len(grid), len(grid[0])
    non_empty = [(i,j) for i in range(H) for j in range(W) if grid[i][j] != "_"]
    if not non_empty:
        return "    C1\nR1  -"
    r0 = min(i for i,_ in non_empty); r1 = max(i for i,_ in non_empty)
    c0 = min(j for _,j in non_empty); c1 = max(j for _,j in non_empty)
    cropped = [row[c0:c1+1] for row in grid[r0:r1+1]]
    return _render_labeled(cropped)

# =========================
# Pruned-grid normalizer
# =========================
EMPTY_MARKERS = {"_", "-"}

def is_empty_cell(s: str) -> bool:
    return s.strip() in EMPTY_MARKERS or s.strip() == ""

def parse_pruned_grid(grid_text: str):
    lines = [ln.rstrip("\n") for ln in (grid_text or "").splitlines() if ln.strip() != ""]
    if not lines: return [], [], []
    header_line = lines[0]
    header_tokens = [t for t in header_line.split() if not re.match(r'R\d+|Row\(\d+\)', t)]
    if not header_tokens: header_tokens = ["C1"]
    row_labels, rows = [], []
    for ln in lines[1:]:
        parts = ln.split()
        if not parts: continue
        if re.match(r'^R\d+$', parts[0]) or re.match(r'^Row\(\d+\)$', parts[0]):
            row_label = parts[0]; cells = parts[1:]
        else:
            continue
        if len(cells) < len(header_tokens):
            cells = cells + ["_"] * (len(header_tokens) - len(cells))
        cells = cells[:len(header_tokens)]
        row_labels.append(row_label); rows.append(cells)
    if not rows and len(lines) >= 2:
        ln = lines[1].split()
        if ln and (re.match(r'^R\d+$', ln[0]) or re.match(r'^Row\(\d+\)$', ln[0])):
            row_labels = [ln[0]]; rows = [ln[1:2] or ["_"]]; header_tokens = header_tokens[:1]
    return rows, row_labels, header_tokens

def drop_empty_rows_and_cols(rows):
    if not rows: return rows
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols == 0: return rows
    rows = [r + ["_"] * (n_cols - len(r)) for r in rows]
    keep_row = [any(not is_empty_cell(c) for c in row) for row in rows]
    rows_kept = [row for row, k in zip(rows, keep_row) if k]
    if not rows_kept: return [["_"]]
    n_cols = len(rows_kept[0])
    keep_col = []
    for j in range(n_cols):
        col_vals = [rows_kept[i][j] for i in range(len(rows_kept))]
        keep_col.append(any(not is_empty_cell(c) for c in col_vals))
    if not any(keep_col):
        keep_col = [True] + [False] * (n_cols - 1)
    rows_final = []
    for r in rows_kept:
        rows_final.append([c for c, kc in zip(r, keep_col) if kc])
    return rows_final

def pad_to_2x2(rows):
    if not rows: rows = [["_"]]
    if len(rows) > 2: rows = rows[:2]
    max_cols = max((len(r) for r in rows), default=0)
    if max_cols == 0:
        rows = [["_"]]; max_cols = 1
    rows = [r + ["_"] * (max_cols - len(r)) for r in rows]
    if len(rows[0]) > 2: rows = [r[:2] for r in rows]
    while len(rows) < 2:
        rows.append(["_"] * len(rows[0]))
    for i in range(len(rows)):
        while len(rows[i]) < 2:
            rows[i].append("_")
    return rows

def format_as_2x2(rows):
    assert len(rows) == 2 and len(rows[0]) == 2 and len(rows[1]) == 2
    c11, c12 = rows[0][0], rows[0][1]
    c21, c22 = rows[1][0], rows[1][1]
    lines = [
        "      Col(1) Col(2)",
        f"Row(1)  {c11}      {c12}",
        f"Row(2)  {c21}      {c22}",
    ]
    return "\n".join(lines)

def normalize_pruned_grid(pruned_grid_text: str) -> str:
    rows, _, _ = parse_pruned_grid(pruned_grid_text)
    rows = drop_empty_rows_and_cols(rows)
    rows = pad_to_2x2(rows)
    return format_as_2x2(rows)

# ============================================================
# ============ PART B: EXACT 3 GRID PROMPTS ==================
# (coords-from-pruned, coarse, straight, diagonal)
# ============================================================
# ---- ADD THIS HELPER (needed by validate_single_sample) ----
import re

_Q_PATTERNS = [
    re.compile(r"relation\s+of\s+the\s+agent\s+([A-Za-z0-9_]+)\s+to\s+the\s+agent\s+([A-Za-z0-9_]+)\??", re.I),
    re.compile(r"relation\s+of\s+([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)\??", re.I),
    re.compile(r"relation\s+between\s+([A-Za-z0-9_]+)\s+and\s+([A-Za-z0-9_]+)\??", re.I),
]

def parse_question_agents(q: str):
    """
    Returns (head_symbol, tail_symbol) as uppercase single tokens, e.g. ('U','A')
    for 'What is the relation of the agent U to the agent A?'.
    Falls back to the first two standalone capital letters if patterns don’t match.
    """
    s = (q or "").strip()
    for rx in _Q_PATTERNS:
        m = rx.search(s)
        if m:
            return m.group(1).upper(), m.group(2).upper()
    picks = re.findall(r"\b[A-Z]\b", s)
    return (picks[0], picks[1]) if len(picks) >= 2 else ("", "")

def build_grid_prune_echo_prompt(sample: Dict[str, Any]) -> str:
    pruned = sample.get("pruned_grid", "").strip()
    return f"You are given this pruned grid. Repeat it EXACTLY:\n\n{pruned}"

def parse_choice_and_justification(text: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        obj = json.loads(text)
        return obj.get("selected_option"), obj.get("justification")
    except Exception:
        pass
    # Minimal fallback:
    m = re.search(r'"selected_option"\s*:\s*"([^"]+)"', text)
    sel = m.group(1) if m else None
    j = re.search(r'"justification"\s*:\s*"([^"]+)"', text)
    just = j.group(1) if j else None
    return sel, just

def parse_reason_and_coords(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract 'reason' and 'coordinates' from the coords JSON."""
    try:
        obj = json.loads(text)
        return obj.get("reason"), obj.get("coordinates")
    except Exception:
        pass
    # fallback: try to find a coordinates line
    m = re.search(
        r'([A-Za-z0-9_]+)\s+at\s+Row\((-?\d+)\),\s*Col\((-?\d+)\)\s*;\s*([A-Za-z0-9_]+)\s+at\s+Row\((-?\d+)\),\s*Col\((-?\d+)\)\.?',
        text
    )
    
    coords = m.group(0) if m else None
    return None, coords

def build_coords_request_line_from_question(question: Optional[str]) -> str:
    m = re.search(r"relation\s+of\s+the\s+agent\s+([A-Za-z0-9_]+)\s+to\s+the\s+agent\s+([A-Za-z0-9_]+)", question or "", re.I)
    if not m:
        m = re.search(r"relation\s+of\s+([A-Za-z0-9_]+)\s+to\s+([A-Za-z0-9_]+)", question or "", re.I)
    if not m:
        return ""
    x, y = m.group(1), m.group(2)
    return (
        f'Now give the coordinates of agent {x} and agent {y} ONLY in this format: '
        f'"{x} at Row(_), Col(_); {y} at Row(_), Col(_)."'
    )
USE_NORMALIZED_IN_COORDS = True



def build_coords_prompt_from_pruned_grid(sample: Dict[str, Any]) -> str:
    pruned_raw = sample.get("pruned_grid", "").strip()
    pruned = normalize_pruned_grid(pruned_raw) if USE_NORMALIZED_IN_COORDS else pruned_raw
    coords_hint = build_coords_request_line_from_question(sample.get("question", ""))  # <-- keep

    examples_block = """
Coordinate examples (2×2):

1) Same column
      Col(1) Col(2)
Row(1)  X      _
Row(2)  Y      _
Coordinates — X at Row(1), Col(1); Y at Row(2), Col(1).

2) Same row
      Col(1) Col(2)
Row(1)  Y      X
Row(2)  _      _
Coordinates — X at Row(1), Col(2); Y at Row(1), Col(1).

3) Diagonal (upper-left vs lower-right)
      Col(1) Col(2)
Row(1)  X      _
Row(2)  _      Y
Coordinates — X at Row(1), Col(1); Y at Row(2), Col(2).

4) Diagonal (upper-right vs lower-left)
      Col(1) Col(2)
Row(1)  _      X
Row(2)  Y      _
Coordinates — X at Row(1), Col(2); Y at Row(2), Col(1).

5) Overlap
      Col(1) Col(2)
Row(1) X/Y     _
Row(2)  _      _
Coordinates — X at Row(1), Col(1); Y at Row(1), Col(1). when / is used for 2 characters it liek A/Y it means overlap
""".strip()

    prompt = f"""
Instructions:
- Use ONLY the 2×2 grid below with exact headers: Row(1), Row(2), Col(1), Col(2).
- Output the single line in the coordinates field using EXACT wording:
  "<name of object 1 at Row(_), Col(_); name of object 2 at Row(_), Col(_)>."
- Do not add any explanations.

{examples_block}

GRID:
{pruned}

Reason very carefully; think it through among all examples before you answer. Provide your reasoning under <think> </think>.
What are the coordinates of the two objects present?
'_' means empty cell dont include this in your coordinates, if elements are lined together liek A/Y that means they share the same cell and same row so same coordinates
Think carefully and look at the each of the rules before you answer
Row parsing only. After the row label (Row(1) or Row(2)), read that row only until the newline. Do not read header tokens (Col(1), Col(2)) as cells. Ignore spaces only.

Exactly two cells per row. Read two cell tokens in order: the first is Col(1), the second is Col(2). '_’ or '-' are valid empty cells—do not skip them. Letters/digits = elements.

Coordinates & output formatting. For each row, assign coordinates as (that row, that column index). Never output _ or - as an object. If a cell shows X/Y, split it and output both symbols with the same coordinates, in the order they appear (e.g., Q/X → Q at Row(1), Col(1); X at Row(1), Col(1)). If a cell has a single symbol S, output S at Row(r), Col(c). Never mention the coordinates of an empty space or '_' in coordinates.Newline ⇒ next row; restart the two-cell read.
{coords_hint}
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "reason": "<think> your reason here </think>",
  "coordinates": "Provide the Row(), Col() for both objects here exactly as in the examples."
}}
""".strip()
    return prompt

def build_relation_prompt_from_coords(coords_line: str) -> str:
    rules_block = """
Rules (2×2, plain text):
- If Row and Col are the same for X and Y → overlap. STOP. Do not consider straight or diagonal.
- If exactly that is true-> rows are the same (but columns differ) → straight: left or right. STOP. Diagonals are disallowed here.
  • If X is in Col(1) and Y is in Col(2) → X is left of Y as Col(1) is less than Col(2) so object in Col(1) should be left of object in Col(2) .
  • If X is in Col(2) and Y is in Col(1) → X is right of Y as Col(2) is more than Col(1) so object in Col(2) should be right of object in Col(1).
- If exactly this is true-> columns are the same (but rows differ) → straight: above or below.STOP. Diagonals are disallowed here.
  • If X is in Row(-1) and Y is in Row(-2) → X is above of Y as Row(-1) is more than Row(-2) so object in Row(-1) should be above object in Row(-2).
  • If X is in Row(-2) and Y is in Row(-1) → X is below of Y as Row(-2) is less than Row(-1) so object in Row(-2) should be below object in Row(-1).
- If both row AND column differ → diagonal:
  • X Row(-1)&Col(1) vs Y Row(-2)&Col(2) → X is upper-left of Y as Row(-1) is more than Row(-2) and Col(1) is less than Col(2) so object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).
  • X Row(-1)&Col(2) vs Y Row(-2)&Col(1) → X is upper-right of Y as Row(-1) is more than Row(-2) and Col(2) is more than Col(1) so object in Row(-1), Col(2) should be upper-right of object in Row(-2), Col(1).
  • X Row(-2)&Col(1) vs Y Row(-1)&Col(2) → X is lower-left of Y as Row(-2) is less than Row(-1) and Col(1) is less than Col(2) so object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).
  • X Row(-2)&Col(2) vs Y Row(-1)&Col(1) → X is lower-right of Y as Row(-2) is less than Row(-1) and Col(2) is more than Col(1) so object in Row(-2), Col(2) should be lower-right of object in Row(-1), Col(1).
  """.strip()

    few_shots = """
Example A
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(1)."
Output:
{
  "justification": "Same column; X is in the above row and Y is in the below row. X is above Y as Row(-1) is more than Row(-2) and object in Row(-1) should be above object in Row(-2).",
  "selected_option": "above"
}

Example B
Input: "X at Row(-1), Col(2); Y at Row(-1), Col(1)."
Output:
{
  "justification": "Same row; X is in the right column and Y is in the left column. X is right of Y as Col(2) is more than Col(1) so object in Col(2) should be right of object in Col(1).",
  "selected_option": "right"
}

Example C
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(2)."
Output:
{
  "justification": "Rows and columns differ; X is top-left while Y is bottom-right. X is upper-left of Y as Row(-1) is more than Row(-2) and Col(1) is less than Col(2) so object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).",
  "selected_option": "upper-left"
}

Example D
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Output:
{
  "justification": "Rows and columns differ; X is bottom-left while Y is top-right. X is lower-left of Y as Row(-2) is less than Row(-1) and Col(1) is less than Col(2) so object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).",
  "selected_option": "lower-left"
}

Example E
Input: "X at Row(-1), Col(1); Y at Row(-1), Col(1)."
Output:
{
  "justification": "Both share the same row and the same column (same cell). X overlaps Y as Row(-1)=Row(-1) and Col(1)=Col(1) so both objects should overlap.",
  "selected_option": "overlap"
}
""".strip()

    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<reason based ONLY on the given coordinates>",
  "selected_option": "<one of: upper-left, upper-right, lower-left, lower-right, above, below, left, right, overlap>"
}}

Instructions:
- There is NO grid here. You are given ONLY a coordinates line like:
  "<X at Row(_), Col(_); Y at Row(_), Col(_).>"
- Decide the single relation label for X relative to Y using these simple rules.

{rules_block}

Few-shot (coordinates only → relation):
{few_shots}

Now use these rules and examples.

Coordinates:
{coords_line}

Think carefully about each of the three points and check in order:
If X and Y are in the same row and the same column, pick overlap.

If exactly one matches (same row or same column), pick a straight label: above / below / left / right.

If row and column are both different, pick a diagonal label: upper-left / upper-right / lower-left / lower-right.
""".strip()
    return prompt

def build_relation_coarse_prompt_from_coords(coords_line: str) -> str:
    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<reason based ONLY on the given coordinates>",
  "selected_option": "<one of: straight, diagonal, overlap>"
}}

You are given ONLY a coordinates line like:
"<X at Row(_), Col(_); Y at Row(_), Col(_).>"
Each Row(-n) and Col(n) is the index of that object’s row and column.

Decide the single relation label for X relative to Y using these simple rules.

Rules (2×2, plain text):
- If Row and Col are the exact same number for X and Y → overlap. STOP. Do not consider straight or diagonal.
- If exactly that is true-> rows are the exact same number (but columns differ) → straight:  STOP. Diagonals are disallowed here.
  • If X is in Col(1) and Y is in Col(2), Col(1) does not equal to Col(2) so straight
  • If X is in Col(2) and Y is in Col(1), Col(2) does not equal to Col(1) so straight
- If exactly this is true-> columns are the exact same number (but rows differ) → straight: STOP. Diagonals are disallowed here.
  • If X is in Row(-1) and Y is in Row(-2), Row(-1) does not equal to Row(-2) so straight
  • If X is in Row(-2) and Y is in Row(-1), Row(-2) does not equal to Row(-1) so straight
- If both row AND column differ → diagonal:
  • X Row(-1)&Col(1) vs Y Row(-2)&Col(2) , Row(-1) does not equal to Row(-2) & Col(1) does not equal to Col(2) so diagonal
  • X Row(-1)&Col(2) vs Y Row(-2)&Col(1), Row(-1) does not equal to Row(-2)  & Col(2) does not equal to Col(1) so diagonal
  • X Row(-2)&Col(1) vs Y Row(-1)&Col(2), Row(-2) does not equal to Row(-1) & Col(1) does not equal to Col(2) so diagonal
  • X Row(-2)&Col(2) vs Y Row(-1)&Col(1), Row(-2) does not equal to Row(-1) & Col(2) does not equal to Col(1) so diagonal

Few-shot (coordinates only → coarse class):

Example Straight
Input: "X at Row(-1), Col(1); Y at Row(-1), Col(2)."
Output:
{{
  "justification": "Rows are the same (Row(-1)), columns differ (Col(1) vs Col(2)); therefore straight, since Col(1) does not equal to Col(2).",
  "selected_option": "straight"
}}

Example Straight
Input: "X at Row(-2), Col(2); Y at Row(-2), Col(1)."
Output:
{{
  "justification": "Rows are the same (Row(-2)), columns differ (Col(2) vs Col(1)); therefore straight, since Col(2) does not equal to Col(1).",
  "selected_option": "straight"
}}

Example Straight
Input: "X at Row(-1), Col(2); Y at Row(-2), Col(2)."
Output:
{{
  "justification": "Columns are the same (Col(2)), rows differ (Row(-1) vs Row(-2)); therefore straight since Row(-1) does not equal to Row(-2).",
  "selected_option": "straight"
}}

Example Straight
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(1)."
Output:
{{
  "justification": "Columns are the same (Col(1)), rows differ (Row(-2) vs Row(-1)); therefore straight, since Row(-2) does not equal to Row(-1).",
  "selected_option": "straight"
}}

Example Diagonal
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(2)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal, since Row(-1) does not equal to Row(-2) and Col(1) does not equal to Col(2).",
  "selected_option": "diagonal"
}}

Example Diagonal
Input: "X at Row(-1), Col(2); Y at Row(-2), Col(1)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal, since Row(-1) does not equal to Row(-2) and Col(2) does not equal to Col(1).",
  "selected_option": "diagonal"
}}

Example Diagonal
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal, since Row(-2) does not equal to Row(-1) and Col(1) does not equal to Col(2).",
  "selected_option": "diagonal"
}}

Example Diagonal
Input: "X at Row(-2), Col(2); Y at Row(-1), Col(1)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal, since Row(-2) does not equal to Row(-1) and Col(2) does not equal to Col(1) .",
  "selected_option": "diagonal"
}}

Example Overlap
Input: "X at Row(-1), Col(1); Y at Row(-1), Col(1)."
Output:
{{
  "justification": "Row and column indices are identical for both; therefore overlap, since Row(-1) equal to Row(-1) and Col(1) equal to Col(1).",
  "selected_option": "overlap"
}}

Now use these rules and examples.

Coordinates:
{coords_line}
THINK CAREFULLY ABOUT THESE RULES in this strict order. The first true rule wins:

- Same row AND same column → overlap.
- Same row OR same column, but not both → straight.
- Otherwise (both row and column are different) → diagonal.

Guardrails:
- If both row and column are different, you must choose “diagonal”.
- If you ever choose “straight” while also saying “rows differ and columns differ”, correct the choice to “diagonal”.

Think carefully based on these. Output JSON only.

""".strip()
    return prompt


def build_relation_prompt_from_coords(coords_line: str) -> str:
    rules_block = """
Rules (2×2, plain text):
- If Row and Col are the same for X and Y → overlap. STOP. Do not consider straight or diagonal.
- If exactly that is true-> rows are the same (but columns differ) → straight: left or right. STOP. Diagonals are disallowed here.
  • If X is in Col(1) and Y is in Col(2) → X is left of Y as Col(1) is less than Col(2) so object in Col(1) should be left of object in Col(2) .
  • If X is in Col(2) and Y is in Col(1) → X is right of Y as Col(2) is more than Col(1) so object in Col(2) should be right of object in Col(1).
- If exactly this is true-> columns are the same (but rows differ) → straight: above or below.STOP. Diagonals are disallowed here.
  • If X is in Row(-1) and Y is in Row(-2) → X is above of Y as Row(-1) is more than Row(-2) so object in Row(-1) should be above object in Row(-2).
  • If X is in Row(-2) and Y is in Row(-1) → X is below of Y as Row(-2) is less than Row(-1) so object in Row(-2) should be below object in Row(-1).
- If both row AND column differ → diagonal:
  • X Row(-1)&Col(1) vs Y Row(-2)&Col(2) → X is upper-left of Y as Row(-1) is more than Row(-2) and Col(1) is less than Col(2) so object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).
  • X Row(-1)&Col(2) vs Y Row(-2)&Col(1) → X is upper-right of Y as Row(-1) is more than Row(-2) and Col(2) is more than Col(1) so object in Row(-1), Col(2) should be upper-right of object in Row(-2), Col(1).
  • X Row(-2)&Col(1) vs Y Row(-1)&Col(2) → X is lower-left of Y as Row(-2) is less than Row(-1) and Col(1) is less than Col(2) so object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).
  • X Row(-2)&Col(2) vs Y Row(-1)&Col(1) → X is lower-right of Y as Row(-2) is less than Row(-1) and Col(2) is more than Col(1) so object in Row(-2), Col(2) should be lower-right of object in Row(-1), Col(1).
  """.strip()

    few_shots = """
Example A
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(1)."
Output:
{
  "justification": "Same column; X is in the above row and Y is in the below row. X is above Y as Row(-1) is more than Row(-2) and object in Row(-1) should be above object in Row(-2).",
  "selected_option": "above"
}

Example B
Input: "X at Row(-1), Col(2); Y at Row(-1), Col(1)."
Output:
{
  "justification": "Same row; X is in the right column and Y is in the left column. X is right of Y as Col(2) is more than Col(1) so object in Col(2) should be right of object in Col(1).",
  "selected_option": "right"
}

Example C
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(2)."
Output:
{
  "justification": "Rows and columns differ; X is top-left while Y is bottom-right. X is upper-left of Y as Row(-1) is more than Row(-2) and Col(1) is less than Col(2) so object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).",
  "selected_option": "upper-left"
}

Example D
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Output:
{
  "justification": "Rows and columns differ; X is bottom-left while Y is top-right. X is lower-left of Y as Row(-2) is less than Row(-1) and Col(1) is less than Col(2) so object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).",
  "selected_option": "lower-left"
}

Example E
Input: "X at Row(-1), Col(1); Y at Row(-1), Col(1)."
Output:
{
  "justification": "Both share the same row and the same column (same cell). X overlaps Y as Row(-1)=Row(-1) and Col(1)=Col(1) so both objects should overlap.",
  "selected_option": "overlap"
}
""".strip()

    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<reason based ONLY on the given coordinates>",
  "selected_option": "<one of: upper-left, upper-right, lower-left, lower-right, above, below, left, right, overlap>"
}}

Instructions:
- There is NO grid here. You are given ONLY a coordinates line like:
  "<X at Row(_), Col(_); Y at Row(_), Col(_).>"
- Decide the single relation label for X relative to Y using these simple rules.

{rules_block}

Few-shot (coordinates only → relation):
{few_shots}

Now use these rules and examples.

Coordinates:
{coords_line}

Think carefully about each of the three points and check in order:
If X and Y are in the same row and the same column, pick overlap.

If exactly one matches (same row or same column), pick a straight label: above / below / left / right.

If row and column are both different, pick a diagonal label: upper-left / upper-right / lower-left / lower-right.
""".strip()
    return prompt

def build_relation_coarse_prompt_from_coords(coords_line: str) -> str:
    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<reason based ONLY on the given coordinates>",
  "selected_option": "<one of: straight, diagonal, overlap>"
}}

You are given ONLY a coordinates line like:
"<X at Row(_), Col(_); Y at Row(_), Col(_).>"
Each Row(n) and Col(n) is the index of that object’s row and column.

Decide the single relation label for X relative to Y using these simple rules.

Rules (2\u00d72, plain text):
- If Row and Col are the same for X and Y \u2192 overlap. STOP. Do not consider straight or diagonal.
- If exactly that is true-> rows are the same (but columns differ) \u2192 straight:  STOP. Diagonals are disallowed here.
  \u2022 If X is in Col(1) and Y is in Col(2) and rows numbers are same
  \u2022 If X is in Col(2) and Y is in Col(1) and rows numbers are same
- If exactly this is true-> columns are the same (but rows differ) \u2192 straight: STOP. Do not consider Diagonals they are disallowed here.
  \u2022 If X is in Row(1) and Y is in Row(2) and cols numbers are same
  \u2022 If X is in Row(2) and Y is in Row(1) and cols numbers are same
  DONT CONSIDER ROW AND COLUMN INDEX MATCHINGFOR 2 ELEMENTS FOR STRAIGHT LABELS, FOR THAT YOU CAN GIVE DIAGONAL
- If both row AND column differ \u2192 diagonal:
  \u2022 X Row(1)&Col(1) vs Y Row(2)&Col(2)
  \u2022 X Row(1)&Col(2) vs Y Row(2)&Col(1)
  \u2022 X Row(2)&Col(1) vs Y Row(1)&Col(2)
  \u2022 X Row(2)&Col(2) vs Y Row(1)&Col(1)

Few-shot (coordinates only \u2192 coarse class):

Example S-row-1
Input: "X at Row(1), Col(1); Y at Row(1), Col(2)."
Output:
{{
  "justification": "Rows are the same (Row(1)), columns differ (Col(1) vs Col(2)); therefore straight.",
  "selected_option": "straight"
}}

Example S-row-2
Input: "X at Row(2), Col(2); Y at Row(2), Col(1)."
Output:
{{
  "justification": "Rows are the same (Row(2)), columns differ (Col(2) vs Col(1)); therefore straight.",
  "selected_option": "straight"
}}

Example S-col-1
Input: "X at Row(1), Col(2); Y at Row(2), Col(2)."
Output:
{{
  "justification": "Columns are the same (Col(2)), rows differ (Row(1) vs Row(2)); therefore straight.",
  "selected_option": "straight"
}}

Example S-col-2
Input: "X at Row(2), Col(1); Y at Row(1), Col(1)."
Output:
{{
  "justification": "Columns are the same (Col(1)), rows differ (Row(2) vs Row(1)); therefore straight.",
  "selected_option": "straight"
}}

Example D-UL
Input: "X at Row(1), Col(1); Y at Row(2), Col(2)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal.",
  "selected_option": "diagonal"
}}

Example D-UR
Input: "X at Row(1), Col(2); Y at Row(2), Col(1)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal.",
  "selected_option": "diagonal"
}}

Example D-LL
Input: "X at Row(2), Col(1); Y at Row(1), Col(2)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal.",
  "selected_option": "diagonal"
}}

Example D-LR
Input: "X at Row(2), Col(2); Y at Row(1), Col(1)."
Output:
{{
  "justification": "Rows differ and columns differ simultaneously; therefore diagonal.",
  "selected_option": "diagonal"
}}

Example OVL
Input: "X at Row(1), Col(1); Y at Row(1), Col(1)."
Output:
{{
  "justification": "Row and column indices are identical for both; therefore overlap.",
  "selected_option": "overlap"
}}

Now use these rules and examples.

Coordinates:
{coords_line}

Think carefully about each of the three points and check in order:
If X and Y are in the same row and the same column, pick overlap.
If exactly one matches (same row or same column), pick straight.
If row and column are both different, pick diagonal.
""".strip()
    return prompt

def build_relation_straight_prompt_from_coords(coords_line: str, question: str) -> str:
    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<detailed reason based ONLY on the given coordinates>",
  "selected_option": "<one of: left, right, above, below>"
}}

You are given ONLY a coordinates line like:
"<object1 at Row(_), Col(_);  object2 at Row(_), Col(_).>"
Each Row(n) and Col(n) is the index of that object’s row and column.

This case is already known to be STRAIGHT (same row XOR same column). Decide the precise straight label for object1 relative to  object2.

Rules (straight only):
- If exactly that is true-> rows are the same but columns differ:
  \u2022 If object1 is in Col(1) and  object2 is in Col(2) \u2192 left.
  \u2022 If object1 is in Col(2) and  object2 is in Col(1) \u2192 right.
- If exactly that is true-> columns are the same but rows differ:
  \u2022 If object1 is in Row(1) and  object2 is in Row(2) \u2192 above.
  \u2022 If object1 is in Row(2) and  object2 is in Row(1) \u2192 below.

Few-shot (coordinates only \u2192 precise straight label):

Example L
Input: "object1 at Row(1), Col(1); object2 at Row(1), Col(2)."
Output:
{{
  "justification": "Both share Row(1). object1 is in Col(1) while object2 is in Col(2), so object1 is to the left of  object2.",
  "selected_option": "left"
}}

Example R
Input: "object1 at Row(1), Col(2); object2 at Row(1), Col(1)."
Output:
{{
  "justification": "Both share Row(2). object1 occupies Col(2) and object2 Col(1); hence object1 is to the right of  object2.",
  "selected_option": "right"
}}

Example A
Input: "object1 at Row(1), Col(1); object2 at Row(2), Col(1)."
Output:
{{
  "justification": "Both share Col(2). object1 lies in Row(1) while object2 lies in Row(2); therefore object1 is above  object2.",
  "selected_option": "above"
}}

Example B
Input: "object1 at Row(2), Col(1); object2 at Row(1), Col(1)."
Output:
{{
  "justification": "Both share Col(1). object1 lies in Row(2) while object2 lies in Row(1); therefore object1 is below  object2.",
  "selected_option": "below"
}}

Now use these rules and examples.

Coordinates:
{coords_line}

Now, think carefully and reason and output the direction as per the coordinates
""".strip()
    return prompt

def build_relation_diagonal_prompt_from_coords(coords_line: str, question: str) -> str:
    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<detailed reason based ONLY on the given coordinates>",
  "sentence": "object1 is <one of: upper-left, upper-right, lower-left, lower-right> of object2"
}}

You are given ONLY a coordinates line like:
"<X at Row(_), Col(_); Y at Row(_), Col(_).>"
Each Row(-n) and Col(n) is the index of that object’s row and column.

This case is already known to be DIAGONAL (rows differ AND columns differ). Decide the precise diagonal label for X relative to Y.

Rules (diagonal only):
• X Row(-1)&Col(1) vs Y Row(-2)&Col(2) → X is upper-left of Y as Row(-1) is more than Row(-2) and Col(1) is less than Col(2) so object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).
• X Row(-1)&Col(2) vs Y Row(-2)&Col(1) → X is upper-right of Y as Row(-1) is more than Row(-2) and Col(2) is more than Col(1) so object in Row(-1), Col(2) should be upper-right of object in Row(-2), Col(1).
• X Row(-2)&Col(1) vs Y Row(-1)&Col(2) → X is lower-left of Y as Row(-2) is less than Row(-1) and Col(1) is less than Col(2) so object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).
• X Row(-2)&Col(2) vs Y Row(-1)&Col(1) → X is lower-right of Y as Row(-2) is less than Row(-1) and Col(2) is more than Col(1) so object in Row(-2), Col(2) should be lower-right of object in Row(-1), Col(1).
  
Few-shot (coordinates only → precise diagonal label):

Example UL
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(2)."
Output:
{{
  "justification": "X is in the top row and left column, so the relation of agent X to Y will be that X is upper-left of Y, as their coordinates, Row(-1) is more than Row(-2) and Col(1) is less than Col(2) and object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).",
  "sentence": "X is upper-left of Y"
}}

Example UR
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Output:
{{
  "justification": "Y is top row, right column to X, so relation of agent Y to X will be that Y is upper-right of X, as their coordinates, Row(-1) is more than Row(-2) and Col(2) is more than Col(1) and object in Row(-1), Col(2) should be upper-right of object in Row(-2), Col(1).",
  "sentence": "Y is upper-right of X"
}}

Example LL
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Output:
{{
  "justification": "X is bottom row, left column;  so relation of agent X to Y that X is lower-left of Y, as their coordinates, Row(-2) is less than Row(-1) and Col(1) is less than Col(2) and object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).",
  "sentence": "X is lower-left of Y"
}}

Example LR
Input: "X at Row(-2), Col(2); Y at Row(-1), Col(1)."
Output:
{{
  "justification": "X is bottom row, right column;  so relation of agent X to Y will be that X is lower-right of Y, as their coordinates, Row(-2) is less than Row(-1) and Col(2) is more than Col(1) and object in Row(-2), Col(2) should be lower-right of object in Row(-1), Col(1).",
  "sentence": "X is lower-right of Y"
}}

Now use these rules and examples.

Coordinates:
{coords_line}
Now, think carefully and reason and output the direction as per the coordinates and answer this question: {question} 
return output in the exact json format

""".strip()
    
    return prompt



def build_relation_diagonal_prompt_from_coords(coords_line: str, question: str) -> str:
    prompt = f"""
Return EXACTLY this JSON (no extra keys, no extra text):
{{
  "justification": "<detailed reason based ONLY on the given coordinates>",
  "sentence": "object1 is <one of: upper-left, upper-right, lower-left, lower-right> of object2"
}}

You are given ONLY a coordinates line like:
"<X at Row(_), Col(_); Y at Row(_), Col(_).>"
Each Row(-n) and Col(n) is the index of that object’s row and column.

This case is already known to be DIAGONAL (rows differ AND columns differ). Decide the precise diagonal label for X relative to Y.

Rules (diagonal only):
• X Row(-1)&Col(1) vs Y Row(-2)&Col(2) → X is upper-left of Y as Row(-1) is more than Row(-2) and Col(1) is less than Col(2) so object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).
• X Row(-1)&Col(2) vs Y Row(-2)&Col(1) → X is upper-right of Y as Row(-1) is more than Row(-2) and Col(2) is more than Col(1) so object in Row(-1), Col(2) should be upper-right of object in Row(-2), Col(1).
• X Row(-2)&Col(1) vs Y Row(-1)&Col(2) → X is lower-left of Y as Row(-2) is less than Row(-1) and Col(1) is less than Col(2) so object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).
• X Row(-2)&Col(2) vs Y Row(-1)&Col(1) → X is lower-right of Y as Row(-2) is less than Row(-1) and Col(2) is more than Col(1) so object in Row(-2), Col(2) should be lower-right of object in Row(-1), Col(1).
  
Few-shot (coordinates only → precise diagonal label):

Example UL
Input: "X at Row(-1), Col(1); Y at Row(-2), Col(2)."
Question: What is the relation of the agent X to the agent Y?
Output:
{{
  "justification": "X is in the top row and left column, so the relation of agent X to Y will be that X is upper-left of Y, as their coordinates, Row(-1) is more than Row(-2) and Col(1) is less than Col(2) and object in Row(-1), Col(1) should be upper-left of object in Row(-2), Col(2).",
  "sentence": "X is upper-left of Y"
}}

Example UR
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Question: What is the relation of the agent Y to the agent X?
Output:
{{
  "justification": "Y is top row, right column to X, so relation of agent Y to X will be that Y is upper-right of X, as their coordinates, Row(-1) is more than Row(-2) and Col(2) is more than Col(1) and object in Row(-1), Col(2) should be upper-right of object in Row(-2), Col(1).",
  "sentence": "Y is upper-right of X"
}}

Example LL
Input: "X at Row(-2), Col(1); Y at Row(-1), Col(2)."
Question: What is the relation of the agent X to the agent Y?
Output:
{{
  "justification": "X is bottom row, left column;  so relation of agent X to Y that X is lower-left of Y, as their coordinates, Row(-2) is less than Row(-1) and Col(1) is less than Col(2) and object in Row(-2), Col(1) should be lower-left of object in Row(-1), Col(2).",
  "sentence": "X is lower-left of Y"
}}

Example LR
Input: "X at Row(-2), Col(2); Y at Row(-1), Col(1)."
Question: What is the relation of the agent X to the agent Y?
Output:
{{
  "justification": "X is bottom row, right column;  so relation of agent X to Y will be that X is lower-right of Y, as their coordinates, Row(-2) is less than Row(-1) and Col(2) is more than Col(1) and object in Row(-2), Col(2) should be lower-right of object in Row(-1), Col(1).",
  "sentence": "X is lower-right of Y"
}}

Now use these rules and examples.

Coordinates:
{coords_line}
Now, think carefully and reason and output the direction as per the coordinates and answer this question: {question} 
return output in the exact json format

""".strip()
    
    return prompt

OVERLAP_FINAL_JSON = (
    '{ "justification": "Row and column indices are identical for both; therefore overlap.", '
    '"selected_option": "overlap" }'
)

# ---------- Runner (prints reasoning from all 3 steps) ----------
def build_all_prompts_for_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    p1 = build_coords_prompt_from_pruned_grid(sample)
    p2_template_marker = "{COORDS}"
    p2_template = build_relation_prompt_from_coords(p2_template_marker)
    p_echo = build_grid_prune_echo_prompt(sample)
    return {
        "coords_from_pruned_grid": p1,
        "minimal_pruned_grid_QA_TEMPLATE": p2_template,
        "grid_prune": p_echo,
    }

def run_suite_on_sample(sample: Dict[str, Any], model, timeout_s: int = 890) -> Dict[str, Any]:
    prompts = build_all_prompts_for_sample(sample)

    # Step 1: coords from pruned grid
    coords_resp = timed_invoke(model, prompts["coords_from_pruned_grid"], timeout_s=timeout_s)
    coords_reason, coords_line = parse_reason_and_coords(coords_resp or "")

    print("\n[Step 1: Coords-from-Pruned-Grid]")
    print("Prompt:\n", prompts["coords_from_pruned_grid"])
    print("Model JSON:\n", coords_resp)
    print("Parsed reason:", coords_reason or "(none)")
    print("Parsed coordinates:", coords_line or "(none)")

    # Step 2a: coarse
    p2a = build_relation_coarse_prompt_from_coords(coords_line or "")
    coarse_resp = timed_invoke(model, p2a, timeout_s=timeout_s)
    coarse_sel, coarse_just = parse_choice_and_justification(coarse_resp or "")

    print("\n[Step 2a: Coarse (straight/diagonal/overlap)]")
    print("Prompt:\n", p2a)
    print("Model JSON:\n", coarse_resp)
    print("Parsed coarse selected_option:", coarse_sel or "(none)")
    print("Parsed coarse justification:", coarse_just or "(none)")

    # Step 2b/2c branch
    specialized_prompt = None
    specialized_resp = None
    final_rel_resp = None
    used_branch = None

    if coarse_sel == "overlap":
        final_rel_resp = OVERLAP_FINAL_JSON
        used_branch = "overlap"
        print("\n[Step 2b/2c: Overlap selected → specialized step skipped]")
        print("Final relation JSON:", final_rel_resp)
    elif coarse_sel == "straight":
        specialized_prompt = build_relation_straight_prompt_from_coords(coords_line or "", sample.get("question",""))
        specialized_resp = timed_invoke(model, specialized_prompt, timeout_s=timeout_s)
        final_rel_resp = specialized_resp
        used_branch = "straight"
        spec_sel, spec_just = parse_choice_and_justification(specialized_resp or "")
        print("\n[Step 2b: Straight-specialization]")
        print("Prompt:\n", specialized_prompt)
        print("Model JSON:\n", specialized_resp)
        print("Parsed straight selected_option:", spec_sel or "(none)")
        print("Parsed straight justification:", spec_just or "(none)")
    elif coarse_sel == "diagonal":
        specialized_prompt = build_relation_diagonal_prompt_from_coords(coords_line or "", sample.get("question",""))
        specialized_resp = timed_invoke(model, specialized_prompt, timeout_s=timeout_s)
        final_rel_resp = specialized_resp
        used_branch = "diagonal"
        spec_sel, spec_just = parse_choice_and_justification(specialized_resp or "")
        print("\n[Step 2c: Diagonal-specialization]")
        # print("Prompt:\n", specialized_prompt)
        print("Model JSON:\n", specialized_resp)
        print("Parsed diagonal selected_option:", spec_sel or "(none)")
        print("Parsed diagonal justification:", spec_just or "(none)")
    else:
        # Fallback to original one-shot if coarse failed
        p2_fallback = prompts["minimal_pruned_grid_QA_TEMPLATE"].replace("{COORDS}", coords_line or "")
        specialized_prompt = p2_fallback
        specialized_resp = timed_invoke(model, p2_fallback, timeout_s=timeout_s)
        final_rel_resp = specialized_resp
        used_branch = "fallback_single_step"
        spec_sel, spec_just = parse_choice_and_justification(specialized_resp or "")
        print("\n[Step 2 Fallback: Single-step relation from coords]")
        print("Prompt:\n", p2_fallback)
        print("Model JSON:\n", specialized_resp)
        print("Parsed selected_option:", spec_sel or "(none)")
        print("Parsed justification:", spec_just or "(none)")
        # Optional: flat summary fields for quick reads
    final_sel = ""
    if used_branch == "overlap":
        final_sel = "overlap"
    else:
        _sel_tmp, _ = parse_choice_and_justification(specialized_resp or "")
        final_sel = (_sel_tmp or "").strip().lower()
        # ---- Final selected_option (original vs flipped-for-order) ----
    selected_option_original = ""
    if used_branch == "overlap":
        selected_option_original = "overlap"
    else:
        _sel_tmp, _ = parse_choice_and_justification(specialized_resp or "")
        selected_option_original = (_sel_tmp or "").strip().lower()

    selected_option_for_question = selected_option_original
    judged_on = "original"
    if selected_option_original and need_flip_for_order(sample.get("question",""), coords_line or ""):
        flipped = flip_relation(selected_option_original)
        if flipped != selected_option_original:  # (overlap stays overlap)
            selected_option_for_question = flipped
            judged_on = "flipped_for_order"


    return {
        "coords_prompt": prompts["coords_from_pruned_grid"],
        "coords_raw": coords_resp,
        "coords_reason": coords_reason,
        "coords_line": coords_line,
        "coarse_prompt": p2a,
        "coarse_raw": coarse_resp,
        "coarse_selected": coarse_sel,
        "coarse_justification": coarse_just,
        "specialized_prompt": specialized_prompt,
        "specialized_raw": specialized_resp,
        "branch_used": used_branch,
        "final_selected_option": final_sel,
        "coords_line": coords_line,
        "final_selected_option_original": selected_option_original,
        "final_selected_option_for_question": selected_option_for_question,
        "judged_on": judged_on,



    }

# ============================================================
# ============ PART C: End-to-end helper =====================
# ============================================================

def build_graphs_from_paragraph(paragraph: str, prune_pair: Optional[Tuple[str,str]] = None) -> Dict:
    rel_obj = extract_relations_from_paragraph(paragraph)
    relations = rel_obj["relations"]
    coords = place_relations(relations)
    full_grid = render_full_grid_str(coords)

    if prune_pair is None:
        prune_pair = (relations[0]["head"], relations[0]["tail"]) if relations else ("","")
    pruned_grid = render_pruned_grid_str(coords, prune_pair)
    pruned_grid_norm = normalize_pruned_grid(pruned_grid)

    return {
        "relations": relations,
        "full_grid": full_grid,
        "pruned_grid": pruned_grid,
        "pruned_grid_normalized": pruned_grid_norm,
        "prune_pair": {"head": prune_pair[0], "tail": prune_pair[1]}
    }

def make_stepgame_like_sample(graph_obj: Dict, question_pair: Tuple[str,str]) -> Dict[str, Any]:
    x, y = question_pair
    q = f"What is the relation of the agent {x} to the agent {y}?"
    return {
        "story": "",
        "question": q,
        "candidate_options": ["left","right","above","below","upper-left","upper-right","lower-left","lower-right","overlap"],
        "relations": graph_obj["relations"],
        "label": "",
        "full_grid": graph_obj["full_grid"],
        "pruned_grid": graph_obj["pruned_grid"],
        "k_hop": None,
        "dataset_id": None,
        "index": None,
    }

def solve_relation_from_paragraph_and_pair(paragraph: str, pair: Tuple[str,str], timeout_s: int = 890) -> Dict:
    # Build graphs
    graph = build_graphs_from_paragraph(paragraph, prune_pair=pair)
    sample = make_stepgame_like_sample(graph, (pair[0], pair[1]))
    suite = run_suite_on_sample(sample, lm2, timeout_s=timeout_s)

    # Parse final selection from specialized_raw if present, else from coarse/overlap
    final_selected = None
    if suite["branch_used"] == "overlap":
        try:
            final_selected = json.loads(OVERLAP_FINAL_JSON).get("selected_option")
        except Exception:
            final_selected = "overlap"
    else:
        sel, _ = parse_choice_and_justification(suite.get("specialized_raw") or "")
        if sel:
            final_selected = sel
        else:
            # fallback to coarse selected if nothing else
            final_selected = suite.get("coarse_selected")

    return {
        "relations_from_sentences": graph["relations"],
        "full_grid": graph["full_grid"],
        "pruned_grid": graph["pruned_grid"],
        "pruned_grid_normalized": graph["pruned_grid_normalized"],
        "question": f"What is the relation of the agent {pair[0]} to the agent {pair[1]}?",
        "final_selected_relation": final_selected,
        "debug_suite": suite,
    }
# =========================
# VALIDATION HELPERS (ADD)
# =========================
# Used when building stepgame_like sample
CHOICES = [
    "left","right","above","below",
    "upper-left","upper-right","lower-left","lower-right","overlap"
]

# Fine→coarse mapping used by gold_coarse_from_label()
FINE_TO_COARSE = {
    "left": "straight",
    "right": "straight",
    "above": "straight",
    "below": "straight",
    "upper-left": "diagonal",
    "upper-right": "diagonal",
    "lower-left": "diagonal",
    "lower-right": "diagonal",
    "overlap": "overlap",
}
_REL_WORDS = r'(left|right|above|below|upper-left|upper-right|lower-left|lower-right|overlap)'

def parse_relation_sentence_to_triplet(s: str) -> Optional[Dict[str,str]]:
    if not s:
        return None
    s = s.strip().rstrip(".")
    m = re.match(
        rf'^\s*([A-Za-z0-9_]+)\s+is\s+{_REL_WORDS}\s+of\s+([A-Za-z0-9_]+)\s*$',
        s, re.I
    )
    if not m:
        return None
    head = m.group(1)
    rel  = m.group(2).lower()
    tail = m.group(3)
    return {"head": head, "relation": rel, "tail": tail}

INV_REL = {
    "left":"right","right":"left",
    "above":"below","below":"above",
    "upper-left":"lower-right","lower-right":"upper-left",
    "upper-right":"lower-left","lower-left":"upper-right",
    "overlap":"overlap",
}

def _split_sentences_story(story: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.?!]+", story or "") if s.strip()]


def extract_relations_with_sources(paragraph: str) -> Dict[str, Any]:
    try:
        sents = _split_sentences_story(paragraph)
        by_sentence = []
        relations = []

        for raw in sents:
            parsed = _rewrite_pair_sentence(raw)
            if not parsed:
                by_sentence.append({"sentence": raw, "parsed": False})
                continue

            pipe = _run_stage_pipeline(parsed["ab_sentence"], a_symbol=parsed["head"], b_symbol=parsed["tail"])

            # ---- your normal success path ----
            if pipe and pipe.get("ok") and pipe.get("final_triplet"):
                trip = pipe["final_triplet"]
                relations.append(trip)
                by_sentence.append({
                    "sentence": raw,
                    "parsed": True,
                    "ok": True,
                    "ab_sentence": parsed["ab_sentence"],
                    "prediction": trip,
                    "two_stage": {
                        "stage1": {
                            "coarse": pipe["stage1"]["coarse"],
                            "reason": pipe["stage1"]["reason"],
                            "raw":    pipe["stage1"]["raw"],
                        },
                        "stage2": {
                            "sentence": pipe["stage2"]["sentence"],
                            "raw":      pipe["stage2"]["raw"],
                        }
                    }
                })
                continue

            if not pipe or not pipe.get("ok"):
                # Safe snapshots (pipe might be None)
                _stage1 = (pipe or {}).get("stage1", {}) or {}
                _stage2 = (pipe or {}).get("stage2", {}) or {}
            
                rec = {
                    "sentence": raw,
                    "parsed": True,
                    "ok": False,
                    "ab_sentence": parsed["ab_sentence"],
                    "stage1_raw": _stage1.get("raw"),
                    "stage2_raw": _stage2.get("raw"),
                    "stage2_sentence": _stage2.get("sentence"),
                }

                s2 = rec.get("stage2_sentence") or ""
                if not s2:
                    s2 = _extract_sentence_field_from_text(rec.get("stage2_raw") or "") or ""
                if not s2:
                    s2 = _recover_relation_sentence_from_text(rec.get("stage2_raw") or "") or ""
                rec["stage2_sentence"] = s2 or None
            
                trip = None
                if s2:
                    trip = parse_relation_sentence_to_triplet_with_mapping(
                        s2,
                        a_symbol=parsed["head"],
                        b_symbol=parsed["tail"]
                    )
            
                if trip:
                    relations.append(trip)
                    by_sentence.append({
                        "sentence": raw,
                        "parsed": True,
                        "ok": True,
                        "ab_sentence": parsed["ab_sentence"],
                        "prediction": trip,
                        "two_stage": {
                            "stage1": {
                                "coarse": _stage1.get("coarse"),
                                "reason": _stage1.get("reason"),
                                "raw":    _stage1.get("raw"),
                            },
                            "stage2": {
                                "sentence": s2,                 # recovered sentence
                                "raw":      rec.get("stage2_raw"),
                            }
                        }
                    })
                    continue
            
                # Recovery failed → keep the failure record
                by_sentence.append(rec)
                continue
            

        # Always return a dict
        return {"relations": relations, "by_sentence": by_sentence}

    except Exception as e:
        # Never return None; fail "closed"
        return {"relations": [], "by_sentence": [{"error": str(e)}]}


def _canon_rel(x: str) -> str:
    return (x or "").strip().lower()

def _same_or_inverse(pred, gold) -> bool:
    """Lenient: exact match OR exact inverse (swap head/tail + invert label)."""
    if not pred or not gold:
        return False
    h1, r1, t1 = pred["head"], _canon_rel(pred["relation"]), pred["tail"]
    h2, r2, t2 = gold["head"], _canon_rel(gold["relation"]), gold["tail"]
    # direct
    if h1 == h2 and t1 == t2 and r1 == r2:
        return True
    # inverse
    inv = INV_REL.get(r2)
    if h1 == t2 and t1 == h2 and r1 == inv:
        return True
    return False

def _index_relations(rel_list: List[Dict[str,str]]):
    """
    Build a lookup by unordered pair so we can lenient-match regardless of direction.
    key = frozenset({head,tail})
    value = list of candidate relations between those two (keep all if duplicates)
    """
    box = {}
    for r in rel_list:
        key = frozenset({r["head"], r["tail"]})
        box.setdefault(key, []).append(r)
    return box

def compare_relations_lenient(pred_rels: List[Dict[str,str]],
                              gold_rels: List[Dict[str,str]]) -> Dict[str, Any]:
    """
    For each gold relation, see if there exists a matching pred relation (direct or inverse).
    Also report extras in the prediction that don’t correspond to any gold pair.
    """
    gold_idx = _index_relations(gold_rels)
    pred_idx = _index_relations(pred_rels)

    per_gold = []
    all_match = True
    for key, gold_list in gold_idx.items():
        # For each gold edge among this pair, try to match any predicted edge among same pair.
        preds_for_pair = pred_idx.get(key, [])
        for gold in gold_list:
            match = None
            for pred in preds_for_pair:
                if _same_or_inverse(pred, gold):
                    match = pred
                    break
            ok = match is not None
            all_match &= ok
            per_gold.append({
                "gold": gold,
                "matched": ok,
                "pred": match
            })

    # extras: predicted edges on pairs not present in gold at all
    extras = []
    for key, pred_list in pred_idx.items():
        if key not in gold_idx:
            extras.extend(pred_list)
            all_match = False

    return {
        "all_relations_lenient_match": all_match,
        "per_gold": per_gold,
        "extra_predictions": extras
    }

def infer_final_direction_from_solver(raw_text: str, question: str) -> str:
    """
    Infer the final directional label from solver raw output and question wording
    using local parsing only.
    """
    qx, qy = parse_question_agents(question or "")

    selected, _ = parse_choice_and_justification(raw_text or "")
    if selected:
        return selected.strip().lower()

    sent = _extract_sentence_field_from_text(raw_text or "")
    trip = parse_relation_sentence_to_triplet(sent) if sent else None
    if trip:
        rel = _canon_rel(trip["relation"])
        if trip["head"].upper() == qx and trip["tail"].upper() == qy:
            return rel
        if trip["head"].upper() == qy and trip["tail"].upper() == qx:
            return INV_REL.get(rel, rel)

    m = re.search(r'\b(left|right|above|below|upper-left|upper-right|lower-left|lower-right|overlap)\b', raw_text or "", re.I)
    return m.group(1).lower() if m else ""
def pruned_grid_matches_exact(generated_pruned: str, given_pruned: str) -> Dict[str, Any]:
    """
    Compare exact text and normalized 2x2 versions.
    """
    exact = (generated_pruned or "").strip() == (given_pruned or "").strip()
    norm_gen = normalize_pruned_grid(generated_pruned or "")
    norm_giv = normalize_pruned_grid(given_pruned or "")
    norm_equal = norm_gen == norm_giv
    return {
        "exact_equal": exact,
        "normalized_equal": norm_equal,
        "normalized_generated": norm_gen,
        "normalized_given": norm_giv
    }

def gold_coarse_from_label(label: str) -> str:
    lab = _canon_rel(label)
    return FINE_TO_COARSE.get(lab, "")

def validate_single_sample(sample: Dict[str, Any], timeout_s: int = 890) -> Dict[str, Any]:
    """
    End-to-end checks for one JSON sample (like the object you pasted).
    - re-extract relations from story
    - lenient-compare to sample['relations']
    - rebuild grids from extracted rels and compare pruned grids (normalized + exact)
    - run 3-prompt grid solver on that pruned grid for the sample's question pair
    - compare predicted label to sample['label']
    - check coarse branch used matches gold coarse
    """
    story = sample.get("story","")
    question = sample.get("question","")
    label_gt = _canon_rel(sample.get("label",""))
    given_pruned = sample.get("pruned_grid","")
    given_relations = sample.get("relations", [])

    # 1) re-extract with provenance
    extracted = extract_relations_with_sources(story)
    rels_pred = extracted["relations"]

    # 2) lenient relation comparison
    rel_cmp = compare_relations_lenient(rels_pred, given_relations)

    # 3) rebuild grids from *our* extracted relations and question’s pair
    q_head, q_tail = parse_question_agents(question)
    if not (q_head and q_tail):
        q_head, q_tail = ("", "")
    coords = place_relations(rels_pred)
    full_grid_gen = render_full_grid_str(coords)
    pruned_grid_gen = render_pruned_grid_str(coords, (q_head, q_tail))

    # 4) pruned grid equality (exact + normalized)
    grid_cmp = pruned_grid_matches_exact(pruned_grid_gen, given_pruned)

    # 5) run your EXACT 3 prompts on our generated pruned grid for that question pair
    stepgame_like = {
        "story": "",
        "question": question,
        "candidate_options": CHOICES,
        "relations": rels_pred,
        "label": label_gt,  # we pass gold for judging
        "full_grid": full_grid_gen,
        "pruned_grid": pruned_grid_gen,
        "k_hop": sample.get("k_hop"),
        "dataset_id": sample.get("dataset_id"),
        "index": sample.get("index"),
    }
    suite = run_suite_on_sample(stepgame_like, lm2, timeout_s=timeout_s)

        # 6) parse model’s final selection (already flip-aware from runner)
    branch_used = suite.get("branch_used", "")
    coords_line = suite.get("coords_line", "")

    model_pred = (suite.get("final_selected_option_for_question") or "").strip().lower()
    judged_on = suite.get("judged_on", "original")
    solver_raw = (suite.get("raw_blocks", {})
                  .get("specialized_step", {})
                  .get("raw", ""))
    question = sample.get("question", "")
    if solver_raw and question:
        model_pred = infer_final_direction_from_solver(solver_raw, question)
        pred_vs_gold["predicted"] = model_pred
        pred_vs_gold["correct"] = (model_pred == label_gt)
    # 7) correctness vs gold
    pred_vs_gold = {
        "predicted": model_pred,
        "gold": label_gt,
        "correct": (model_pred == label_gt),
        "judged_on": judged_on,
        "original_model_choice": suite.get("final_selected_option_original", "")
    }

    if not model_pred:
        solver_raw = (suite.get("specialized_raw", ""))
        question = sample.get("question", "")
        if solver_raw and question:
            model_pred = infer_final_direction_from_solver(solver_raw, question)
            pred_vs_gold["predicted"] = model_pred
            pred_vs_gold["correct"] = (model_pred == label_gt)


    # 8) coarse branch verification vs gold
    gold_coarse = gold_coarse_from_label(label_gt)
    branch_map = {
        "overlap": "overlap",
        "straight": "straight",
        "diagonal": "diagonal",
        # fallback_single_step could be anything — treat as "unknown"
    }
    branch_ok = (branch_map.get(branch_used, "unknown") == gold_coarse)

    # 9) Assemble pretty JSON
    out = {
        "meta": {
            "dataset_id": sample.get("dataset_id"),
            "index": sample.get("index"),
            "k_hop": sample.get("k_hop"),
        },
        "story_sentences": [s["sentence"] for s in extracted["by_sentence"] if s.get("sentence")],
        "sentence_level_reasoning": extracted["by_sentence"],

        "relation_extraction": {
            "lenient_all_match": rel_cmp["all_relations_lenient_match"],
            "mismatches": [
                {
                    "gold": g["gold"],
                    "pred": g["pred"],
                    "sentence": next((bs["sentence"] for bs in extracted["by_sentence"]
                                      if bs.get("prediction")==g["pred"]), None)
                }
                for g in rel_cmp["per_gold"] if not g["matched"]
            ],
            "extra_predictions": rel_cmp["extra_predictions"],
            "predicted_relations": rels_pred,
            "gold_relations": given_relations
        },
        "grids": {
            "full_grid_generated": full_grid_gen,
            "pruned_grid_generated": pruned_grid_gen,
            "pruned_grid_given": given_pruned,
            "match": grid_cmp
        },
        "solver": {
            "coords_line": coords_line,
            "branch_used": branch_used,
            "gold_coarse": gold_coarse,
            "branch_matches_gold_coarse": branch_ok,
            "final_prediction_vs_gold": pred_vs_gold,
            "raw_blocks": {
                "coords_step": {
                    "prompt": suite.get("coords_prompt", ""),
                    "raw": suite.get("coords_raw", ""),
                },
                "coarse_step": {
                    "prompt": suite.get("coarse_prompt", ""),
                    "raw": suite.get("coarse_raw", ""),
                },
                "specialized_step": {
                    "prompt": suite.get("specialized_prompt", ""),
                    "raw": suite.get("specialized_raw", ""),
                    "branch_used": branch_used,
                }
            }
        }


        }

    return out

# =========================
# OPTIONAL: JSONL driver (ADD)
# =========================

def _normalize_sample(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept either the original flat input format OR a previous validation-report
    output format and return the flat dict that validate_single_sample expects.
    """
    # Already flat format — has 'story' at top level
    if "story" in raw and "label" in raw:
        return raw

    # Output/report format — remap nested fields
    meta = raw.get("meta", {})
    story_sents = raw.get("story_sentences", [])
    story = ". ".join(story_sents) + "." if story_sents else ""

    return {
        "story":       story,
        "question":    raw.get("question", ""),
        "label":       raw.get("solver", {}).get("final_prediction_vs_gold", {}).get("gold", ""),
        "pruned_grid": raw.get("grids", {}).get("pruned_grid_given", ""),
        "relations":   raw.get("relation_extraction", {}).get("gold_relations", []),
        "k_hop":       meta.get("k_hop"),
        "dataset_id":  meta.get("dataset_id"),
        "index":       meta.get("index"),
    }

def validate_jsonl(input_jsonl: str, output_jsonl: str, limit: Optional[int]=None, timeout_s: int = 890):
    n = 0
    with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_jsonl, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            sample = _normalize_sample(raw)
            report = validate_single_sample(sample, timeout_s=timeout_s)
            fout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            n += 1
            if limit is not None and n >= limit:
                break
    print(f"✅ wrote {n} validation reports → {output_jsonl}")
# =========================
# Demo
# =========================
if __name__ == "__main__":
    validate_jsonl(
        input_jsonl="./validation_1reportsqwen14b (2).jsonl",
        output_jsonl="./validation_reportsqwen14breal.jsonl",
        limit=2,           # or an int
        timeout_s=890
    )
