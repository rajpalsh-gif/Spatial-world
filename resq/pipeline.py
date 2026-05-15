# -*- coding: utf-8 -*-
"""
SpRLQA-YN pipeline (end-to-end, JSON-saved, + entity-select + per-question pruned-grid)

CHANGES YOU ASKED FOR (implemented):
1) The full pipeline now runs through OpenAI Responses with GPT-5.1.
    - The old Ollama wrapper name is kept only for compatibility.
    - Relation extraction, entity selection, grid generation/repair, and QA all use GPT-5.1.

2) Relation extraction keeps the same robust parser.
    - It still accepts either JSON {"triples":[...]} OR plain "head REL tail" lines.

3) Entity extraction / selection (per-question) is kept.
    - Build universe entities from relations (unique head/tail strings, plus any entities in relations JSON).
    - Select the relevant entity or entities for the question.

4) Pruned grid per question:
   - Takes the final (repaired) grid_ascii
   - Keeps ONLY selected entities (and their nested brace paths), removes everything else
   - Renumbers rows/cols to the minimal bounding box, then renders again as a 5x5 grid
     (so your GRID_ONLY prompt format stays identical)
   - Timing is recorded separately for:
     - entity selection
     - prune grid build
     - full-grid QA
     - pruned-grid QA
     - everything else already timed

5) Output JSON now includes:
    - relations extracted by GPT-5.1 (raw + parsed + triples_text)
   - entity selection per question (raw + parsed + selected_entities)
   - full grid + pruned grid per question
   - QA results for 5 modes: text_only, relations_only, text_relations, grid_only_full, grid_only_pruned
   - timings for each step

IMPORTANT SECURITY NOTE:
- Use env var OPENAI_API_KEY (do not hardcode keys).
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================
INPUT_JSON = "./SpRLQA-YN-test.json"
OUTPUT_JSON = "./SpRLQA-YN-test_results_gpt51.json"  # single file, all results appended here

SKIP_CONTEXTS = 0
PROCESS_CONTEXTS: Optional[int] = 5  # None = all

# ---------------- OpenAI client ----------------
from utils.llm_clients import call_gpt as _call_gpt_llm


# ---------------- Models ----------------
# GPT-5.1 for relations + entity selection
PIPELINE_MODEL = "gpt-5.1"

REL_OLLAMA_MODEL_NAME = PIPELINE_MODEL
REL_OLLAMA_TEMPERATURE = 0.0

ENTSEL_OLLAMA_MODEL_NAME = PIPELINE_MODEL
ENTSEL_OLLAMA_TEMPERATURE = 0.0

GPT_GRID_MODEL = PIPELINE_MODEL

# GPT-5.1 for QA too
QWEN_MODEL_NAME = PIPELINE_MODEL
QWEN_TEMPERATURE = 0.0

# Printing
PRINT_CONTEXT = True
PRINT_RELATIONS = True
PRINT_GRID = True
PRINT_PRUNED_GRID = True
PRINT_QA_SUMMARY = True

# If True, also store full raw grid ASCII in index summary (can be huge)
INDEX_INCLUDE_GRID = False

# ============================================================
# YOU PROVIDED THESE FUNCTIONS (kept style; just used differently)
# ============================================================

def call_ollama_llama(prompt: str,
                      model: str = "gpt-5.1",
                      temperature: float = 0.0) -> str:
    """
    Compatibility wrapper for the old Ollama call sites.
    Routes those requests to OpenAI Responses using GPT-5.1.
    """
    return _call_gpt_llm(prompt, model=model)


def call_gpt_5_mini(prompt: str, model: str = "gpt-5.1") -> str:
    """
    Responses API call.
    """
    return _call_gpt_llm(prompt, model=model)


# ============================================================
# UTILITIES
# ============================================================
def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _safe_slug(s: str) -> str:
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^a-zA-Z0-9_\-\.]", "", s)
    return s[:160] if s else "item"

def _clean_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = re.sub(r"\s*,\s*", ", ", s)
    return s.strip()
def _split_top_level_commas_and_cells(text: str) -> List[str]:
    """
    Splits on commas at top-level (not inside nested () or {}), similar to _split_top_level_commas,
    but also trims and drops empties.
    """
    parts = _split_top_level_commas(text)
    return [p.strip() for p in parts if p.strip()]


def _extract_entities_from_paren_expr(expr: str) -> List[str]:
    """
    Given "(head { a , (b { c }) , d })" or "(head)" or "token",
    returns a flat list of all surface entity strings:
      [head, a, b, c, d]
    """
    expr = (expr or "").strip()
    if not expr:
        return []

    # If parenthesized:
    m = re.match(r"^\((.*)\)$", expr)
    if not m:
        # bare token
        return [expr.strip()]

    inner = m.group(1).strip()

    # head { inside }
    m2 = re.match(r"^(.*)\{\s*(.*)\s*\}\s*$", inner)
    if not m2:
        # plain "(head)"
        head = inner.strip()
        return [head] if head else []

    head = m2.group(1).strip()
    inside = m2.group(2).strip()

    out = []
    if head:
        out.append(head)

    items = _split_top_level_commas_and_cells(inside)
    for it in items:
        it = it.strip()
        if not it:
            continue
        # recurse if nested "(...)"
        if it.startswith("(") and it.endswith(")"):
            out.extend(_extract_entities_from_paren_expr(it))
        else:
            out.append(it)

    return out


def extract_grid_entity_list(grid_ascii: str) -> List[str]:
    """
    Extracts the *exact surface strings* that appear in the grid.
    This is the ONLY vocabulary the pruned-grid entity selector should use.
    """
    cells = _parse_grid_cells(grid_ascii)
    if not cells:
        return []

    ents = []
    for (r, c), cell in cells.items():
        s = (cell or "").strip()
        if not s or s == "—":
            continue

        # multiple items in one cell: split by " , " (your grid rule)
        # But allow models that sometimes output "," spacing inconsistently:
        # We'll split top-level commas safely.
        items = _split_top_level_commas_and_cells(s)

        # If the whole cell is not comma-separated (common), items will be [s]
        if len(items) == 1 and items[0] == s:
            items = [s]

        for it in items:
            it = it.strip()
            if not it or it == "—":
                continue

            # If it's a paren-expr, extract nested entities; else keep literal token.
            if it.startswith("(") and it.endswith(")"):
                ents.extend(_extract_entities_from_paren_expr(it))
            else:
                ents.append(it)

    # de-dup, preserve order, and keep exact strings
    seen = set()
    out = []
    for e in ents:
        e = _clean_spaces(e)
        if not e:
            continue
        if e not in seen:
            out.append(e)
            seen.add(e)

    # helpful: longer first for selection (reduces picking "bed" when "two wooden single beds" exists)
    out = sorted(out, key=lambda x: (-len(x), x.lower()))
    return out

def _format_story(story_list: List[str]) -> str:
    return _clean_spaces(" ".join([_clean_spaces(x) for x in (story_list or []) if str(x).strip()]))

def _normalize_yn(ans: str) -> str:
    a = (ans or "").strip().lower()
    if a in ("yes", "y", "true"):
        return "Yes"
    if a in ("no", "n", "false"):
        return "No"
    return (ans or "").strip()

def _normalize_candidate(ans: str, candidates: List[str]) -> str:
    ans_clean = (ans or "").strip()
    if not candidates:
        return ans_clean
    for c in candidates:
        if ans_clean.lower() == str(c).strip().lower():
            return str(c).strip()
    yn = _normalize_yn(ans_clean)
    for c in candidates:
        if yn.lower() == str(c).strip().lower():
            return str(c).strip()
    return ans_clean

def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try balanced-brace extraction (handles trailing text after JSON)
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\':
            if in_str:
                escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                blob = text[start:i+1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    break
    # Greedy fallback
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def _extract_answer_fallback(text: str, candidates: List[str]) -> Optional[str]:
    if not text:
        return None
    m = re.search(r'"answer"\s*:\s*"([^"]+)"', text, flags=re.I)
    if m:
        return _normalize_candidate(m.group(1), candidates)
    m2 = re.search(r'"selected_option"\s*:\s*"([^"]+)"', text, flags=re.I)
    if m2:
        return _normalize_candidate(m2.group(1), candidates)
    for c in candidates or []:
        if re.search(rf"\b{re.escape(str(c))}\b", text, flags=re.I):
            return str(c).strip()
    m3 = re.search(r"\b(Yes|No)\b", text, flags=re.I)
    if m3:
        return _normalize_candidate(m3.group(1), candidates)
    return None

# ---------------- TIMING HELPERS ----------------
def _now() -> float:
    return time.time()

def _secs(t0: float, t1: Optional[float] = None) -> float:
    return float((t1 if t1 is not None else _now()) - t0)

# ============================================================
# MODEL CALL HELPERS (relations + entity selection + QA)
# ============================================================
def call_ollama_cached(prompt: str, model: str, temperature: float) -> str:
    return call_ollama_llama(prompt, model=model, temperature=temperature)

def call_qwen_cached(prompt: str) -> str:
    return call_ollama_cached(prompt, model=QWEN_MODEL_NAME, temperature=QWEN_TEMPERATURE)

# ============================================================
# RELATION EXTRACTION PROMPT (GPT-5.1)
# ============================================================
RELATION_EXTRACTION_PROMPT_OLLAMA = r"""
You extract spatial relations from a short STORY.

Output ONLY valid JSON in this exact schema (no extra keys):
{
  "entities": ["..."],
  "triples": [
    {"head": "<entity>", "relation": "<REL>", "tail": "<entity>"}
  ]
}

REL must be one of:
RCC8: DC, EC, PO, EQ, TPP, NTPP, TPPI, NTPPI
Directional: LEFT, RIGHT, ABOVE, BELOW, FRONT, BEHIND
Distance: NEAR, FAR

Meaning: REL(head, tail) => "head is REL of tail".
Example: "tree LEFT house" means tree is left of house.

STRICT RULES:
- Extract as many true relations as possible, but DO NOT hallucinate entities.
- Keep full surface names from the story (include colors/adjectives if they identify the object).
- If story says "X with Y" or "X has Y", usually use EC(X, Y) (attachment).
- For containment:
  - "in / inside / within" => NTPPI(container, item)
  - "part of / of the building" does NOT automatically mean inside.
- For "a room with ..." treat room as container: NTPPI(room, item) for items mentioned.

Directional normalization:
- "in front of" => FRONT
- "behind / at the back of" => BEHIND

Quick examples (copy the style):

1) Courtyard/building:
STORY: "the view into the courtyard of a yellow building with white doors. a palm trunk on the left."
JSON:
{
  "entities":["courtyard","yellow building","white doors","palm trunk"],
  "triples":[
    {"head":"courtyard","relation":"FRONT","tail":"yellow building"},
    {"head":"yellow building","relation":"EC","tail":"white doors"},
    {"head":"palm trunk","relation":"LEFT","tail":"yellow building"}
  ]
}

2) Room with bed + left/right tables:
STORY: "a room with grey walls. a bed and a table on both sides."
JSON:
{
  "entities":["room with grey walls","bed","bed table on left","bed table on right"],
  "triples":[
    {"head":"room with grey walls","relation":"NTPPI","tail":"bed"},
    {"head":"room with grey walls","relation":"NTPPI","tail":"bed table on left"},
    {"head":"room with grey walls","relation":"NTPPI","tail":"bed table on right"},
    {"head":"bed table on left","relation":"LEFT","tail":"bed"},
    {"head":"bed table on right","relation":"RIGHT","tail":"bed"}
  ]
}

3) Window with curtains (attachment):
STORY: "a window with green curtains on the right."
JSON:
{
  "entities":["window","green curtains"],
  "triples":[
    {"head":"window","relation":"EC","tail":"green curtains"}
  ]
}

4) RCC8 containment:
STORY: "children in a swimming pool in the garden."
JSON:
{
  "entities":["garden","swimming pool","children"],
  "triples":[
    {"head":"garden","relation":"NTPPI","tail":"swimming pool"},
    {"head":"swimming pool","relation":"NTPPI","tail":"children"}
  ]
}

Now do the STORY:
{story_text}
"""

# ============================================================
# PROMPT 2: GRID (GPT-5 / GPT-5.1)
# ============================================================
GRID_GENERATION_PROMPT = r"""
You create a ROW/COL ASCII grid for spatial reasoning.

Return ONLY valid JSON with this exact schema:
{
  "grid_size": {"rows": <int>, "cols": <int>},
  "grid_ascii": "<ASCII grid>",
  "placement": [
    {"row": <int>, "col": <int>, "cell_text": "<exact cell contents>"}
  ]
}

=====================
GRID STYLE (EXACT)
=====================
- Use "—" for empty cells (dash).
- Each cell text is plain text; if multiple items share a cell, separate by " , ".
- Use parentheses + curly braces for attachments / associated-with:
    (X { a , b , c })

Projection rule (for visualization only):
- Treat these relations as "put inside braces" (same cell with braces):
  EC, NTPPI, TPPI, TPP, NTPP, EQ  -> render as (Container { item })
- Directional relations are encoded by row/col positions.

=====================
NUMERIC COORDINATE RULES (CRITICAL)
=====================
- Columns increase to the RIGHT:
  Col1 < Col2 < Col3 < Col4 < Col5
  So Col1 is LEFT of Col3, and Col3 is RIGHT of Col1.

- Rows increase DOWNWARD:
  Row1 < Row2 < Row3 < Row4 < Row5
  So Row2 is ABOVE Row4, and Row4 is BELOW Row2.

=====================
DIRECTION RULES (CRITICAL)
=====================
Interpret a triple as REL(A, B) meaning "A is REL of B".

LEFT(A,B):  col(A) < col(B)
RIGHT(A,B): col(A) > col(B)
ABOVE(A,B): row(A) < row(B)
BELOW(A,B): row(A) > row(B)

FRONT(A,B): A is IN FRONT OF B -> place A to the RIGHT of B
  Practical: col(A) = col(B) + 1 (or +2 if needed for spacing)
  ALSO: append the tag  #front(<B>)  to A's cell text.

BEHIND(A,B): A is BEHIND B -> place A to the LEFT of B
  Practical: col(A) = col(B) - 1 (or -2 if needed)
  ALSO: append the tag  #behind(<B>)  to A's cell text.

IMPORTANT – FRONT/BEHIND vs LEFT/RIGHT DISAMBIGUATION:
- FRONT/BEHIND are encoded using columns (same axis as LEFT/RIGHT)
  BUT they carry an EXPLICIT TAG so the QA stage can distinguish them.
- Column position alone only proves LEFT/RIGHT.
- A cell is FRONT/BEHIND of another ONLY when the #front(...) / #behind(...) tag is present.
- If a triple says FRONT or BEHIND, you MUST add the tag. Never omit it.

=====================
FOREGROUND / BACKGROUND RULE (CRITICAL)
=====================
Image descriptions often use "in the foreground" / "in the background".
These describe VIEWER DEPTH — NOT left/right.

- "in the foreground" = closer to the viewer = BOTTOM of the scene = Row4–Row5.
- "in the background" = farther from viewer = TOP of the scene = Row1–Row2.

If the story says "X in the foreground" and "Y in the background":
  -> place X at Row4–5, Y at Row1–2.
  -> Also add  #front(<Y>)  tag on X  and  #behind(<X>)  tag on Y
     so FRONT/BEHIND is recorded explicitly.

=====================
FEW-SHOT MINI EXAMPLES (HOW TO PLACE)
=====================
Example 1 (FRONT/BEHIND with tags + columns):
If:
  car BEHIND house
Then valid placement:
  house at Row3 Col3, car at Row3 Col1  #behind(house)
Because col(car) < col(house) AND the #behind tag is present.

Example 2 (Rows):
If:
  lamp ABOVE table
Then valid placement:
  lamp at Row2, table at Row4
Because row(lamp) < row(table) => lamp is ABOVE table.

Example 3 (Foreground / Background with tags):
If story says:
  "bed in the foreground, sofa in the background"
Then valid placement:
  Row1 Col3: sofa  #behind(bed)
  Row5 Col3: bed   #front(sofa)
Because foreground = bottom rows, background = top rows,
and FRONT/BEHIND tags are added explicitly.

Example 4 (Scene-level container — room):
If story says:
  "Interior view of a room with a bed, a desk below a TV, and a fridge"
Do NOT render:  (room { bed , desk , TV , fridge })
Instead, place each object individually:
  Row1 Col3: TV
  Row2 Col3: desk
  Row5 Col2: bed
  Row5 Col4: fridge
The room is the implicit scene — not a brace container.

=====================
MULTI-STOREY RULE (CRITICAL UPDATE)
=====================
If story says "three-story" / "3-storey" building:
- Interpret as: ground floor + 3 floors above it => span 4 rows total.
General rule: N-storey => span (N + 1) rows.

Placement:
- Put the building in Col2 spanning consecutive rows.
- The BOTTOM row of the span is the GROUND FLOOR.
- The rows ABOVE the ground are floors 1..N.
- Each floor row gets a UNIQUE label: "floor3", "floor2", "floor1", "ground".
- Do NOT repeat the full building expression "(building { roof, windows })" on every row.
  Instead, mention the building name ONCE (e.g., on the ground row) and use floor labels
  on the other rows. Place per-floor features (balcony, rail) on the appropriate floor row.

Example:
three-story building spans 4 rows:
Row2: floor3 (balcony , rail)
Row3: floor2 (balcony , rail)
Row4: floor1 (balcony , rail)
Row5: (green building { flat roof , small windows })

WRONG (duplicates building on every row):
Row2: (green building { flat roof , small windows })
Row3: (green building { flat roof , small windows })
Row4: (green building { flat roof , small windows })
Row5: (green building { flat roof , small windows })

=====================
EACH LEVEL HAS BALCONY RULE (CRITICAL UPDATE)
=====================
If story says "each level has a balcony with rail":
- Apply to FLOORS ONLY (not ground floor).
- For each floor row (rows above ground), render:
  (building { large balcony , arched rail })
- Ground row should NOT include balcony/rail unless story explicitly says ground has it.

=====================
COURTYARD RULE (CRITICAL)
=====================
"courtyard of a building" is NOT inside braces.
Place courtyard OUTSIDE the building, usually IN FRONT (to the right), and aligned with ground row.

Example:
Row5 Col2: (building ...)
Row5 Col3 or Col4: courtyard

=====================
SCENE-LEVEL CONTAINER RULE (CRITICAL)
=====================
Some stories describe a large scene-level container such as:
  "a room with …", "interior of …", "view of a hall with …"

Do NOT put all objects inside one giant brace cell like  (room { bed , desk , TV , … }).
A room/building/hall that contains MOST or ALL objects is a SCENE, not a small container.

Instead:
- OMIT the scene-level container from the grid entirely (it is implicit).
- Place each object at its OWN row/col position using directional and depth cues
  (above/below, left/right, foreground/background).
- ONLY use brace containment  (X { Y })  for SMALL, TIGHT containers where one
  object truly encloses/touches another:
    - a table with items ON it
    - a shelf holding objects
    - a door with a frame
    - a window with curtains
    - a wall with pictures hanging on it

How to decide:
  If the container holds ≤ 3 tightly-associated items -> use braces.
  If the container is the whole room / scene / building covering many objects -> skip braces,
  place objects individually.

=====================
DEFAULT SIZE
=====================
- Use 5 rows.
- Use 5 columns if FRONT/BEHIND appears (recommended).

=====================
NO DUPLICATE ENTITIES (CRITICAL)
=====================
Each distinct entity must appear in EXACTLY ONE cell.
- Do NOT place the same entity in multiple rows or multiple cells.
- For multi-storey buildings: the building name appears ONCE (in one cell spanning
  description), and each floor is described by a LABEL like "floor1", "floor2", etc.
  Do NOT repeat "(building { roof, windows })" on every row.
- If an entity is associated with a container via braces, it appears ONLY inside
  that container's cell — never also as a standalone in another cell.
- WRONG example:  Row1: large window | Row2: (large window { curtains })
  CORRECT example: Row2: (large window { curtains })
  (the standalone duplicate on Row1 must be removed)
- Check your output: scan all 25 cells and verify no entity string appears more than once.

=====================
ASCII FORMAT (MUST MATCH EXACTLY)
=====================

Your grid_ascii MUST be exactly this shape:

<HEADER LINE>
Row1: <cell1> | <cell2> | <cell3> | <cell4> | <cell5>
Row2: <cell1> | <cell2> | <cell3> | <cell4> | <cell5>
Row3: <cell1> | <cell2> | <cell3> | <cell4> | <cell5>
Row4: <cell1> | <cell2> | <cell3> | <cell4> | <cell5>
Row5: <cell1> | <cell2> | <cell3> | <cell4> | <cell5>

Where the HEADER LINE is:
Col1 | Col2 | Col3 | Col4 | Col5
--------------------------------

Rules:
- Always include the header line and the dashed separator line.
- Always include exactly 5 rows labeled Row1..Row5.
- Every row must contain exactly 5 cells separated by " | ".
- Use "—" for empty cells.
=====================
INPUTS
=====================
STORY:
{story_text}

RELATIONS (verbatim triples):
{triples_text}

Now output the JSON with grid_size, grid_ascii, placement.
"""

DIR_RELS = {"LEFT", "RIGHT", "ABOVE", "BELOW", "FRONT", "BEHIND"}

# ============================================================
# GRID REPAIR PROMPT (one pass only)
# ============================================================
GRID_REPAIR_PROMPT_V2 = r"""
You generated a grid but it violates the rules. Fix it.

Return ONLY valid JSON:
{
  "grid_size": {"rows": 5, "cols": 5},
  "grid_ascii": "<fixed grid>",
  "placement": [{"row":<int>,"col":<int>,"cell_text":"<text>"}]
}

grid should be of the format
      Col(1) Col(2) --- Col(n)
Row(1) |
Row(2)|
Row(n)|
RULES TO ENFORCE (DO NOT IGNORE):
1) If story has N-storey building, it spans N+1 rows (ground + N floors).
2) Ground row is the bottom row of the span.
3) If story says "each level has balcony with rail", apply balcony+rail ONLY to floors (rows above ground), not ground.
4) Courtyard is outside building braces, aligned with ground row, and in FRONT => placed to the right column + #front tag.
5) FRONT/BEHIND use columns AND explicit tags: FRONT = right + #front(Y), BEHIND = left + #behind(Y).
   Column position alone only proves LEFT/RIGHT. FRONT/BEHIND requires the tag.
6) Use "—" for empty.
7) SCENE-LEVEL CONTAINERS: If the story describes a room / hall / building that contains MOST objects,
   do NOT put everything inside one brace cell. Place each object at its own row/col.
   Only use braces for small, tight containers (table with items, wall with pictures, door with frame).
8) ORDER-ENCODES-DEPTH (REPAIR-ONLY): Inside any braces list {a , b , c}, earlier items are BEHIND later items; reorder list items only when the question/relations involve FRONT/BEHIND (or "in front of / behind / back").
9) FOREGROUND / BACKGROUND (MANDATORY):
   - "in the foreground" = Row4–5 (bottom). "in the background" = Row1–2 (top).
   - If story says "X in the foreground" and X is at Row1–2, move X to Row4–5.
   - If story says "Y in the background" and Y is at Row4–5, move Y to Row1–2.
   - Add #front / #behind tags between foreground and background groups.
10) RECONCILE STORY ⇄ TRIPLES ⇄ GRID (MANDATORY):
   - Read the entire STORY end-to-end. Identify any "global placement" clauses such as:
     "in the background", "in the foreground", "behind X", "in front of X", "on the left/right",
     "in the corner", "in the shade behind", and clauses like "everything mentioned earlier is behind/ahead of X".
   - Verify that the GRID reflects these clauses. If not, fix the GRID.
11) NO DUPLICATES ACROSS ROW-LINES/CELLS (CRITICAL):
   - Each distinct entity must appear EXACTLY ONCE in the entire grid.
   - Scan all 25 cells: if any entity string appears in more than one cell, REMOVE duplicates.
   - If an entity appears both (a) as a standalone outside a container row and (b) inside a container's
     braces, remove the standalone duplicate and keep the braces version.
   - For multi-storey buildings: the building name + its attributes (roof, windows, etc.)
     should appear ONCE on the ground row. Upper floor rows use labels like "floor1", "floor2"
     with per-floor features (balcony, rail). Do NOT repeat the full building expression
     on every row — that is the most common source of duplicates.
   - Common duplicate patterns to fix:
     * "(building { X, Y })" repeated on Row1, Row2, Row3 → keep on ONE row only
     * "single bed" in Row3 AND Row5 → keep only one occurrence
     * "large window" standalone AND "(large window { curtains })" → keep only the braces version
12) NESTED CONTAINMENT MUST BE VISUALLY NESTED (CRITICAL):
   - If triples imply a chain like:
     children NTPPI swimming pool
     swimming pool NTPPI garden
     then the grid must show nested structure (not flattened):
     (garden { (swimming pool { children }) , ... })
   - Preserve transitive containment in the visual nesting.
13) MINIMAL-EDIT PRINCIPLE:
   - Fix only what is necessary to satisfy rules (1–12).
   - Do NOT introduce new entities. Do NOT drop entities unless required by rule 11 (duplicate removal).

STORY:
{story_text}

RELATIONS:
{triples_text}

BAD GRID (fix this):
{grid_ascii}

"""

# ============================================================
# ENTITY SELECTION PROMPT (GPT-5.1)
# ============================================================
ENTITY_SELECTION_PROMPT_OLLAMA = r"""
You select which entities from a provided LIST are referenced by the QUESTION.

Return ONLY valid JSON:
{
  "selected_entities": ["<exact entity string from the list>", ...]
}

RULES:
- Use ONLY strings that appear EXACTLY in the entity list (copy-paste match).
- Select all entities that the question refers to (often 1 or 2, sometimes more).
- If question says "white thing" / "grey thing" / "green thing":
  choose ALL entities from the list that contain that color word like white carpet, white door etc. This is also true for any other such adjectives.
- If question says "tree", choose the entity that is a tree etc.
- If question says "bedcovers" choose the whole name of bedcovers, like purple bedcover accordingly.
- If question references "room" or "wall" and the list has "room with white walls" or similar,
  select that entity. Match partial names to their full entity string in the list.
- If question references something NOT in the list at all (e.g., "man", "people" but list
  has only building parts), return ALL entities from the list (do NOT return []).
- NEVER return an empty list []. Always select at least one entity.
- If you are uncertain between two close matches, include BOTH.

Examples:
{
  "selected_entities": ["<exact entity string from the list>", ...]
["dark yellow building","white doors","grey gates","trunk of a palm tree","courtyard"]
Q: "Is the tree left of the white thing?"
RULES:
{"selected_entities":["trunk of a palm tree","white doors"]}
- Select all entities that the question refers to (often 1 or 2, sometimes more).
Entity list:
["room with green walls","two wooden single beds","night table","lamp","painting","window","green and orange curtain"]
Q: "Is the lamp below the painting?"
Answer:
{"selected_entities":["lamp","painting"]}

Entity list:
["two wooden single beds","white bedcovers","room","white walls","window","brownish green curtains"]
Q: "Is the bed contain the bedcovers?"
Answer:
{"selected_entities":["two wooden single beds","white bedcovers"]}

Now do it.
Be mindful of getting this right.
ENTITY LIST:
{entity_list}

QUESTION:
{question}
"""

# ============================================================
# QA PROMPTS (GPT-5.1) — MODES
# ============================================================
def build_qa_schema(candidates: List[str]) -> str:
    cand = candidates if candidates else ["Yes", "No"]
    return (
        "Return ONLY valid JSON with keys:\n"
        '{\n'
        '  "justification": "concise reasoning grounded ONLY in the provided input",\n'
        f'  "answer": "one of: {cand}"\n'
        '}\n'
        "Do not output anything else.\n"
    )

PROMPT_TEXT_ONLY = r"""
You are answering a multiple-choice question using ONLY the STORY text.

STORY:
{story_text}

QUESTION:
{question}

{schema}
"""

PROMPT_RELATIONS_ONLY = r"""
You are answering a multiple-choice question using ONLY the extracted RELATIONS (triples).
Treat each triple as REL(head, tail) with directional meaning for LEFT/RIGHT/ABOVE/BELOW/FRONT/BEHIND.

RELATIONS:
{triples_text}

QUESTION:
{question}

{schema}
"""

PROMPT_TEXT_RELATIONS = r"""
You are answering a multiple-choice question using STORY + RELATIONS.
Use story for commonsense when relations are missing or ambiguous.
Treat each triple as REL(head, tail) with directional meaning for LEFT/RIGHT/ABOVE/BELOW/FRONT/BEHIND.
STORY:
{story_text}

RELATIONS:
{triples_text}

QUESTION:
{question}

{schema}
"""

PROMPT_GRID_ONLY = r"""
You are answering a multiple-choice question using ONLY the GRID below,
but you may apply LIMITED real-world commonsense for attachment relations.

Return ONLY valid JSON with keys:
{
  "justification": "think carefully and give your reasoning grounded in grid + minimal commonsense when needed",
  "answer": "<one of the candidate answers>"
}

=====================
HOW TO READ THIS GRID (CRITICAL)
=====================
REL(A,B) is checked by rows/cols AND explicit tags:
- A header line: "Col1 | Col2 | Col3 | Col4 | Col5"
- Rows labeled exactly: "Row1:" ... "Row5:"
(1) Numeric ordering:
- Columns increase to the RIGHT: Col1 < Col2 < Col3 < Col4 < Col5
- Rows increase DOWNWARD: Row1 < Row2 < Row3 < Row4 < Row5

So:
- If an object is in Col1 and another in Col3, then Col1-object is LEFT of Col3-object.
- If an object is in Row2 and another in Row4, then Row2-object is ABOVE Row4-object.

(2) Direction rules:
REL(A,B) is checked by rows/cols AND explicit tags:
- LEFT(A,B)  -> col(A) < col(B)
- RIGHT(A,B) -> col(A) > col(B)
- ABOVE(A,B) -> row(A) < row(B)
- BELOW(A,B) -> row(A) > row(B)
- FRONT(A,B) -> A's cell contains  #front(<B>)  tag. Column alone is NOT enough.
- BEHIND(A,B)-> A's cell contains  #behind(<B>)  tag. Column alone is NOT enough.

CRITICAL: LEFT/RIGHT are determined by column comparison.
FRONT/BEHIND are ONLY true when an explicit #front(...) or #behind(...) tag is present.
Do NOT confuse LEFT with BEHIND or RIGHT with FRONT.

(3) Braces / attachments:
(X { Y , Z }) means Y and Z are attached/associated with X.
NESTING / CONTAINMENT RULE:
- (A { (B { C }) }) => C inside B, B inside A.

ORDER-ENCODES-DEPTH (ONLY for FRONT/BEHIND questions):
Inside { A , B , C } interpret A behind B behind C.

MINIMAL COMMONSENSE OVERRIDE (only for micro-structure):
- rail above balcony
- curtains attached to window (not inside)
- bedcover on bed (not inside)

GRID:
{grid_ascii}

QUESTION:
{question}

{schema}
"""

# ============================================================
# RELATIONS: OLLAMA -> parsed triples
# ============================================================
def _canon_rel(r: str) -> str:
    r = (r or "").strip().upper()
    # map synonyms lightly
    syn = {
        "INFRONT": "FRONT",
        "IN_FRONT": "FRONT",
        "IN FRONT": "FRONT",
        "BACK": "BEHIND",
        "BEHIND": "BEHIND",
        "LEFT": "LEFT",
        "RIGHT": "RIGHT",
        "ABOVE": "ABOVE",
        "BELOW": "BELOW",
        "NEAR": "NEAR",
        "FAR": "FAR",
    }
    return syn.get(r, r)

def _parse_triples_from_text_fallback(raw: str) -> List[Dict[str, Any]]:
    """
    Fallback parser for non-JSON outputs:
    accepts lines like:
      head REL tail
    """
    triples: List[Dict[str, Any]] = []
    if not raw:
        return triples
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in lines:
        # remove bullets
        ln2 = re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip()
        # try match head REL tail (REL token is one of allowed)
        m = re.match(
            r"^(.*)\s+(DC|EC|PO|EQ|TPP|NTPP|TPPI|NTPPI|LEFT|RIGHT|ABOVE|BELOW|FRONT|BEHIND|NEAR|FAR)\s+(.*)$",
            ln2,
            flags=re.I
        )
        if not m:
            continue
        h = _clean_spaces(m.group(1))
        r = _canon_rel(m.group(2))
        t = _clean_spaces(m.group(3))
        if h and r and t:
            triples.append({"head": h, "relation": r, "tail": t})
    return triples

def _triples_to_text(triples: List[Dict[str, Any]]) -> str:
    lines = []
    for t in triples or []:
        h = _clean_spaces(str(t.get("head", "")))
        r = _clean_spaces(str(t.get("relation", ""))).upper()
        ta = _clean_spaces(str(t.get("tail", "")))
        if h and r and ta:
            lines.append(f"{h} {r} {ta}")
    return "\n".join(lines)
def ollama_extract_relations(story_text: str) -> Tuple[str, Dict[str, Any]]:
    prompt = RELATION_EXTRACTION_PROMPT_OLLAMA.replace("{story_text}", story_text)
    raw = call_ollama_cached(prompt, model=REL_OLLAMA_MODEL_NAME, temperature=REL_OLLAMA_TEMPERATURE)
    obj = _extract_first_json_obj(raw)
    triples: List[Dict[str, Any]] = []
    entities: List[str] = []

    if obj and isinstance(obj, dict) and "triples" in obj:
        for t in (obj.get("triples") or []):
            if not isinstance(t, dict):
                continue
            h = _clean_spaces(str(t.get("head", "")))
            r = _canon_rel(str(t.get("relation", "")))
            ta = _clean_spaces(str(t.get("tail", "")))
            if h and r and ta:
                triples.append({"head": h, "relation": r, "tail": ta})
        entities = [str(x).strip() for x in (obj.get("entities") or []) if str(x).strip()]
    else:
        # fallback: parse line triples
        triples = _parse_triples_from_text_fallback(raw)
        entities = []

    # if entities missing, derive from triples
    if not entities:
        s = set()
        for t in triples:
            if t.get("head"):
                s.add(t["head"])
            if t.get("tail"):
                s.add(t["tail"])
        entities = sorted(s, key=lambda x: (len(x), x.lower()))

    return raw, {"entities": entities, "triples": triples}

def ollama_extract_relations_timed(story_text: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    t0 = _now()
    raw, obj = ollama_extract_relations(story_text)
    t1 = _now()
    return raw, obj, {"total_sec": _secs(t0, t1)}

# ============================================================
# GRID: GPT generate + validate/repair (one pass)
# ============================================================
def _grid_format_ok(grid_ascii: str) -> bool:
    if not grid_ascii:
        return False
    lines = [ln.rstrip("\n") for ln in grid_ascii.splitlines() if ln.strip() != ""]
    if len(lines) < 7:
        return False
    if lines[0].strip() != "Col1 | Col2 | Col3 | Col4 | Col5":
        return False
    if "-" not in lines[1]:
        return False
    row_lines = [ln for ln in lines if ln.strip().startswith("Row")]
    if len(row_lines) != 5:
        return False
    for i, ln in enumerate(row_lines, start=1):
        if not ln.strip().startswith(f"Row{i}:"):
            return False
        parts = ln.split(": ", 1)
        if len(parts) != 2:
            return False
        cells = parts[1].split(" | ")
        if len(cells) != 5:
            return False
    return True

def _grid_has_relation_tokens_inside_cells(grid_ascii: str) -> bool:
    if not grid_ascii:
        return False
    for ln in grid_ascii.splitlines():
        if not ln.strip().lower().startswith("row"):
            continue
        for rel in DIR_RELS:
            if re.search(rf"\b{rel}\b", ln):
                return True
    return False

def _parse_grid_cells(grid_ascii: str) -> Dict[Tuple[int, int], str]:
    """
    Parses the 5x5 cells from grid_ascii.
    Returns {(row,col): cell_text}
    """
    cells = {}
    if not grid_ascii:
        return cells
    for ln in grid_ascii.splitlines():
        m = re.match(r"Row(\d+)\s*:\s*(.*)$", ln.strip(), flags=re.I)
        if not m:
            continue
        r = int(m.group(1))
        parts = m.group(2).split(" | ")
        if len(parts) != 5:
            continue
        for c in range(1, 6):
            cells[(r, c)] = parts[c - 1].strip()
    return cells

def _find_entity_in_cell(cell_text_lower: str, entity_lower: str) -> bool:
    # safer-ish boundary matching to avoid bed vs bedcover substring collisions
    if not entity_lower:
        return False
    pat = rf"(?<![a-z0-9]){re.escape(entity_lower)}(?![a-z0-9])"
    return re.search(pat, cell_text_lower) is not None

def _direction_violations(grid_ascii: str, triples: List[Dict[str, Any]]) -> List[str]:
    viols = []
    cells = _parse_grid_cells(grid_ascii)
    if not cells:
        return ["grid_parse_failed"]

    # locate entity: first cell where entity appears
    def locate(ent: str) -> Optional[Tuple[int, int]]:
        e = (ent or "").strip().lower()
        if not e:
            return None
        for r in range(1, 6):
            for c in range(1, 6):
                txt = (cells.get((r, c), "") or "").lower()
                if _find_entity_in_cell(txt, e):
                    return (r, c)
        return None

    for t in triples or []:
        h = str(t.get("head", "")).strip()
        r = str(t.get("relation", "")).strip().upper()
        ta = str(t.get("tail", "")).strip()
        if r not in DIR_RELS:
            continue
        ph = locate(h)
        pt = locate(ta)
        if not ph or not pt:
            continue
        rh, ch = ph
        rt, ct = pt
        ok = True
        if r == "LEFT":
            ok = ch < ct
        elif r == "RIGHT":
            ok = ch > ct
        elif r == "ABOVE":
            ok = rh < rt
        elif r == "BELOW":
            ok = rh > rt
        elif r == "FRONT":
            ok = ch > ct
        elif r == "BEHIND":
            ok = ch < ct
        if not ok:
            viols.append(f"{h} {r} {ta} violated head=({rh},{ch}) tail=({rt},{ct})")
    return viols

def grid_needs_repair_v2(story_text: str, grid_ascii: str, triples: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    reasons = []
    if not _grid_format_ok(grid_ascii):
        reasons.append("bad_grid_format")
    if _grid_has_relation_tokens_inside_cells(grid_ascii):
        reasons.append("relation_tokens_inside_cells")
    viols = _direction_violations(grid_ascii, triples)
    if viols and viols != ["grid_parse_failed"]:
        reasons.append("directional_violations: " + "; ".join(viols[:5]))
    # Check for duplicate entities across cells
    dup_reasons = _detect_duplicate_entities(grid_ascii)
    if dup_reasons:
        reasons.append("duplicate_entities: " + "; ".join(dup_reasons[:5]))
    return (len(reasons) > 0), reasons


def _detect_duplicate_entities(grid_ascii: str) -> List[str]:
    """Detect entities that appear in more than one cell."""
    cells = _parse_grid_cells(grid_ascii)
    if not cells:
        return []

    # Extract entities per cell
    from collections import Counter
    entity_cells: Dict[str, List[str]] = {}  # entity -> list of cell locations
    for (r, c), cell_text in cells.items():
        s = (cell_text or "").strip()
        if not s or s == "\u2014":
            continue
        items = _split_top_level_commas_and_cells(s)
        for it in items:
            it = it.strip()
            if not it or it == "\u2014":
                continue
            if it.startswith("(") and it.endswith(")"):
                ents = _extract_entities_from_paren_expr(it)
            else:
                ents = [it]
            for e in ents:
                e_clean = _clean_spaces(e).lower()
                if not e_clean:
                    continue
                if e_clean not in entity_cells:
                    entity_cells[e_clean] = []
                entity_cells[e_clean].append(f"R{r}C{c}")

    dups = []
    for ent, locs in entity_cells.items():
        if len(locs) > 1:
            dups.append(f"'{ent}' in {','.join(locs)}")
    return dups

def gpt_generate_grid(story_text: str, triples: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    triples_text = _triples_to_text(triples)

    # NOTE: prompts are unchanged byte-for-byte; we only append a tiny extra hint at call time.
    # This fixes the “light brown house” kind of cases + roof tiles intuition for GRID_ONLY questions.
    extra_hint = (
        "\n\nADDITIONAL_HINTS (follow these, do not add new entities):\n"
        "- If you render (house { light brown , red rooftiles }), that means a light brown house with red rooftiles.\n"
        "- When a container like house/building is kept, keep its descriptive items inside braces too (colors/materials/roof tiles).\n"
    )

    prompt = GRID_GENERATION_PROMPT.replace("{story_text}", story_text).replace("{triples_text}", triples_text) + extra_hint

    raw = call_gpt_5_mini(prompt, model=GPT_GRID_MODEL)
    obj = _extract_first_json_obj(raw)
    if not obj or "grid_ascii" not in obj:
        raise ValueError(f"Grid JSON parse failed.\nRAW:\n{raw[:2000]}")

    grid_ascii = str(obj.get("grid_ascii", "")).strip()
    needs, reasons = grid_needs_repair_v2(story_text, grid_ascii, triples)
    if not needs:
        return raw, obj

    repair = GRID_REPAIR_PROMPT_V2 \
        .replace("{story_text}", story_text) \
        .replace("{triples_text}", triples_text) \
        .replace("{grid_ascii}", grid_ascii)

    raw2 = call_gpt_5_mini(repair, model=GPT_GRID_MODEL)
    obj2 = _extract_first_json_obj(raw2)
    if obj2 and "grid_ascii" in obj2:
        obj2["_repair_reasons"] = reasons
        return raw + "\n\n#REPAIR_PASS\n" + raw2, obj2

    obj["_repair_reasons"] = reasons + ["repair_parse_failed"]
    return raw + "\n\n#REPAIR_PARSE_FAILED\n" + (raw2 or ""), obj

def gpt_generate_grid_timed(story_text: str, triples: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    t0 = _now()
    raw, obj = gpt_generate_grid(story_text, triples)
    t1 = _now()
    timing = {"total_sec": _secs(t0, t1), "repair_happened": ("#REPAIR_PASS" in (raw or ""))}
    return raw, obj, timing

# ============================================================
# ENTITY UNIVERSE + PER-QUESTION ENTITY SELECTION (GPT-5.1)
# ============================================================
def build_entity_universe(rel_obj: Dict[str, Any]) -> List[str]:
    ents = [str(x).strip() for x in (rel_obj.get("entities") or []) if str(x).strip()]
    # also include head/tail from triples
    s = set(ents)
    for t in (rel_obj.get("triples") or []):
        h = str(t.get("head", "")).strip()
        ta = str(t.get("tail", "")).strip()
        if h:
            s.add(h)
        if ta:
            s.add(ta)
    # stable-ish ordering: longer first often helps “white thing” disambiguation
    out = sorted(s, key=lambda x: (-len(x), x.lower()))
    return out

def ollama_select_entities(entity_list: List[str], question: str) -> Tuple[str, Dict[str, Any]]:
    ent_list_str = json.dumps(entity_list, ensure_ascii=False)
    prompt = ENTITY_SELECTION_PROMPT_OLLAMA \
        .replace("{entity_list}", ent_list_str) \
        .replace("{question}", question)

    raw = call_ollama_cached(prompt, model=ENTSEL_OLLAMA_MODEL_NAME, temperature=ENTSEL_OLLAMA_TEMPERATURE)
    obj = _extract_first_json_obj(raw)

    selected: List[str] = []
    if obj and isinstance(obj, dict) and "selected_entities" in obj:
        sel = obj.get("selected_entities") or []
        if isinstance(sel, list):
            for x in sel:
                xs = str(x).strip()
                if xs in entity_list:
                    selected.append(xs)
                else:
                    # Fuzzy fallback: LLM may return cell text with braces/tags
                    xs_clean = re.sub(r'[{}()\[\]]', ' ', xs)
                    xs_clean = re.sub(r'#\w+\([^)]*\)', '', xs_clean)
                    xs_clean = _clean_spaces(xs_clean).lower()
                    for e in entity_list:
                        el = e.lower()
                        if el in xs_clean or xs_clean in el:
                            if e not in selected:
                                selected.append(e)

    # fallback: greedy substring matching (colors + nouns)
    if not selected:
        ql = (question or "").lower()
        colors = []
        for c in ["white", "grey", "gray", "green", "yellow", "orange", "purple", "blue", "brown", "black", "red", "dark", "light", "beige", "pink"]:
            if re.search(rf"\b{c}\b", ql):
                colors.append(c)
        # map gray->grey
        colors = [("grey" if c == "gray" else c) for c in colors]

        # “thing” questions: select any entity containing the color
        if "thing" in ql and colors:
            for e in entity_list:
                el = e.lower()
                if any(col in el for col in colors):
                    selected.append(e)

        # common nouns -- expanded list
        noun_map = [
            ("tree", ["tree", "palm", "cactus", "trunk"]),
            ("bed", ["bed"]),
            ("table", ["table"]),
            ("window", ["window"]),
            ("curtain", ["curtain", "curtains"]),
            ("door", ["door", "doors"]),
            ("gate", ["gate", "gates"]),
            ("building", ["building", "house"]),
            ("courtyard", ["courtyard"]),
            ("lamp", ["lamp"]),
            ("painting", ["painting", "picture"]),
            ("rail", ["rail", "rails", "railing"]),
            ("balcony", ["balcony"]),
            ("room", ["room"]),
            ("wall", ["wall", "walls"]),
            ("man", ["man", "person", "people", "men"]),
            ("people", ["people", "person", "man", "men", "crowd"]),
            ("mountain", ["mountain", "mountains"]),
            ("frame", ["frame", "frames"]),
            ("bench", ["bench", "benches"]),
            ("pillar", ["pillar", "pillars", "column", "columns"]),
            ("terrace", ["terrace"]),
            ("sofa", ["sofa", "couch"]),
            ("chair", ["chair", "chairs"]),
            ("mirror", ["mirror"]),
            ("roof", ["roof", "rooftiles"]),
            ("pillow", ["pillow"]),
            ("cover", ["cover", "bedcover", "bedcovers"]),
            ("floor", ["floor"]),
            ("carpet", ["carpet", "rug"]),
            ("shelf", ["shelf", "shelves"]),
        ]
        for _, keys in noun_map:
            if any(re.search(rf"\b{k}\b", ql) for k in keys):
                for e in entity_list:
                    el = e.lower()
                    if any(k in el for k in keys):
                        selected.append(e)

    # unique preserve order
    seen = set()
    selected_u = []
    for x in selected:
        if x not in seen:
            selected_u.append(x)
            seen.add(x)

    return raw, {"selected_entities": selected_u}

def ollama_select_entities_timed(entity_list: List[str], question: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    t0 = _now()
    raw, obj = ollama_select_entities(entity_list, question)
    t1 = _now()
    return raw, obj, {"total_sec": _secs(t0, t1)}

# ============================================================
# GRID PRUNING PER QUESTION
# - keep only selected entities (+ nested brace context)
# - renumber rows/cols to minimal bounding box
# - re-render back to 5x5 format
# ============================================================

def _split_top_level_commas(text: str) -> List[str]:
    out, buf = [], []
    depth_paren = 0
    depth_brace = 0
    for ch in text:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)
        if ch == "," and depth_paren == 0 and depth_brace == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out
GRID_PRUNE_BY_GPT_PROMPT = r"""
You are given a FULL 5x5 grid, the exact entity strings found in that grid, and a QUESTION.

Your job has two parts:
1. Identify the IMPORTANT ENTITIES needed to answer the question.
2. Produce a QUESTION-AWARE PRUNED GRID that keeps only those entities and the minimum local
   container/context needed to interpret them.

CRITICAL RULES:
- For selected_entities, use ONLY strings that appear EXACTLY in the ENTITY LIST below.
  Copy-paste the exact string. Do NOT use cell text with braces/parens — use the clean entity name.
  WRONG: "white house { windows, terrace }"
  CORRECT: "white house"
- Select all entities directly referred to by the question.
- If the question uses vague descriptions like "white thing", "tree", "building", or
  "curtains", map them to the exact matching entity strings from the ENTITY LIST.
- If the question references something like "room", "wall", "man", "people" that is absent
  from the entity list, still select the closest matching entities if any exist.
  If nothing matches, return ALL entities so the grid is not empty.
- In the pruned grid, keep ONLY the selected entities and minimal supporting structure.
- Do NOT move entities to different cells.
- Keep the grid format EXACTLY the same: same 5x5 header, same separators, same row labels.
- Use "—" for empty cells.

DUPLICATE REMOVAL:
- If the same entity appears multiple times in the full grid, keep only the best-supported occurrence.
- Remove repeated standalone/container duplicates when they refer to the same entity.
- Prefer the occurrence that preserves the local structure needed for the question.

MINIMAL CONTEXT RULE:
- If a selected entity appears inside a container expression like (house { red roof , white door }),
  keep the minimal container text needed so the selected entity still makes sense.
- Do not keep unrelated entities just because they share a cell.

NEVER RETURN AN EMPTY GRID. If unsure which entities matter, keep ALL of them.

Return ONLY valid JSON:
{
  "selected_entities": ["<exact entity string>", "..."],
  "duplicate_entities_removed": ["<entity string>", "..."],
  "grid_ascii": "<pruned 5x5 grid>"
}

ENTITY LIST:
{entity_list}

QUESTION:
{question}

FULL GRID:
{grid_ascii}
"""
def gpt_prune_grid(grid_ascii: str, question: str, entity_list: List[str]) -> Tuple[str, Dict[str, Any]]:
    prompt = GRID_PRUNE_BY_GPT_PROMPT \
        .replace("{grid_ascii}", grid_ascii) \
        .replace("{question}", question) \
        .replace("{entity_list}", json.dumps(entity_list, ensure_ascii=False))

    raw = call_gpt_5_mini(prompt, model=GPT_GRID_MODEL)
    obj = _extract_first_json_obj(raw)

    if obj and "grid_ascii" in obj:
        selected_entities = obj.get("selected_entities") or []
        if not isinstance(selected_entities, list):
            selected_entities = []
        # Exact match first, then fuzzy fallback for unmatched
        matched = []
        for x in selected_entities:
            xs = str(x).strip()
            if xs in entity_list:
                if xs not in matched:
                    matched.append(xs)
            else:
                # Fuzzy: strip braces/tags, try substring match
                xs_clean = re.sub(r'[{}()\[\]]', ' ', xs)
                xs_clean = re.sub(r'#\w+\([^)]*\)', '', xs_clean)
                xs_clean = _clean_spaces(xs_clean).lower()
                for e in entity_list:
                    el = e.lower()
                    if el in xs_clean or xs_clean in el:
                        if e not in matched:
                            matched.append(e)
        obj["selected_entities"] = matched

        removed = obj.get("duplicate_entities_removed") or []
        if not isinstance(removed, list):
            removed = []
        obj["duplicate_entities_removed"] = [str(x).strip() for x in removed if str(x).strip()]

        # Safety net: if GPT returned an all-empty pruned grid, check if it's actually empty
        pruned_grid = obj.get("grid_ascii", "")
        grid_is_empty = True
        for line in pruned_grid.split('\n'):
            if not line.strip().startswith('Row'):
                continue
            content = re.sub(r'^Row\d+:\s*', '', line.strip())
            for cell in content.split('|'):
                cell = cell.strip()
                if cell and cell != '\u2014' and cell != '-':
                    grid_is_empty = False
                    break
            if not grid_is_empty:
                break

        if grid_is_empty:
            # GPT pruned grid is empty — try deterministic pruning with matched entities
            if matched:
                fallback_grid, _ = prune_grid_ascii(grid_ascii, matched)
                obj["grid_ascii"] = fallback_grid
                obj["_note"] = "gpt_pruned_was_empty_used_deterministic"
            else:
                # Last resort: use full grid as pruned grid
                obj["grid_ascii"] = grid_ascii
                obj["_note"] = "gpt_pruned_was_empty_using_full_grid"

        return raw, obj

    # Fallback: if GPT prune JSON fails, still let GPT select entities and use deterministic prune.
    sel_raw, sel_obj = ollama_select_entities(entity_list, question)
    selected_entities = sel_obj.get("selected_entities", []) or []
    if selected_entities:
        fallback_grid_ascii, fallback_meta = prune_grid_ascii(grid_ascii, selected_entities)
    else:
        # No entities matched at all — use full grid rather than blank
        fallback_grid_ascii = grid_ascii
        fallback_meta = {"note": "no_entities_matched_using_full_grid"}
    return raw + "\n\n#FALLBACK_SELECTION\n" + sel_raw, {
        "selected_entities": selected_entities,
        "duplicate_entities_removed": [],
        "grid_ascii": fallback_grid_ascii,
        "fallback_meta": fallback_meta,
    }

def _entity_matches_container_plus_descriptors(selected_ent: str, head_text: str, inside_items: List[str]) -> bool:
    """
    NEW: matches things like "light brown house" to "(house { light brown , red rooftiles })"
    Rule:
      - selected contains head token (e.g., "house")
      - AND selected contains at least one inside descriptor token (e.g., "light brown")
    """
    se = (selected_ent or "").strip().lower()
    head = (head_text or "").strip().lower()
    if not se or not head:
        return False
    if head not in se:
        return False
    for it in inside_items or []:
        itl = (it or "").strip().lower()
        if not itl:
            continue
        # require whole-word-ish match for descriptors too
        if _find_entity_in_cell(se, itl):
            return True
        # also allow substring for short adjectives like "brown"/"red"/"light"
        if itl in se and len(itl) <= 8:
            return True
    return False

def _cell_contains_selected_or_composite(cell: str, keep_list: List[str]) -> bool:
    """
    NEW: hit test that handles:
      - direct mention of selected entity
      - composite mention where selected spans container + brace descriptor
    """
    s = (cell or "").strip()
    if not s or s == "—":
        return False

    low = s.lower()

    # direct mention
    for e in keep_list:
        if _find_entity_in_cell(low, e.lower()):
            return True

    # composite: (head { items })
    m = re.match(r"^\((.*)\)$", s)
    if not m:
        return False
    inner = m.group(1).strip()
    m2 = re.match(r"^(.*)\{\s*(.*)\s*\}\s*$", inner)
    if not m2:
        return False
    head = m2.group(1).strip()
    inside = m2.group(2).strip()
    items = [x.strip() for x in _split_top_level_commas(inside) if x.strip()]
    for e in keep_list:
        if _entity_matches_container_plus_descriptors(e, head, items):
            return True

    return False
def _prune_cell_text(cell: str, keep_set: set) -> str:
    """
    Keeps only entity mentions from keep_set, but tries to preserve brace nesting structure.

    FIX YOU ASKED FOR:
    - If selected includes "light brown house" and grid has "(house { light brown , red rooftiles })",
      we KEEP the whole group.
    - If container is kept (house/building/window/etc.), we keep its brace items too.
    """
    s = (cell or "").strip()
    if not s or s == "—":
        return "—"

    keep_list = [x for x in keep_set if x]
    # NEW quick hit: includes composite match across container+descriptors
    if not _cell_contains_selected_or_composite(s, keep_list):
        return "—"

    def prune_expr(expr: str) -> str:
        expr = expr.strip()
        if not expr:
            return ""

        # parenthesized group
        m = re.match(r"^\((.*)\)$", expr.strip())
        if m:
            inner = m.group(1).strip()
            m2 = re.match(r"^(.*)\{\s*(.*)\s*\}\s*$", inner)
            if m2:
                head = m2.group(1).strip()
                inside = m2.group(2).strip()

                items = [x.strip() for x in _split_top_level_commas(inside) if x.strip()]

                # head direct match?
                head_keep = any(_find_entity_in_cell(head.lower(), e.lower()) for e in keep_list)

                # composite match? (e.g. "light brown house")
                composite_keep = any(_entity_matches_container_plus_descriptors(e, head, items) for e in keep_list)

                # If container is kept OR composite says "this is that house", KEEP ALL items (attributes)
                if head_keep or composite_keep:
                    # still prune nested parentheses inside items (for deeper containers),
                    # but we do NOT drop non-selected attribute items.
                    kept_items = []
                    for it in items:
                        itp = it.strip()
                        if not itp:
                            continue
                        if itp.startswith("(") and itp.endswith(")"):
                            itp_pruned = prune_expr(itp)
                            if itp_pruned:
                                kept_items.append(itp_pruned)
                        else:
                            kept_items.append(itp)
                    if kept_items:
                        return f"({head} {{ " + " , ".join(kept_items) + " }})"
                    return f"({head})"

                # Otherwise: keep only children that match selected (nested context)
                kept_items = []
                for it in items:
                    itp = it.strip()
                    if not itp:
                        continue
                    itp_pruned = prune_expr(itp)
                    if itp_pruned:
                        kept_items.append(itp_pruned)

                if kept_items:
                    return f"({head} {{ " + " , ".join(kept_items) + " }})"

                # if head not kept and no kept children -> drop
                return ""

            else:
                # plain parens: keep only if directly selected
                head_keep = any(_find_entity_in_cell(inner.lower(), e.lower()) for e in keep_list)
                return f"({inner})" if head_keep else ""
        else:
            # bare token(s): keep if contains a kept entity
            ok = any(_find_entity_in_cell(expr.lower(), e.lower()) for e in keep_list)
            return expr if ok else ""

    pruned = prune_expr(s)
    # fallback: if we had composite hit but pruning dropped it, keep original (better than blank)
    if not pruned:
        return s

    pruned = pruned.strip()
    return pruned if pruned else "—"

def prune_grid_ascii(grid_ascii: str, selected_entities: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (pruned_grid_ascii, meta)
    - meta includes mapping and bounding box info
    """
    cells = _parse_grid_cells(grid_ascii)
    if not cells:
        return grid_ascii, {"note": "grid_parse_failed"}

    keep_set = set([x for x in (selected_entities or []) if x])

    # prune each cell
    pruned_cells = {}
    for (r, c), txt in cells.items():
        pruned_cells[(r, c)] = _prune_cell_text(txt, keep_set)

    # compute bounding box of non-empty cells
    non_empty = [(r, c) for (r, c), txt in pruned_cells.items() if (txt or "").strip() != "—"]
    if not non_empty:
        # if nothing, keep a minimal blank grid
        blank = _render_5x5({(r, c): "—" for r in range(1, 6) for c in range(1, 6)})
        return blank, {"bbox": None, "note": "no_entities_found_in_grid"}

    rmin = min(r for r, _ in non_empty)
    rmax = max(r for r, _ in non_empty)
    cmin = min(c for _, c in non_empty)
    cmax = max(c for _, c in non_empty)

    height = rmax - rmin + 1
    width  = cmax - cmin + 1

    start_r = max(1, min(6 - height, 3 - (height // 2)))
    start_c = max(1, min(6 - width, 3 - (width // 2)))

    embedded = {(r, c): "—" for r in range(1, 6) for c in range(1, 6)}
    for rr in range(rmin, rmax + 1):
        for cc in range(cmin, cmax + 1):
            txt = pruned_cells.get((rr, cc), "—")
            tr = start_r + (rr - rmin)
            tc = start_c + (cc - cmin)
            if 1 <= tr <= 5 and 1 <= tc <= 5:
                embedded[(tr, tc)] = txt

    pruned_ascii = _render_5x5(embedded)
    meta = {
        "bbox": {"rmin": rmin, "rmax": rmax, "cmin": cmin, "cmax": cmax},
        "embedded_at": {"start_row": start_r, "start_col": start_c},
        "selected_entities": selected_entities,
    }
    return pruned_ascii, meta

def _render_5x5(cells: Dict[Tuple[int, int], str]) -> str:
    header = "Col1 | Col2 | Col3 | Col4 | Col5"
    sep = "--------------------------------"
    lines = [header, sep]
    for r in range(1, 6):
        row_cells = []
        for c in range(1, 6):
            txt = (cells.get((r, c), "—") or "—").strip()
            row_cells.append(txt if txt else "—")
        lines.append(f"Row{r}: " + " | ".join(row_cells))
    return "\n".join(lines)

def prune_grid_ascii_timed(grid_ascii: str, selected_entities: List[str]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    t0 = _now()
    pruned, meta = prune_grid_ascii(grid_ascii, selected_entities)
    t1 = _now()
    return pruned, meta, {"total_sec": _secs(t0, t1)}
def _normalize_relation_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accepts multiple possible extractor schemas and normalizes to:
    {"entities":[...], "triples":[{"head","relation","tail"}]}
    """
    if not isinstance(obj, dict):
        return {"entities": [], "triples": []}
    # Case A: already correct
    if "triples" in obj and isinstance(obj.get("triples"), list):
        return obj

    triples: List[Dict[str, Any]] = []
    entities: set = set()
    # Case B: your nested "room" format
    if "room" in obj and isinstance(obj["room"], dict):
        room = obj["room"]

        # room.contains: treat as room NTPPI entity
        contains = room.get("contains", [])
        if isinstance(contains, list):
            for item in contains:
                if not isinstance(item, dict):
                    continue
                ent = _clean_spaces(str(item.get("entity", "")))
                rel = _canon_rel(str(item.get("relation", "NTPPI")))
                if ent:
                    triples.append({"head": "room", "relation": rel or "NTPPI", "tail": ent})
                    entities.add(ent)
        entities.add("room")

        # room.relations: already head/rel/tail triples
        rels = room.get("relations", [])
        if isinstance(rels, list):
            for t in rels:
                if not isinstance(t, dict):
                    continue
                h = _clean_spaces(str(t.get("head", "")))
                r = _canon_rel(str(t.get("relation", "")))
                ta = _clean_spaces(str(t.get("tail", "")))
                if h and r and ta:
                    triples.append({"head": h, "relation": r, "tail": ta})
                    entities.add(h)
                    entities.add(ta)

        return {"entities": sorted(entities, key=lambda x: (-len(x), x.lower())), "triples": triples}

    # Case C: "relations" key at top-level instead of "triples"
    if "relations" in obj and isinstance(obj.get("relations"), list):
        for t in obj["relations"]:
            if not isinstance(t, dict):
                continue
            h = _clean_spaces(str(t.get("head", "")))
            r = _canon_rel(str(t.get("relation", "")))
            ta = _clean_spaces(str(t.get("tail", "")))
            if h and r and ta:
                triples.append({"head": h, "relation": r, "tail": ta})
                entities.add(h); entities.add(ta)
        return {"entities": sorted(entities, key=lambda x: (-len(x), x.lower())), "triples": triples}

    return {"entities": [], "triples": []}

# ============================================================
# GPT-5.1 QA
# ============================================================
def qwen_answer(prompt: str, candidates: List[str]) -> Tuple[str, Dict[str, Any]]:
    raw = call_qwen_cached(prompt)
    obj = _extract_first_json_obj(raw)
    if obj and isinstance(obj, dict):
        ans = obj.get("answer", obj.get("selected_option", ""))
        just = obj.get("justification", "")
        ans_norm = _normalize_candidate(str(ans), candidates)
        return raw, {"justification": str(just), "answer": ans_norm}
    ans2 = _extract_answer_fallback(raw, candidates)
    if ans2:
        return raw, {"justification": raw.strip()[:2000], "answer": ans2}

    return raw, {"justification": f"(parse_failed) {raw.strip()[:2000]}", "answer": candidates[0] if candidates else ""}

def qwen_answer_timed(prompt: str, candidates: List[str]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    t0 = _now()
    raw, obj = qwen_answer(prompt, candidates)
    t1 = _now()
    return raw, obj, {"total_sec": _secs(t0, t1)}

# ============================================================
# RUN ONE CONTEXT
# ============================================================
def run_one_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    t_ctx0 = _now()
    timings: Dict[str, Any] = {"context_total_sec": None}

    ctx_id = ctx.get("Context_id", "")
    story_list = ctx.get("story", [])
    t_story0 = _now()
    story_text = _format_story(story_list)
    timings["story_format_sec"] = _secs(t_story0)

    questions = ctx.get("questions", []) or []

    # 1) GPT relations
    t_rel0 = _now()
    rel_prompt = RELATION_EXTRACTION_PROMPT_OLLAMA.replace("{story_text}", story_text)
    rel_raw, rel_obj, rel_timing = ollama_extract_relations_timed(story_text)
    timings["gpt_relations"] = rel_timing
    timings["gpt_relations"]["wall_total_sec"] = _secs(t_rel0)

    triples = rel_obj.get("triples", [])
    triples_text = rel_raw.strip()[:4000]

    # 2) GPT grid (gen + possible repair)
    t_grid0 = _now()
    grid_prompt = GRID_GENERATION_PROMPT.replace("{story_text}", story_text).replace("{triples_text}", triples_text)
    grid_raw, grid_obj, grid_timing = gpt_generate_grid_timed(story_text, triples)
    timings["gpt_grid"] = grid_timing
    timings["gpt_grid"]["wall_total_sec"] = _secs(t_grid0)

    grid_ascii = str(grid_obj.get("grid_ascii", "")).strip()
    grid_entity_list = extract_grid_entity_list(grid_ascii)

    if PRINT_CONTEXT:
        print("\n" + "=" * 110)
        print("Context_id:", ctx_id)
        print("STORY:", story_text)

    if PRINT_RELATIONS:
        print("\nTRIPLES:\n", triples_text if triples_text else "(none)")

    if PRINT_GRID:
        print("\nGRID:\n", grid_ascii)
        print("Context_id:", ctx_id)

    # entity universe (once per context)
    # entity list derived from what ACTUALLY appears in the grid (critical for pruning)
    grid_entity_list = extract_grid_entity_list(grid_ascii)

    grid_raw, grid_obj, grid_timing = gpt_generate_grid_timed(story_text, triples)

    per_question = []
    mode_correct = {"text_only": 0, "relations_only": 0, "text_relations": 0, "grid_only_full": 0, "grid_only_pruned": 0}
    mode_total = 0

    timings["questions"] = []

    for qi, q in enumerate(questions):
        qtext = _clean_spaces(str(q.get("question", "")))
        candidates = q.get("candidate_answers", ["Yes", "No"])
        candidates = [str(x).strip() for x in (candidates or ["Yes", "No"])]
        gt = q.get("answer", [""])[0] if isinstance(q.get("answer"), list) else q.get("answer", "")
        gt = _normalize_candidate(str(gt), candidates)

        schema = build_qa_schema(candidates)

        # (A+B) question-aware GPT entity selection + prune in one step
        ent_prompt = GRID_PRUNE_BY_GPT_PROMPT.replace(
            "{grid_ascii}", grid_ascii
        ).replace(
            "{question}", qtext
        ).replace(
            "{entity_list}", json.dumps(grid_entity_list, ensure_ascii=False)
        )
        gt = _normalize_candidate(str(gt), candidates)
        t_prune0 = _now()
        pruned_raw, pruned_obj = gpt_prune_grid(grid_ascii, qtext, grid_entity_list)
        pruned_grid_ascii = pruned_obj.get("grid_ascii", "").strip()

        # Final safety net: if pruned grid is still entirely empty, use full grid
        _pruned_has_content = False
        for _line in pruned_grid_ascii.split('\n'):
            if not _line.strip().startswith('Row'):
                continue
            _content = re.sub(r'^Row\d+:\s*', '', _line.strip())
            for _cell in _content.split('|'):
                _cell = _cell.strip()
                if _cell and _cell != '\u2014' and _cell != '-':
                    _pruned_has_content = True
                    break
            if _pruned_has_content:
                break
        if not _pruned_has_content:
            pruned_grid_ascii = grid_ascii  # fall back to full grid
            pruned_obj["_note"] = "pruned_was_empty_fell_back_to_full_grid"

        prune_timing = {"total_sec": _secs(t_prune0)}
        ent_timing = {"total_sec": prune_timing["total_sec"], "source": "question_aware_gpt_prune"}
        selected_entities = pruned_obj.get("selected_entities", []) or []
        pruned_meta = {
            "method": "question_aware_gpt_prune",
            "duplicate_entities_removed": pruned_obj.get("duplicate_entities_removed", []) or [],
        }

        candidates = [str(x).strip() for x in (candidates or ["Yes", "No"])]
        if PRINT_PRUNED_GRID:
            print("\n[PRUNED GRID Q{}] selected={}".format(qi, selected_entities))
            print(pruned_grid_ascii)

        # prompts
        p_text = PROMPT_TEXT_ONLY.replace("{story_text}", story_text).replace("{question}", qtext).replace("{schema}", schema)
        p_rel  = PROMPT_RELATIONS_ONLY.replace("{triples_text}", triples_text).replace("{question}", qtext).replace("{schema}", schema)
        p_tr   = PROMPT_TEXT_RELATIONS.replace("{story_text}", story_text).replace("{triples_text}", triples_text).replace("{question}", qtext).replace("{schema}", schema)

        p_grid_full = PROMPT_GRID_ONLY.replace("{grid_ascii}", grid_ascii).replace("{question}", qtext).replace("{schema}", schema)
        p_grid_pruned = PROMPT_GRID_ONLY.replace("{grid_ascii}", pruned_grid_ascii).replace("{question}", qtext).replace("{schema}", schema)

        t_q0 = _now()
        raw_text, out_text, t_text = qwen_answer_timed(p_text, candidates)
        raw_rel,  out_rel,  t_relq = qwen_answer_timed(p_rel, candidates)
        raw_tr,   out_tr,   t_trq  = qwen_answer_timed(p_tr, candidates)
        raw_grid_full, out_grid_full, t_grid_full = qwen_answer_timed(p_grid_full, candidates)
        raw_grid_pruned, out_grid_pruned, t_grid_pruned = qwen_answer_timed(p_grid_pruned, candidates)
        t_q1 = _now()
        timings["questions"].append({
            "q_index": qi,
            "total_question_sec": _secs(t_q0, t_q1),

            "entity_selection": ent_timing,
            "grid_prune": prune_timing,

            "text_only": t_text,
            "relations_only": t_relq,
            "text_relations": t_trq,
            "grid_only_full": t_grid_full,
            "grid_only_pruned": t_grid_pruned,
        })

        def is_correct(pred_obj: Dict[str, Any]) -> bool:
            return _normalize_candidate(pred_obj.get("answer", ""), candidates) == gt

        rec = {
            "q_index": qi,
            "question": qtext,
            "ground_truth": gt,
            "meta": {
                "q_type": q.get("q_type"),
                "num_1st_context_sentences": q.get("num_1st_context_sentences"),
                "step_of_reasoning": q.get("step_of_reasoning"),
                "commonsense_question": q.get("commonsense_question"),
                "candidate_answers": candidates,
            },

            "entity_selection": {
                "entity_universe": grid_entity_list,
                "prompt": ent_prompt,
                "raw": pruned_raw,
                "selected_entities": selected_entities,
            },

            "grid_full": {
                "grid_ascii": grid_ascii,
                "prompt": p_grid_full,
            },
            "grid_pruned": {
                "grid_ascii": pruned_grid_ascii,
                "meta": pruned_meta,
                "prompt": p_grid_pruned,
            },

            "text_only": {
                "prompt": p_text,
                "raw": raw_text,
                "justification": out_text.get("justification", ""),
                "answer": out_text.get("answer", ""),
                "lm_correct": is_correct(out_text),
            },
            "relations_only": {
                "prompt": p_rel,
                "raw": raw_rel,
                "justification": out_rel.get("justification", ""),
                "answer": out_rel.get("answer", ""),
                "lm_correct": is_correct(out_rel),
            },
            "text_relations": {
                "prompt": p_tr,
                "raw": raw_tr,
                "justification": out_tr.get("justification", ""),
                "answer": out_tr.get("answer", ""),
                "lm_correct": is_correct(out_tr),
            },
            "grid_only_full": {
                "prompt": p_grid_full,
                "raw": raw_grid_full,
                "justification": out_grid_full.get("justification", ""),
                "answer": out_grid_full.get("answer", ""),
                "lm_correct": is_correct(out_grid_full),
            },
            "grid_only_pruned": {
                "prompt": p_grid_pruned,
                "raw": raw_grid_pruned,
                "justification": out_grid_pruned.get("justification", ""),
                "answer": out_grid_pruned.get("answer", ""),
                "lm_correct": is_correct(out_grid_pruned),
            },
        }

        per_question.append(rec)

        mode_total += 1
        for m in ["text_only", "relations_only", "text_relations", "grid_only_full", "grid_only_pruned"]:
            if rec[m]["lm_correct"]:
                mode_correct[m] += 1

        if PRINT_QA_SUMMARY:
            print("\n--- Q", qi, "---")
            print("Q:", qtext)
            print("GT:", gt)
            print(" text_only        :", rec["text_only"]["answer"], rec["text_only"]["lm_correct"])
            print(" relations_only   :", rec["relations_only"]["answer"], rec["relations_only"]["lm_correct"])
            print(" text_relations   :", rec["text_relations"]["answer"], rec["text_relations"]["lm_correct"])
            print(" grid_only_full   :", rec["grid_only_full"]["answer"], rec["grid_only_full"]["lm_correct"])
            print(" grid_only_pruned :", rec["grid_only_pruned"]["answer"], rec["grid_only_pruned"]["lm_correct"])

    ctx_summary = {}
    if mode_total > 0:
        for k in mode_correct:
            ctx_summary[k] = {
                "correct": mode_correct[k],
                "total": mode_total,
                "accuracy": mode_correct[k] / mode_total
            }
    else:
        for k in mode_correct:
            ctx_summary[k] = {"correct": 0, "total": 0, "accuracy": None}

    timings["context_total_sec"] = _secs(t_ctx0)

    return {
        "Context_id": ctx_id,
        "story": story_list,
        "story_text": story_text,

        "prompts": {
            "relations_prompt": rel_prompt,
            "grid_prompt": grid_prompt,
        },

        "gpt_relations": {
            "raw": rel_raw,
            "parsed": rel_obj,
            "triples_text": triples_text,
        },

        "gpt_grid": {
            "raw": grid_raw,
            "parsed": grid_obj,
            "grid_ascii": grid_ascii,
        },

        "questions": questions,
        "per_question_records": per_question,
        "context_summary": ctx_summary,
        "timings": timings,
    }

# ============================================================
# MAIN
# ============================================================
def _already_done_ok(out_path: str) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and obj.get("error"):
            return False
        if isinstance(obj, dict) and ("context_summary" in obj or "per_question_records" in obj):
            return True
        return False
    except Exception:
        return False

def main() -> None:
    t_run0 = _now()

    # Load existing results from single output file (resume support)
    _all_results: List[Dict] = []
    if Path(OUTPUT_JSON).exists():
        try:
            _prev = _read_json(OUTPUT_JSON)
            _all_results = _prev.get("results", []) if isinstance(_prev, dict) else []
        except Exception:
            _all_results = []
    _done_ids = {r.get("Context_id") for r in _all_results if isinstance(r, dict)}

    dataset = _read_json(INPUT_JSON)
    data = dataset.get("data", []) or []

    if SKIP_CONTEXTS:
        data = data[int(SKIP_CONTEXTS):]
    if PROCESS_CONTEXTS is not None:
        data = data[: int(PROCESS_CONTEXTS)]
    print(f"Loaded dataset: {dataset.get('name','(unknown)')}  contexts_in_file={len(data)}")
    print(f"Output -> {OUTPUT_JSON}")
    print("Resume mode: will SKIP contexts that already have a successful output in the results file.\n")
    index_entries = []
    global_counts = defaultdict(int)
    global_totals = defaultdict(int)

    processed = 0
    skipped = 0

    for i, ctx in enumerate(data):
        ctx_id = ctx.get("Context_id", f"ctx_{i}")

        if ctx_id in _done_ids:
            skipped += 1
            index_entries.append({
                "Context_id": ctx_id,
                "error": None,
                "status": "skipped_existing_ok"
            })
            continue

        try:
            result = run_one_context(ctx)
            err = None
            processed += 1

            cs = result.get("context_summary", {})
            for mode in ["text_only", "relations_only", "text_relations", "grid_only_full", "grid_only_pruned"]:
                c = cs.get(mode, {}).get("correct", 0)
                t = cs.get(mode, {}).get("total", 0)
                global_counts[mode] += int(c)
                global_totals[mode] += int(t)

        except Exception as e:
            err = str(e)
            processed += 1
            result = {
                "Context_id": ctx_id,
                "error": err,
                "story": ctx.get("story", []),
                "questions": ctx.get("questions", []),
            }
            print("\n[ERROR]", ctx_id, "->", err)
            processed += 1
        _all_results.append(result)
        _done_ids.add(ctx_id)
        _write_json(OUTPUT_JSON, {"results": _all_results})

        entry = {
            "Context_id": ctx_id,
            "error": err,
            "status": "processed"
        }
        if INDEX_INCLUDE_GRID and isinstance(result, dict):
            entry["grid_ascii"] = result.get("gpt_grid", {}).get("grid_ascii", "")
        index_entries.append(entry)

    global_summary = {}
    for mode in ["text_only", "relations_only", "text_relations", "grid_only_full", "grid_only_pruned"]:
        t = global_totals[mode]
        c = global_counts[mode]
        global_summary[mode] = {
            "correct": c,
            "total": t,
            "accuracy": (c / t) if t else None
        }

    run_total_sec = _secs(t_run0)

    index_obj = {
        "dataset_name": dataset.get("name"),
        "skip_contexts": SKIP_CONTEXTS,
        "process_contexts": PROCESS_CONTEXTS,
        "num_contexts_in_input_file": len(data),
        "num_contexts_processed_this_run": processed,
        "num_contexts_skipped_existing_ok": skipped,
        "models": {
            "relations_model": REL_OLLAMA_MODEL_NAME,
            "entity_selection_model": ENTSEL_OLLAMA_MODEL_NAME,
            "gpt_grid_model": GPT_GRID_MODEL,
            "qa_model": QWEN_MODEL_NAME,
            "qa_temperature": QWEN_TEMPERATURE,
        },
        "run_timing": {
            "run_total_sec": run_total_sec
        },
        "global_summary": global_summary,
        "index": index_entries,
        "results": _all_results,
    }

    _write_json(OUTPUT_JSON, index_obj)
    print("\nDone.")
    print("Results written:", OUTPUT_JSON)
    print(f"\nProcessed this run: {processed}   Skipped (already done): {skipped}")
    print("\nGlobal summary (THIS RUN ONLY, over newly processed contexts):")
    for k, v in global_summary.items():
        print(f"  {k}: {v['correct']}/{v['total']}  acc={v['accuracy']}")

    print(f"\nRun time (sec): {run_total_sec}")


if __name__ == "__main__":
    main()
