# prompts_lib.py
# All prompt builders live here (so the main script is clean).

import re
from typing import Any, Dict, List, Optional

# -----------------------------
# Prompt builders: relations/story
# -----------------------------

def build_prompt_relations_only(relations_text: str, question: str, candidate_answers: List[str]) -> str:
    cand_text = ", ".join(candidate_answers) if candidate_answers else "Yes, No"
    return (
        "You are given ONLY spatial relations extracted from text.\n"
        "Think carefully but DO NOT show steps.\n"
        'Return ONLY JSON: {"justification": "...", "selected_option": "<all valid candidates>"}\n\n'
        f"Relations:\n{relations_text}\n\n"
        f"Question: {question}\n"
        f"Candidate answers: {cand_text}\n"
        "Answer:"
    )


def build_prompt_story_only(story: Any, question: str, candidate_answers: List[str]) -> str:
    story_text = "\n".join(story) if isinstance(story, list) else str(story)
    cand_text = ", ".join(candidate_answers) if candidate_answers else "Yes, No"
    return (
        "You are given the story text. Answer ONLY from this story.\n"
        "Think carefully but DO NOT show steps.\n"
        'Return ONLY JSON: {"justification": "...", "selected_option": "<all valid candidates>"}\n\n'
        f"Story:\n{story_text}\n\n"
        f"Question: {question}\n"
        f"Candidate answers: {cand_text}\n"
        "Answer:"
    )


# =========================================================
# TWO-PROMPT GRID SETUP (these names MUST exist)
# =========================================================
# TWO-PROMPT GRID SETUP (these names MUST exist)
# =========================================================

from typing import List, Optional
from typing import List, Optional

# =========================================================
# PROMPT 1 — Grid → Simple interpretation (NO answering)
# =========================================================
def build_prompt_grid_interpretation(grid_str: str) -> str:
    """
    PROMPT 1 — Grid → Simple interpretation (NO answering)
    Output MUST be plain text interpretation only (no JSON).
    """
    return (
        "You are given a textual grid that encodes spatial layout with boxes (containers) and objects inside them.\n\n"
        "Your task: ONLY interpret the grid in simple human sentences.\n"
        "Do NOT answer the question. Do NOT select options.\n\n"
        "How to read the grid:\n"
        "1. Each Row(k) is listed from top to bottom; smaller Row(k) is higher in space.\n"
        "2. Each Col(k) is listed from left to right; smaller Col(k) is more left in space.\n"
        "3. A line like `[box X:` starts a container (a box).\n"
        "4. The indented lines that follow it (e.g. `Row(r) Col(c) object_in(box X) ...`) are the objects INSIDE that box.\n"
        "5. If a box A appears above a box B in the grid, treat A as ABOVE B (unless the grid says otherwise).\n"
        "6. Use ONLY what is written in the grid. Do not invent extra objects or relations.\n\n"
        "COLUMN FORMAT REQUIREMENT (MANDATORY):\n"
        "- The grid header lists columns like: `Col(1)  Col(2)`.\n"
        "- EVERY entity line MUST explicitly contain `Row(r) Col(c)`.\n"
        "- Container boundary lines like `Row(k) [box X:` do NOT need Col(c).\n\n"
        "INTERPRETATION FORMAT (plain text; bullets ok):\n"
        "- Boxes present: <list box names in the order they appear>\n"
        "- Global objects (if any are outside boxes):\n"
        "  - <object name> at Row(r) Col(c) with tags: <tags>\n"
        "- For each box:\n"
        "  - Box <name> has these objects:\n"
        "    - <object name> at Row(r) Col(c) with tags: <tags>\n"
        "  - If any explicit tags exist between objects (e.g., `#near-to(X)`, `#dc-from(X)`, `#touching: X`), write them as explicit statements.\n"
        "- If a box has box-level tags like `#dc-from(BoxX)` / `#ec-from(BoxX)` / `#po-from(BoxX)` / "
        "`#near-to(BoxX)` / `#far-from(BoxX)`, state them explicitly.\n\n"
        "CRITICAL TAG RULES FOR INTERPRETATION:\n"
        "- Only treat dc/ec/po/near/far/behind/front as TRUE if they appear as explicit tags in the grid.\n"
        "- Do NOT infer dc/ec/po/near/far/behind/front from spacing, row/col differences, or being in different boxes.\n"
        "- `#touch-edge` and `#inside-clear` are containment signals (TPP/NTPP), NOT dc/ec.\n"
        "- `#touching: X` is an explicit entity-entity touch statement.\n"
        "- If a BOX has a tag like `#dc-from(BoxX)`, it is a box-level property (later steps may propagate it to contents).\n"
        "- If an OBJECT has a tag like `#dc` or `#dc-from(X)`, it applies only to that object.\n\n"
        "EXAMPLES:\n\n"
        "Example 1 – ABOVE / BELOW\n"
        "            Col(1)\n"
        "Row(5) Col(1) medium orange apple_in(box two)\n"
        "Row(6) Col(1) medium red apple_in(box two)\n"
        "Explanation: Same column. Row(5) < Row(6) ⇒ orange apple is ABOVE red apple.\n\n"
        "---\n\n"
        "Example 2 – LEFT / RIGHT\n"
        "            Col(1)  Col(2)\n"
        "Row(4) Col(1) small green cube_in(box X)\n"
        "Row(4) Col(2) small red cube_in(box X)\n"
        "Explanation: Same row. Col(1) < Col(2) ⇒ green cube is LEFT of red cube.\n\n"
        "---\n\n"
        "Example 3 – FRONT / BEHIND (TAG ONLY)\n"
        "Row(2) Col(1) tiny blue sphere_in(box A)   #behind(large red sphere)\n"
        "Row(3) Col(1) large red sphere_in(box A)  #front(tiny blue sphere)\n"
        "Explanation: behind/front are TRUE only because explicit tags exist.\n\n"
        "---\n\n"
        "Example 4 – NEAR / FAR (TAG ONLY)\n"
        "Row(3) Col(1) medium green apple_in(box two)   #near-to(medium red apple)\n"
        "Row(4) Col(1) medium red apple_in(box two)\n"
        "Explanation: near is TRUE only because `#near-to(...)` tag exists.\n\n"
        "---\n\n"
        "Example 5 – DC / EC / PO (TAG ONLY)\n"
        "Row(1) [box HHH:\n"
        "Row(2)   Col(1) medium red hexagon_in(box HHH)\n"
        "Row(3) ]\n"
        "Row(4) [box LLL:   #dc-from(box HHH)\n"
        "Row(5)   Col(1) medium grey hexagon_in(box LLL)\n"
        "Row(6) ]\n"
        "Explanation: dc is TRUE only because `#dc-from(box HHH)` exists.\n\n"
        "---\n\n"
        "Example 6 – TPP / NTPP (CONTAINMENT TAGS)\n"
        "Row(1) [box ONE:\n"
        "Row(2)   Col(1) medium yellow apple_in(box ONE)   #touch-edge\n"
        "Row(3) ]\n"
        "Explanation: #touch-edge => apple tpp box ONE.\n\n"
        "GRID:\n"
        f"{grid_str}\n"
    )


# =========================================================
# PROMPT 1-FR — Grid → Interpretation for FR questions
# =========================================================
def build_prompt_grid_interpretation_fr(grid_str: str) -> str:
    """
    PROMPT 1 (FR-specific) — Grid → Interpretation.
    Identical to YN version EXCEPT:
      - Requires noting box header Row/Col for cross-box directional inference.
      - Requires a cross-box spatial summary section.
      - Includes Example 7 (cross-box directional).
    """
    return (
        "You are given a textual grid that encodes spatial layout with boxes (containers) and objects inside them.\n\n"
        "Your task: ONLY interpret the grid in simple human sentences.\n"
        "Do NOT answer the question. Do NOT select options.\n\n"
        "How to read the grid:\n"
        "1. Each Row(k) is listed from top to bottom; smaller Row(k) is higher in space (ABOVE).\n"
        "2. Each Col(k) is listed from left to right; smaller Col(k) is more left in space.\n"
        "3. A line like `[box X:` starts a container (a box). The Row(N) Col(M) on that line is the box's spatial position.\n"
        "4. The indented lines that follow it (e.g. `Row(r) Col(c) object_in(box X) ...`) are the objects INSIDE that box.\n"
        "5. To determine if box A is ABOVE/BELOW/LEFT/RIGHT of box B, compare their HEADER Row/Col values.\n"
        "   A box at Row(1) is ABOVE a box at Row(6). A box at Col(2) is RIGHT of a box at Col(1).\n"
        "6. Objects INHERIT the spatial position of their enclosing box for cross-box comparisons.\n"
        "   An object in box A (header Row 1) is ABOVE any object in box B (header Row 6).\n"
        "7. Use ONLY what is written in the grid. Do not invent extra objects or relations.\n\n"
        "COLUMN FORMAT REQUIREMENT (MANDATORY):\n"
        "- The grid header lists columns like: `Col(1)  Col(2)`.\n"
        "- EVERY entity line MUST explicitly contain `Row(r) Col(c)`.\n"
        "- Container boundary lines like `Row(k) [box X:` do NOT need Col(c).\n\n"
        "INTERPRETATION FORMAT (plain text; bullets ok):\n"
        "- Boxes present: <list box names in the order they appear>\n"
        "- Global objects (if any are outside boxes):\n"
        "  - <object name> at Row(r) Col(c) with tags: <tags>\n"
        "- For each box:\n"
        "  - Box <name> starts at Row(r) Col(c) [ALWAYS note the box header's Row and Col]\n"
        "  - Box <name> has these objects:\n"
        "    - <object name> at Row(r) Col(c) with tags: <tags>\n"
        "  - If any explicit tags exist between objects (e.g., `#near-to(X)`, `#dc-from(X)`, `#touching: X`),\n"
        "    write them as explicit statements.\n"
        "- If a box has box-level tags like `#dc-from(BoxX)` / `#ec-from(BoxX)` / `#po-from(BoxX)` / "
        "`#near-to(BoxX)` / `#far-from(BoxX)` / `#behind(BoxX)` / `#front(BoxX)`, state them explicitly.\n"
        "- CROSS-BOX SPATIAL SUMMARY (MANDATORY when multiple boxes exist):\n"
        "  Compare the box header Row/Col values and state which box is ABOVE/BELOW/LEFT/RIGHT\n"
        "  of which other box. This is critical for answering directional questions later.\n\n"
        "CRITICAL TAG RULES FOR INTERPRETATION:\n"
        "- Only treat dc/ec/po/near/far/behind/front as TRUE if they appear as explicit tags in the grid.\n"
        "- Do NOT infer dc/ec/po/near/far/behind/front from spacing, row/col differences, or being in different boxes.\n"
        "- `#touch-edge` and `#inside-clear` are containment signals (TPP/NTPP), NOT dc/ec.\n"
        "- `#touching: X` is an explicit entity-entity touch statement.\n"
        "- If a BOX has a tag like `#dc-from(BoxX)`, it is a box-level property (later steps may propagate it to contents).\n"
        "- If an OBJECT has a tag like `#dc` or `#dc-from(X)`, it applies only to that object.\n\n"
        "EXAMPLES:\n\n"
        "Example 1 – ABOVE / BELOW (same box)\n"
        "            Col(1)\n"
        "Row(5) Col(1) medium orange apple_in(box two)\n"
        "Row(6) Col(1) medium red apple_in(box two)\n"
        "Explanation: Same column. Row(5) < Row(6) ⇒ orange apple is ABOVE red apple.\n\n"
        "---\n\n"
        "Example 2 – LEFT / RIGHT (same box)\n"
        "            Col(1)  Col(2)\n"
        "Row(4) Col(1) small green cube_in(box X)\n"
        "Row(4) Col(2) small red cube_in(box X)\n"
        "Explanation: Same row. Col(1) < Col(2) ⇒ green cube is LEFT of red cube.\n\n"
        "---\n\n"
        "Example 3 – FRONT / BEHIND (TAG ONLY)\n"
        "Row(2) Col(1) tiny blue sphere_in(box A)   #behind(large red sphere)\n"
        "Row(3) Col(1) large red sphere_in(box A)  #front(tiny blue sphere)\n"
        "Explanation: behind/front are TRUE only because explicit tags exist.\n\n"
        "---\n\n"
        "Example 4 – NEAR / FAR (TAG ONLY)\n"
        "Row(3) Col(1) medium green apple_in(box two)   #near-to(medium red apple)\n"
        "Row(4) Col(1) medium red apple_in(box two)\n"
        "Explanation: near is TRUE only because `#near-to(...)` tag exists.\n\n"
        "---\n\n"
        "Example 5 – DC / EC / PO (TAG ONLY)\n"
        "Row(1) [box HHH:\n"
        "Row(2)   Col(1) medium red hexagon_in(box HHH)\n"
        "Row(3) ]\n"
        "Row(4) [box LLL:   #dc-from(box HHH)\n"
        "Row(5)   Col(1) medium grey hexagon_in(box LLL)\n"
        "Row(6) ]\n"
        "Explanation: dc is TRUE only because `#dc-from(box HHH)` exists.\n\n"
        "---\n\n"
        "Example 6 – TPP / NTPP (CONTAINMENT TAGS)\n"
        "Row(1) [box ONE:\n"
        "Row(2)   Col(1) medium yellow apple_in(box ONE)   #touch-edge\n"
        "Row(3) ]\n"
        "Explanation: #touch-edge => apple tpp box ONE.\n\n"
        "---\n\n"
        "Example 7 – CROSS-BOX DIRECTIONAL (from box header positions)\n"
        "Row(1) Col(2) [box LLL:   #dc(box HHH)\n"
        "Row(2) Col(2) medium grey star_in(box LLL)   #inside-clear\n"
        "Row(3) Col(2) ]\n"
        "Row(5) Col(1) [box HHH:   #dc(box LLL)\n"
        "Row(6) Col(1) purple pentagon_in(box HHH)   #touch-edge\n"
        "Row(7) Col(1) ]\n"
        "Interpretation output should include:\n"
        "  - Box LLL starts at Row(1) Col(2), has #dc(box HHH)\n"
        "  - Box HHH starts at Row(5) Col(1), has #dc(box LLL)\n"
        "  - Cross-box spatial summary: LLL is ABOVE HHH (Row 1 < Row 5) and RIGHT of HHH (Col 2 > Col 1)\n"
        "  - Therefore: grey star (in LLL) is ABOVE and RIGHT of purple pentagon (in HHH)\n\n"
        "GRID:\n"
        f"{grid_str}\n"
    )


# =========================================================
# PROMPT 2 — Question → Question-plan JSON (NO answering)
# =========================================================
def build_prompt_question_interpretation(question: str, candidate_answers: List[str]) -> str:
    cand_text = ", ".join(candidate_answers) if candidate_answers else "Yes, No"
    return (
        "You are given:\n"
        "(1) QUESTION\n"
        "(2) Candidate answers\n\n"
        "Your task: interpret the QUESTION ONLY (no grid). Do NOT answer.\n\n"
        "OUTPUT: Return ONLY JSON with this schema:\n"
        "{"
        "\"q_mode\":\"FR|YN\","
        "\"referents\":[\"...\"],"
        "\"quantifiers\":{\"any\":false,\"all\":false,\"other\":false},"
        "\"scope\":\"GLOBAL|IN-BOX\","
        "\"requested_relations\":[\"...\"]"
        "}\n\n"
        "DETERMINE q_mode:\n"
        "- If candidates are exactly Yes/No (or only contain Yes/No), set q_mode=YN.\n"
        "- Otherwise set q_mode=FR.\n\n"
        "requested_relations RULE:\n"
        "- requested_relations MUST be a subset of Candidate answers.\n"
        "- NEVER add a relation that is not in candidates.\n\n"
        "WHAT TO PUT IN referents (CRITICAL):\n"
        "- Put the exact entity strings mentioned in the question (boxes/blocks/objects/categories).\n"
        "- Boxes/Blocks are interchangeable words.\n"
        "- If the question mentions TWO things (subject + target), include both in referents in the same order.\n"
        "- If the question is about a SET/category (e.g., \"medium thing\", \"any square\", \"all apples\"),\n"
        "  include that category phrase as a referent string (do NOT invent indices).\n\n"
        "SCOPE RULE:\n"
        "- If the question says \"in box X\" / \"inside block X\" / \"within X\" => scope=IN-BOX and include X as a referent.\n"
        "- Else => scope=GLOBAL.\n\n"
        "QUANTIFIERS (WITH EXAMPLES):\n"
        "- any / a / an / some / is there a ... => quantifiers.any=true (EXISTENTIAL)\n"
        "  Example: \"Is any medium object left of the red hexagon?\" -> any=true\n"
        "- all / each / every => quantifiers.all=true (UNIVERSAL)\n"
        "  Example: \"Are all squares inside box HHH?\" -> all=true\n"
        "- other / another (when it excludes the referenced item itself) => quantifiers.other=true\n"
        "  Example: \"Is any other apple above the apple?\" -> other=true\n\n"
        "CATEGORY / TYPE MATCHING INTENT (IMPORTANT):\n"
        "- If question says \"medium thing\" => it refers to ANY entity whose name includes \"medium\" (shape type can vary).\n"
        "- If question says \"square\" => it refers to ANY entity whose name includes \"square\" (color/size may vary unless specified).\n"
        "- If question says \"purple thing\" => it refers to ANY entity whose name includes \"purple\".\n"
        "- If question specifies both size+color+shape, match that full description.\n\n"
        "NUMBERING NORMALIZATION (CRITICAL):\n"
        "- If question says \"X number one\" but only \"X\" exists in the world, treat them as the same referent.\n"
        "- If question says \"X\" with no number, interpret it as \"X number one\".\n\n"
        "RELATION DIRECTION (ORDERING):\n"
        "- If the question says \"relation of X to Y\" or \"X relative to Y\" or \"Where is X regarding Y\",\n"
        "  treat the ordered pair as (X, Y) when later computing relations.\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Candidate answers:\n"
        f"{cand_text}\n\n"
        "Return ONLY JSON.\n"
    )


# =========================================================
# PROMPT 3 — Interpretation + Question-plan → Answer JSON
# =========================================================
def build_prompt_grid_answer_from_interp_and_question_plan(
    interp_text: str,
    question_plan_json: str,
    candidate_answers: List[str],
) -> str:
    cand_text = ", ".join(candidate_answers) if candidate_answers else "Yes, No"
    return (
        "You are given:\n"
        "(1) INTERPRETATION TEXT of the grid (from Prompt 1)\n"
        "(2) QUESTION_PLAN JSON (from Prompt 2)\n"
        "(3) Candidate answers\n\n"
        "TASK:\n"
        "- Use ONLY INTERPRETATION TEXT + QUESTION_PLAN.\n"
        "- Output ONLY JSON:\n"
        "{\"justification\":\"...\",\"selected_option\":[...]}\n\n"
        "NON-NEGOTIABLE RULES:\n"
        "A) Use QUESTION_PLAN as ground truth for: q_mode, scope, referents, quantifiers, requested_relations.\n"
        "B) MULTI-LABEL: output ALL true relations that are in candidates.\n"
        "C) INDEPENDENCE: front/behind does not cancel other relations.\n"
        "D) Do NOT claim 'no other objects exist' unless Prompt 1 enumerates them.\n\n"
        "TOP-LEVEL PRINCIPLE:\n"
        "- You MUST evaluate ALL 15 relations one-by-one and explicitly justify TRUE vs NOT-TRUE for each.\n"
        "- Only return those that are TRUE AND present in candidate answers.\n\n"
        "THE 15 RELATIONS (MUST CHECK ALL):\n"
        "left, right, above, below, behind, front, near, far, dc, ec, po, tpp, ntpp, tppi, ntppi\n\n"

        "═══════════════════════════════════════════════════════\n"
        "CROSS-BOX DIRECTIONAL RULE (CRITICAL — TYPE 1 FIX)\n"
        "═══════════════════════════════════════════════════════\n"
        "When the subject and reference are in DIFFERENT boxes, you MUST determine\n"
        "left/right/above/below by comparing the Row and Col of each box's OPENING\n"
        "HEADER LINE (the line that starts with `[box X:` or `[block X:`).\n\n"
        "  - Find the Row(N) on the header line of subject's box → call it subj_box_row.\n"
        "  - Find the Row(N) on the header line of reference's box → call it ref_box_row.\n"
        "  - If subj_box_row < ref_box_row → subject is ABOVE reference.\n"
        "  - If subj_box_row > ref_box_row → subject is BELOW reference.\n"
        "  - Similarly compare Col(N) on the header lines for LEFT / RIGHT.\n\n"
        "Objects INHERIT the directional position of their enclosing box.\n"
        "An object in box A at Row(1) is ABOVE an object in box B at Row(6),\n"
        "REGARDLESS of the object's own Row within its box. The box header Row is\n"
        "what determines cross-box direction.\n\n"
        "Example:\n"
        "  Row(1) [box LLL:       ← LLL header at Row(1) Col(2)\n"
        "  Row(2)   star_in(LLL)\n"
        "  Row(5) [box HHH:       ← HHH header at Row(5) Col(1)\n"
        "  Row(6)   pentagon_in(HHH)\n"
        "  → star is ABOVE pentagon (Row 1 < Row 5)\n"
        "  → star is RIGHT of pentagon (Col 2 > Col 1)\n\n"
        "NEVER say 'directional NOT-TRUE because different boxes'.\n"
        "Different boxes is EXACTLY when you MUST compare box header Row/Col.\n\n"

        "═══════════════════════════════════════════════════════\n"
        "ORDERED PAIR RULE (CRITICAL — TYPE 2/3/6 FIX)\n"
        "═══════════════════════════════════════════════════════\n"
        "The question defines an ORDERED PAIR: 'Where is A relative to B' means\n"
        "you are computing relation(A, B). ALL relations must be from A's perspective.\n\n"
        "MUTUAL-EXCLUSION CONSTRAINTS for a single ordered pair (A, B):\n"
        "- Output AT MOST ONE of {above, below}. If A.row < B.row → above. Never both.\n"
        "- Output AT MOST ONE of {left, right}. If A.col < B.col → left. Never both.\n"
        "- Output AT MOST ONE of {ntpp, ntppi}. These are mutually exclusive:\n"
        "    ntpp(A,B) means A is properly inside B.\n"
        "    ntppi(A,B) means B is properly inside A.\n"
        "    NEVER output both for the same ordered pair.\n"
        "- Output AT MOST ONE of {tpp, tppi}. Same logic.\n"
        "- NEVER output a containment relation AND its inverse for the same pair.\n"
        "    e.g., [ntpp, ntppi] is ALWAYS WRONG for one ordered pair.\n"
        "    e.g., [tpp, tppi] is ALWAYS WRONG for one ordered pair.\n\n"
        "HOW TO PICK THE CORRECT CONTAINMENT DIRECTION:\n"
        "- If A is inside B (A has #inside-clear in B) → ntpp(A,B). NOT ntppi.\n"
        "- If B is inside A (B has #inside-clear in A) → ntppi(A,B). NOT ntpp.\n"
        "- If A touches edge of B (A has #touch-edge in B) → tpp(A,B). NOT tppi.\n"
        "- If B touches edge of A (B has #touch-edge in A) → tppi(A,B). NOT tpp.\n\n"

        "═══════════════════════════════════════════════════════\n"
        "ROW DIRECTION RULE (CRITICAL — TYPE 4 FIX)\n"
        "═══════════════════════════════════════════════════════\n"
        "Lower Row number = geometrically HIGHER in space (ABOVE / NORTH).\n"
        "Higher Row number = geometrically LOWER in space (BELOW / SOUTH).\n\n"
        "For ordered pair (subject, reference):\n"
        "  - If subject.Row < reference.Row → subject is ABOVE reference.\n"
        "  - If subject.Row > reference.Row → subject is BELOW reference.\n"
        "  - If subject.Col < reference.Col → subject is LEFT of reference.\n"
        "  - If subject.Col > reference.Col → subject is RIGHT of reference.\n"
        "Always evaluate from the SUBJECT's perspective. Never flip.\n\n"

        "═══════════════════════════════════════════════════════\n"
        "TAG-ONLY POLICY (CRITICAL — TYPE 5/9 FIX)\n"
        "═══════════════════════════════════════════════════════\n"
        "dc/ec/po/near/far/front/behind require EXPLICIT TAGS:\n"
        "- dc: TRUE only with explicit #dc or #dc-from(...) tag relevant to the pair.\n"
        "  Do NOT infer DC from the ABSENCE of #touching — two objects in the same box\n"
        "  that do not touch are NOT automatically DC unless an explicit #dc tag exists.\n"
        "- ec: TRUE only with explicit #ec, #ec-from(...), or #touching: X between the pair.\n"
        "- po: TRUE only with explicit #po or #po-from(...) tag.\n"
        "- near: TRUE only if the SUBJECT OBJECT has an explicit #near(target) or #near-to(target) tag,\n"
        "  OR both subject and reference are in boxes where the subject's box has #near(ref_box) or\n"
        "  #near-to(ref_box). Do NOT output near just because a box has #near(other_box) if the\n"
        "  reference is not in that other_box.\n"
        "- far: TRUE only with explicit #far-from(...) or #far(...) tag with same propagation logic.\n"
        "- behind/front: TRUE only with explicit #behind(...) / #front(...) tags.\n"
        "- NEVER infer dc/ec/po/near/far/behind/front from row/col distance, spacing, or box membership.\n\n"

        "ONE-WAY PROPAGATION (CRITICAL):\n"
        "- If a BOX/BLOCK A has #dc-from(B) / #ec-from(B) / #po-from(B) / #far-from(B) / #behind(B) / #front(B),\n"
        "  then EVERY object inside A inherits that relation w.r.t. every object inside B (and w.r.t. B itself).\n"
        "- If an OBJECT has #dc/#ec/#po/#near/#far/#behind/#front, it applies ONLY to that object.\n"
        "  (Do NOT propagate object tags up to the box.)\n\n"

        "═══════════════════════════════════════════════════════\n"
        "CONTAINMENT RULES (TYPE 7/10 FIX)\n"
        "═══════════════════════════════════════════════════════\n"
        "- #inside-clear → object ntpp box (ONLY ntpp, never also tpp).\n"
        "- #touch-edge   → object tpp box  (ONLY tpp, never also ntpp).\n"
        "- An object has EXACTLY ONE containment relation to its direct container.\n"
        "  Never output all four {tpp, tppi, ntpp, ntppi} — pick the ONE that matches.\n\n"
        "TRANSITIVE CONTAINMENT (TYPE 10 FIX):\n"
        "- If object X is ntpp of box C, and box C is ntpp of box B, then X is ntpp of B.\n"
        "- If object X is tpp of box C, and box C is ntpp of box B, then X is ntpp of B.\n"
        "- Apply this chain when the question asks about an object relative to an outer container.\n\n"
        "Inverses (only when question is BOX relative to OBJECT):\n"
        "  if object ntpp box → box ntppi object.\n"
        "  if object tpp  box → box tppi  object.\n\n"

        "BOX vs OBJECT:\n"
        "- If one referent is a BOX/BLOCK and the other is an OBJECT:\n"
        "  * First check containment (tpp/ntpp/tppi/ntppi) — these take priority.\n"
        "  * If the object is INSIDE the box, output the correct containment relation.\n"
        "  * Use the box header Row/Col for directional comparison if needed.\n\n"

        "CATEGORY SETS + QUANTIFIERS:\n"
        "- If a referent is a category phrase, treat it as a SET of all matching entities in scope.\n"
        "- If quantifiers.all=true: required relation(s) must hold for ALL required matches.\n"
        "- If quantifiers.any=true: at least one match satisfies.\n"
        "- If quantifiers.other=true: exclude self-pairs.\n\n"

        "PROCEDURE (FOLLOW EXACTLY):\n"
        "Step 1) Read QUESTION_PLAN and restate: subject (A), target (B), scope, q_mode, quantifiers.\n"
        "Step 2) From INTERPRETATION TEXT, locate subject and target. Note their Row/Col and which box they are in.\n"
        "Step 3) Determine if subject and target are in the SAME box or DIFFERENT boxes.\n"
        "  - SAME BOX: compare their Row/Col directly for directional relations.\n"
        "  - DIFFERENT BOXES: compare their BOX HEADER Row/Col for directional relations.\n"
        "  - BOX vs OBJECT: check containment first, then use box header for direction if needed.\n"
        "Step 4) Evaluate ALL 15 relations one-by-one for the ordered pair (A, B):\n"
        "   1 left  2 right  3 above  4 below  5 behind  6 front\n"
        "   7 near  8 far    9 dc    10 ec    11 po\n"
        "  12 tpp  13 ntpp  14 tppi  15 ntppi\n"
        "   For each: cite specific Row/Col/tags. Apply CROSS-BOX DIRECTIONAL RULE if different boxes.\n"
        "Step 5) SANITY CHECK before outputting:\n"
        "   - Do I have both above AND below? → ERROR, keep only the correct one.\n"
        "   - Do I have both left AND right? → ERROR, keep only the correct one.\n"
        "   - Do I have both ntpp AND ntppi? → ERROR, keep only the correct one.\n"
        "   - Do I have both tpp AND tppi? → ERROR, keep only the correct one.\n"
        "Step 6) selected_option = all relations found TRUE that appear in Candidate answers.\n"
        "   If q_mode=YN: output ONLY [\"Yes\"] or [\"No\"].\n"
        "Step 7) Justification must include:\n"
        "- Scope used (GLOBAL vs IN-BOX)\n"
        "- Whether same-box or cross-box comparison (with box header Row/Col if cross-box)\n"
        "- Exact evidence (row/col/tags)\n"
        "- For each of the 15 relations: why TRUE or why NOT-TRUE\n"
        "- 1–2 key alternatives not selected\n\n"
        "INTERPRETATION TEXT:\n"
        f"{interp_text}\n\n"
        "QUESTION_PLAN JSON:\n"
        f"{question_plan_json}\n\n"
        "Candidate answers:\n"
        f"{cand_text}\n\n"
        "Return ONLY JSON:\n"
        "{\"justification\":\"...\",\"selected_option\":[...]}\n"
    )


# =========================================================
# YN-SPECIFIC PROMPTS (quantifier-aware, formal predicate)
# =========================================================

def build_prompt_question_plan_yn(question: str) -> str:
    """
    PROMPT 2 (YN-specific) — Parse a Yes/No spatial question into a formal
    logical predicate with per-referent quantifiers.
    Replaces build_prompt_question_interpretation for YN questions.
    """
    return (
        "You are given a YES/NO spatial reasoning question.\n\n"
        "TASK: Parse the question into a formal logical predicate.\n"
        "Do NOT answer the question. Do NOT use any grid or spatial data.\n\n"
        "OUTPUT — Return ONLY JSON with this exact schema:\n"
        "{\n"
        '  "subject":  {"description": "<noun phrase>", "quantifier": "exists|forall"},\n'
        '  "target":   {"description": "<noun phrase>", "quantifier": "exists|forall"},\n'
        '  "relation_raw": "<spatial phrase copied from the question>",\n'
        '  "relation_canonical": "<one of the canonical values below>",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": false,\n'
        '  "predicate": "<formal logic expression>"\n'
        "}\n\n"

        "══════════════════════════════════════\n"
        "STEP 1 — IDENTIFY SUBJECT AND TARGET\n"
        "══════════════════════════════════════\n"
        "YN spatial questions follow the pattern:\n"
        '  "{Is/Are/Does/Do} {Q1} {SUBJECT} {RELATION} {Q2} {TARGET}?"\n'
        "- SUBJECT = the first noun phrase (the entity being asked about).\n"
        "- TARGET  = the second noun phrase (the reference point).\n"
        "- Keep size + color + type words in the description.\n"
        "  These will later be matched against entity descriptions in the grid\n"
        "  (e.g., 'medium red triangle' from a line like `Row(2) Col(1) medium red triangle_in(box AAA)`).\n"
        "- Boxes/blocks are interchangeable.\n\n"
        "If the question has only ONE referent (e.g. 'Is any box within any box?'),\n"
        "use the SAME category for both subject and target.\n\n"

        "══════════════════════════════════════\n"
        "STEP 2 — HOW MANY GRID ENTITIES MUST MATCH\n"
        "══════════════════════════════════════\n"
        "Read the word before each noun phrase to decide scope:\n"
        "  'a' / 'an' / 'any' / 'some' / bare noun  →  \"exists\"  (at least one grid entity)\n"
        "  'all' / 'every' / 'each'                  →  \"forall\"  (every grid entity)\n"
        "  'the' / specific named entity              →  \"exists\"  (single entity)\n"
        "Subject and target each get their own value.\n\n"

        "══════════════════════════════════════\n"
        "STEP 3 — EXTRACT AND MAP THE RELATION\n"
        "══════════════════════════════════════\n"
        "Set relation_raw = the spatial phrase as it appears in the question.\n"
        "Map to relation_canonical using this table:\n\n"
        "  Directional (Row / Col comparison):\n"
        "    'north of' / 'above' / 'over'                    → above\n"
        "    'south of' / 'below' / 'under' / 'beneath'       → below\n"
        "    'east of'  / 'right of' / 'to the right of'      → right\n"
        "    'west of'  / 'left of'  / 'to the left of'       → left\n\n"
        "  Topological (tag-only evidence):\n"
        "    'in front of'                                     → front\n"
        "    'behind'                                          → behind\n"
        "    'near' / 'close to'                               → near\n"
        "    'far' / 'away from' / 'farther from'              → far\n"
        "    'disconnected from'                               → dc\n"
        "    'touch' / 'touching' / 'externally connected'     → ec\n"
        "    'overlap'                                         → po\n\n"
        "  Containment:\n"
        "    'inside' / 'within' / 'in' / 'covered by'        → inside\n"
        "    'inside and touching'                             → inside_touching\n"
        "    'contain' / 'have' / 'has' / 'cover' / 'covers'  → contains\n\n"

        "══════════════════════════════════════\n"
        "STEP 4 — SELF-PAIR DETECTION\n"
        "══════════════════════════════════════\n"
        "Set exclude_self_pairs = true when the subject and target\n"
        "refer to the SAME category or overlapping sets, for example:\n"
        "  'Is any X above all X?'               → same category\n"
        "  'Are all things inside all things?'    → same category\n"
        "  'Is any box within any box?'           → same category\n"
        "  'Is any blue object north of all blue squares?'\n"
        "      → blue object ⊇ blue squares, so they overlap → true\n"
        "Also set it when subject is clearly a member / subset of target\n"
        "(e.g. 'midsize orange rectangle' vs 'midsize orange rectangles').\n\n"

        "══════════════════════════════════════\n"
        "STEP 5 — BUILD FORMAL PREDICATE\n"
        "══════════════════════════════════════\n"
        "Combine the pieces:\n"
        "  quantifier(subject) x∈{S}: quantifier(target) y∈{T}: relation(x, y)\n"
        "If exclude_self_pairs=true, add 'y≠x' constraint.\n\n"
        "SCOPE:\n"
        "  Default: GLOBAL\n"
        "  If question explicitly says 'in box X' / 'inside block X' → IN-BOX\n\n"

        "══════════════════════════════════════\n"
        "WORKED EXAMPLES\n"
        "══════════════════════════════════════\n\n"

        "Q: 'Are all medium yellow apples covered by all boxes?'\n"
        "{\n"
        '  "subject":  {"description": "medium yellow apples", "quantifier": "forall"},\n'
        '  "target":   {"description": "boxes", "quantifier": "forall"},\n'
        '  "relation_raw": "covered by",\n'
        '  "relation_canonical": "inside",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": false,\n'
        '  "predicate": "∀x∈{medium yellow apples}: ∀y∈{boxes}: inside(x, y)"\n'
        "}\n\n"

        "Q: 'Is a red object north of all grey hexagons?'\n"
        "{\n"
        '  "subject":  {"description": "red object", "quantifier": "exists"},\n'
        '  "target":   {"description": "grey hexagons", "quantifier": "forall"},\n'
        '  "relation_raw": "north of",\n'
        '  "relation_canonical": "above",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": false,\n'
        '  "predicate": "∃x∈{red object}: ∀y∈{grey hexagons}: above(x, y)"\n'
        "}\n\n"

        "Q: 'Is any midsize orange rectangle below all midsize orange rectangles?'\n"
        "{\n"
        '  "subject":  {"description": "midsize orange rectangle", "quantifier": "exists"},\n'
        '  "target":   {"description": "midsize orange rectangles", "quantifier": "forall"},\n'
        '  "relation_raw": "below",\n'
        '  "relation_canonical": "below",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": true,\n'
        '  "predicate": "∃x∈{midsize orange rectangle}: ∀y∈{midsize orange rectangles}, y≠x: below(x, y)"\n'
        "}\n\n"

        "Q: 'Does any block cover a medium red hexagon?'\n"
        "{\n"
        '  "subject":  {"description": "block", "quantifier": "exists"},\n'
        '  "target":   {"description": "medium red hexagon", "quantifier": "exists"},\n'
        '  "relation_raw": "cover",\n'
        '  "relation_canonical": "contains",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": false,\n'
        '  "predicate": "∃x∈{block}: ∃y∈{medium red hexagon}: contains(x, y)"\n'
        "}\n\n"

        "Q: 'Is any thing north of all things?'\n"
        "{\n"
        '  "subject":  {"description": "thing", "quantifier": "exists"},\n'
        '  "target":   {"description": "things", "quantifier": "forall"},\n'
        '  "relation_raw": "north of",\n'
        '  "relation_canonical": "above",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": true,\n'
        '  "predicate": "∃x∈{thing}: ∀y∈{things}, y≠x: above(x, y)"\n'
        "}\n\n"

        "Q: 'Does box one touch all green watermelons?'\n"
        "{\n"
        '  "subject":  {"description": "box one", "quantifier": "exists"},\n'
        '  "target":   {"description": "green watermelons", "quantifier": "forall"},\n'
        '  "relation_raw": "touch",\n'
        '  "relation_canonical": "ec",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": false,\n'
        '  "predicate": "∃x∈{box one}: ∀y∈{green watermelons}: ec(x, y)"\n'
        "}\n\n"

        "Q: 'Is a block inside and touching LLL?'\n"
        "{\n"
        '  "subject":  {"description": "block", "quantifier": "exists"},\n'
        '  "target":   {"description": "LLL", "quantifier": "exists"},\n'
        '  "relation_raw": "inside and touching",\n'
        '  "relation_canonical": "inside_touching",\n'
        '  "scope": "GLOBAL",\n'
        '  "exclude_self_pairs": true,\n'
        '  "predicate": "∃x∈{block}: ∃y∈{LLL}, y≠x: inside_touching(x, y)"\n'
        "}\n\n"

        f"QUESTION:\n{question}\n\n"
        "Return ONLY the JSON. No other text.\n"
    )


def build_prompt_grid_answer_yn(
    interp_text: str,
    question_plan_json: str,
    question: str,
) -> str:
    """
    PROMPT 3 (YN-specific) — Evaluate a formal predicate from the YN question
    plan against the grid interpretation. Outputs Yes or No.
    Replaces build_prompt_grid_answer_from_interp_and_question_plan for YN.
    """
    return (
        "You are given:\n"
        "(1) INTERPRETATION TEXT — structured description of a spatial grid\n"
        "(2) QUESTION_PLAN JSON  — formal predicate for a Yes/No question\n"
        "(3) ORIGINAL QUESTION   — for reference only\n\n"
        "TASK: Evaluate the formal predicate and return Yes or No.\n\n"
        "OUTPUT — Return ONLY JSON:\n"
        '{"justification": "...", "selected_option": ["Yes"] or ["No"]}\n\n'

        "═══════════════════════════════════════════════════\n"
        "PROCEDURE — FOLLOW THESE STEPS EXACTLY\n"
        "═══════════════════════════════════════════════════\n\n"

        # ---- STEP 1 ----
        "STEP 1 — PARSE THE PLAN\n"
        "Read the question_plan JSON and extract:\n"
        "  • S_desc   = subject.description     (noun phrase)\n"
        "  • S_quant  = subject.quantifier       (exists | forall)\n"
        "  • T_desc   = target.description       (noun phrase)\n"
        "  • T_quant  = target.quantifier        (exists | forall)\n"
        "  • R        = relation_canonical       (the spatial relation to check)\n"
        "  • EXCL     = exclude_self_pairs       (true | false)\n"
        "Restate the predicate in one plain-English sentence.\n\n"

        # ---- STEP 2 ----
        "STEP 2 — ENUMERATE ENTITY SETS FROM THE GRID\n"
        "From INTERPRETATION TEXT, build:\n"
        "  S_entities = ALL entities whose name matches S_desc\n"
        "  T_entities = ALL entities whose name matches T_desc\n\n"
        "How entities appear in the grid interpretation:\n"
        "  Each entity comes from a grid line like:\n"
        "    Row(r) Col(c) <size> <color> <shape>_in(<container>)  [tags]\n"
        "  Example: Row(2) Col(1) medium red triangle_in(box AAA)  #touch-edge\n"
        "  The entity description is 'medium red triangle', it is inside 'box AAA',\n"
        "  located at Row(2) Col(1), and has tag #touch-edge.\n\n"
        "Matching rules (scan every entity line in the interpretation):\n"
        "  • Category match: case-insensitive substring against entity descriptions.\n"
        "    'medium thing' → any entity with 'medium' in its description.\n"
        "    'square' → any entity with 'square' in its description.\n"
        "  • 'things' / 'objects' / 'shapes' = all non-container entity lines.\n"
        "  • 'boxes' / 'blocks' = all containers (lines with `[box X:` markers).\n"
        "  • Specific named entity (e.g. 'box DDD') = that one entity.\n"
        "  • Scan every container globally, including nested ones.\n\n"
        "List every match with: name, container, Row, Col, tags.\n"
        "If no grid entity lines match a description:\n"
        "  forall → treat as satisfied (nothing to contradict).\n"
        "  exists → treat as not satisfied (no entity found).\n\n"

        # ---- STEP 3 ----
        "STEP 3 — EVALUATE EACH PAIR\n"
        "For every pair (s, t) where s ∈ S_entities, t ∈ T_entities:\n"
        "  1) If EXCL = true AND s is the SAME entity as t → SKIP this pair.\n"
        "  2) Check: does R(s, t) hold? Use the RELATION RULES below.\n"
        "  3) Record TRUE or FALSE with brief evidence.\n\n"

        # ---- STEP 4 ----
        "STEP 4 — COMBINE PAIR RESULTS\n"
        "Based on the plan, decide how grid entity pairs are scanned:\n"
        "  exists × exists: Yes if at least one (s,t) grid pair is TRUE.\n"
        "  exists × forall: Yes if some s satisfies R for every t in the grid.\n"
        "    (Only ONE such s needs to work — do not require all s's to pass.)\n"
        "    *** WARNING: You must verify that the SAME s works for EVERY t.\n"
        "        Do NOT find one s-t pair that works and stop.\n"
        "        For each candidate s, check R(s,t) for ALL t's. ***\n"
        "  forall × exists: Yes if every s in the grid has at least one t that satisfies R.\n"
        "  forall × forall: Yes if every (s,t) grid pair is TRUE.\n"
        "    (Check every grid entity — do not stop after the first pair.)\n"
        "When exclude_self_pairs=true, skip pairs where s and t are the same entity line.\n\n"
        "*** QUANTIFIER CROSS-CHECK (do this BEFORE outputting Yes/No): ***\n"
        "  If the plan says forall for EITHER side, you MUST have checked EVERY\n"
        "  entity on that side — list them all. If you checked fewer entities\n"
        "  than exist in the grid, your answer is WRONG. Go back and recount.\n\n"

        # ---- STEP 5 ----
        "STEP 5 — OUTPUT\n"
        "selected_option: [\"Yes\"] if predicate is satisfied, [\"No\"] otherwise.\n"
        "justification: list S_entities, T_entities, pair-by-pair results,\n"
        "               and the final verdict.\n\n"

        "═══════════════════════════════════════════════════\n"
        "RELATION RULES\n"
        "═══════════════════════════════════════════════════\n\n"

        "A) DIRECTIONAL — above / below / left / right\n"
        "   *** DIRECTIONAL IS NEVER TAG-ONLY — ALWAYS compare Row/Col numbers. ***\n"
        "   Determined by comparing Row and Col numbers from interpretation.\n"
        "   • above(s, t): Row(s) < Row(t)   (smaller row = higher position)\n"
        "   • below(s, t): Row(s) > Row(t)\n"
        "   • left(s, t):  Col(s) < Col(t)   (smaller col = more left)\n"
        "   • right(s, t): Col(s) > Col(t)\n"
        "   Rules:\n"
        "   • CAN compare across different boxes — use the grid Row/Col values.\n"
        "     Row(3) in box A vs Row(7) in box B → above(A_obj, B_obj) = TRUE.\n"
        "   • Even if two entities are in DIFFERENT boxes with no explicit tags\n"
        "     between them, Row/Col comparison is STILL valid for above/below/left/right.\n"
        "   • Same Row → above/below = FALSE.\n"
        "   • Same Col → left/right = FALSE.\n"
        "   • An object INSIDE a container is NOT 'above/below/left/right of'\n"
        "     that container because it is CONTAINED, not spatially beside it.\n"
        "   • Do NOT say 'no explicit above/below tag exists' for directional —\n"
        "     there ARE no tags for directional; you MUST use Row/Col numbers.\n\n"

        "B) TOPOLOGICAL (TAG-ONLY) — dc / ec / po / near / far / front / behind\n"
        "   TRUE only if EXPLICIT TAG EVIDENCE exists in the interpretation.\n"
        "   No tag → FALSE.  NEVER infer from row/col distance or different boxes.\n\n"
        "   Where to look for evidence:\n"
        "   1. OBJECT-LEVEL TAG:  entity line says '#dc(target)' / '#ec(target)' etc.\n"
        "   2. BOX-LEVEL PROPAGATION:\n"
        "      If container A has #dc(container B) at the box level, then:\n"
        "        • A  is dc from  B                              (box ↔ box)\n"
        "        • A  is dc from  every object in B              (box → obj)\n"
        "        • every object in A  is dc from  B              (obj → box)\n"
        "        • every object in A  is dc from  every object in B (obj → obj)\n"
        "      Same propagation for ec / po / near / far / front / behind.\n"
        "   3. CRITICAL — SAME-CONTAINER EXCEPTIONS:\n"
        "      ✗  A container is NEVER dc from its own contents.\n"
        "         (it is tppi / ntppi of them — the opposite of dc)\n"
        "      ✗  ec ('touch') means EXTERNAL contact between separate bodies.\n"
        "         #touch-edge on an object inside a box = tpp (not ec to the box).\n"
        "         A box 'touching' its own contents is containment (tppi), NOT ec.\n"
        "      ✗  po ('partial overlap') ≠ containment.\n"
        "         An object fully inside a box is tpp/ntpp, NOT po.\n\n"

        "C) CONTAINMENT — inside / inside_touching / contains\n"
        "   inside(s, t):   s is spatially inside t.\n"
        "     Evidence: s appears as '<name>_in(t)' in interpretation.\n"
        "       #inside-clear → ntpp(s,t)    (inside, not touching boundary)\n"
        "       #touch-edge   → tpp(s,t)     (inside, touching boundary)\n"
        "     Both count as inside(s,t) = TRUE.\n\n"
        "   inside_touching(s, t):   s is inside t AND touches t's boundary.\n"
        "     Evidence: s is _in(t) AND has #touch-edge   → TRUE.\n"
        "     If s is _in(t) with #inside-clear            → FALSE.\n\n"
        "   contains(s, t):   t is inside s (inverse of 'inside').\n"
        "     Evidence: t appears as '<name>_in(s)' in interpretation.\n\n"
        "   TRANSITIVE NESTING:\n"
        "     If s is _in(A) and A is _in(t), then inside(s, t) = TRUE.\n"
        "     If t is _in(A) and A is _in(s), then contains(s, t) = TRUE.\n"
        "     This applies to any depth of nesting.\n\n"
        "   DIRECTION OF 'covered by' vs 'covers':\n"
        "     'X covered by Y' → inside(X, Y)   (X is the one inside Y)\n"
        "     'Y covers X'     → contains(Y, X)  (Y is the container)\n"
        "   SELF-CONTAINMENT: An entity is NEVER inside itself.\n"
        "     inside(x, x) = FALSE.  contains(x, x) = FALSE.\n\n"

        "═══════════════════════════════════════════════════\n"
        "COMMON PITFALLS — READ BEFORE ANSWERING\n"
        "═══════════════════════════════════════════════════\n\n"

        "P1  SELF-REFERENTIAL / REFLEXIVE:\n"
        "    A grid entity cannot be above/below/left/right/inside itself.\n"
        "    When exclude_self_pairs = true, skip (x, x) pairs.\n"
        "    Even when exclude_self_pairs = false, no entity satisfies\n"
        "    a directional or containment relation with itself.\n\n"

        "P2  CONTAINER ≠ DISCONNECTED FROM CONTENTS:\n"
        "    A box is NEVER dc from objects inside it (it is tppi/ntppi).\n"
        "    Objects inside a box are NEVER dc from that box.\n"
        "    'Is box X dc from all objects?' → No, because box X tppi its own objects.\n\n"

        "P3  CONTAINMENT ≠ DIRECTIONAL:\n"
        "    Objects INSIDE a box are not 'above' or 'east of' that box.\n"
        "    They are contained by it. These are fundamentally different relations.\n"
        "    Directional comparisons are between spatial peers, not container↔content.\n\n"

        "P4  TOUCH (ec) ≠ CONTAINMENT (tpp):\n"
        "    #touch-edge means the object touches the INNER boundary of its container.\n"
        "    This is tpp (contained + touching), NOT ec (external contact).\n"
        "    A box's ec with another entity requires an explicit #ec tag.\n\n"

        "P5  OVERLAP (po) ≠ CONTAINMENT (tpp/ntpp):\n"
        "    po is partial overlap, where neither is fully inside the other.\n"
        "    If object is fully in a box, that is tpp/ntpp, NOT po.\n\n"

        "P6  RELATION DIRECTION — DON'T INVERT:\n"
        "    'X covered by Y' means X inside Y → check inside(X, Y).\n"
        "    'Y covers X' means Y contains X → check contains(Y, X).\n"
        "    Do NOT swap the direction.\n\n"

        "P7  CROSS-BOX POSITIONAL COMPARISON:\n"
        "    Row/Col values from the grid ARE valid across different boxes.\n"
        "    'Entity at Row(3) in box X' IS above 'entity at Row(7) in box Y'.\n"
        "    Do NOT refuse cross-box Row/Col comparisons for directional relations.\n\n"

        "P8  NESTED CONTAINERS (TRANSITIVE CONTAINMENT):\n"
        "    If box A is _in(box B), then ALL objects in A are also inside B.\n"
        "    'Does block B contain object Z?' → Yes, if Z is in A and A is in B.\n\n"

        "P9  'WITHIN ANY BLOCK' = ∃block:∀shapes (NOT ∀shapes:∃block):\n"
        "    'Are all X within any block?' asks: does there EXIST a single block\n"
        "    that contains ALL X's?  It does NOT ask whether each X is in SOME block.\n"
        "    Example: X1 in box A, X2 in box B → No single box has both → No.\n"
        "    But if X1 and X2 are both in box A → Yes.\n\n"

        "P10 DIRECTIONAL IS NEVER TAG-ONLY:\n"
        "    above/below/left/right are determined PURELY by Row/Col comparison.\n"
        "    Do NOT look for #above or #below tags — they do not exist.\n"
        "    Do NOT say 'no directional tag found.' Just compare Row/Col numbers.\n"
        "    This works even across different boxes.\n\n"

        "═══════════════════════════════════════════════════\n\n"

        "INTERPRETATION TEXT:\n"
        f"{interp_text}\n\n"

        "QUESTION_PLAN JSON:\n"
        f"{question_plan_json}\n\n"

        "ORIGINAL QUESTION:\n"
        f"{question}\n\n"

        "Return ONLY JSON:\n"
        '{"justification": "...", "selected_option": ["Yes"] or ["No"]}\n'
    )


# =========================================================
# (Optional) Legacy Prompt 2 — Interpretation + raw question
# Keep but align to same quantifier + tag-only rules
# =========================================================
def build_prompt_grid_answer_from_interpretation(
    interp_text: str,
    question: str,
    candidate_answers: List[str],
) -> str:
    """
    PROMPT 2 (legacy) — Interpretation + Question → Final answer (JSON only)
    """
    cand_text = ", ".join(candidate_answers) if candidate_answers else "Yes, No"
    return (
        "You are given:\n"
        "(1) INTERPRETATION of the grid in simple sentences (from Prompt 1)\n"
        "(2) the QUESTION\n"
        "(3) Candidate answers\n\n"
        "Your task:\n"
        "- Understand the question wording carefully.\n"
        "- Apply the rules below.\n"
        "- Output ONLY JSON:\n"
        "{\"justification\":\"...\",\"selected_option\":[...]}\n"
        "selected_option MUST be a JSON array of strings.\n\n"
        "CORE PRINCIPLE:\n"
        "- Evaluate ALL 15 relations one-by-one and justify TRUE vs NOT-TRUE for each.\n"
        "- Return ALL relations that are TRUE AND appear in the candidate list.\n\n"
        "THE 15 RELATIONS (MUST CHECK ALL):\n"
        "left, right, above, below, behind, front, near, far, dc, ec, po, tpp, ntpp, tppi, ntppi\n\n"
        "ROW/COL RULES:\n"
        "- Smaller Row(k) means higher position: Row(3) is ABOVE Row(4).\n"
        "- Smaller Col(k) means more left: Col(1) is LEFT of Col(2).\n\n"
        "TAG-ONLY POLICY (CRITICAL):\n"
        "- Only output dc/ec/po/near/far/behind/front if they are explicitly present in INTERPRETATION TEXT as tags/statements.\n"
        "- Do NOT compute near/far from row-gap.\n"
        "- Do NOT infer dc/ec/po from distance, spacing, or different boxes.\n"
        "- #touch-edge is ONLY containment (tpp), NOT ec.\n\n"
        "TOPOLOGY PROPAGATION (ONE-WAY):\n"
        "- If a BOX has #dc-from(BoxX) / #ec-from(BoxX) / #po-from(BoxX) / #near-to(BoxX) / #far-from(BoxX),\n"
        "  every object inside inherits that relation to BoxX.\n"
        "- If an OBJECT has #dc/#ec/#po/#near/#far/#behind/#front, it applies only to that object.\n\n"
        "CONTAINMENT RULES:\n"
        "- #inside-clear => object ntpp box.\n"
        "- #touch-edge  => object tpp box.\n"
        "- Inverses: object ntpp box => box ntppi object; object tpp box => box tppi object.\n"
        "- Nesting is transitive.\n\n"
        "QUANTIFIERS + CATEGORY MATCHING:\n"
        "- If question mentions a category (e.g., \"medium thing\", \"square\", \"apple\"), treat it as a SET in scope.\n"
        "- any/a/an/some => existential; all/each/every => universal; other => exclude self.\n\n"
        "NUMBERING NORMALIZATION:\n"
        "- If question refers to \"X number one\" but interpretation only lists \"X\", treat them as the SAME.\n"
        "- If question refers to \"X\" without a number, treat it as \"X number one\".\n\n"
        "DIRECTIONAL SANITY:\n"
        "- Do NOT output both above and below for the SAME ordered pair.\n"
        "- Do NOT output both left and right for the SAME ordered pair.\n"
        "- \"X regarding Y\" means compute relation(X, Y).\n\n"
        "PROCEDURE:\n"
        "Step 1) Copy the question EXACTLY and identify: referents, q_mode (FR/YN), scope, quantifiers.\n"
        "Step 2) Locate referents in INTERPRETATION TEXT (category sets allowed).\n"
        "Step 3) Evaluate ALL 15 relations one-by-one with evidence lines.\n"
        "Step 4) Candidate filter (and YN reduction).\n"
        "Step 5) Justification: scope + evidence + TRUE/NOT-TRUE for each relation + key alternatives not selected.\n\n"
        "INTERPRETATION TEXT:\n"
        f"{interp_text}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Candidate answers:\n"
        f"{cand_text}\n\n"
        "Return ONLY JSON:\n"
        "{\"justification\":\"...\",\"selected_option\":[...]}\n"
    )


# -----------------------------
# Small-grid wrappers (optional but recommended)
# -----------------------------

def build_prompt_grid_small_interpretation(pruned_grid: str) -> str:
    preface = (
        "IMPORTANT CONTEXT (small pruned grid):\n"
        "- The grid below is a PRUNED SUBGRID containing only the entities relevant to the question.\n"
        "- Row/Col indices have been RENAMED to run from 1..N in this pruned view.\n"
        "- Treat Row(1), Col(1) as the top-left of the PRUNED view.\n"
        "- Do NOT assume missing entities exist; reason ONLY over what is present here.\n\n"
        "READING ENTITY LINES IN THIS GRID:\n"
        "- Each entity line has the format:\n"
        "    Row(r) Col(c) <description>_in(<container>)  [optional tags]\n"
        "  Example: Row(2) Col(1) medium red triangle_in(box AAA)  #touch-edge\n"
        "  This means one entity called 'medium red triangle' is inside 'box AAA'\n"
        "  at grid position Row(2), Col(1).\n"
        "- Container boundaries are marked by `[box X:` (opening) and `]` (closing).\n"
        "  All entity lines indented between these markers belong to that container.\n"
        "- The entities listed in this grid are the COMPLETE set for interpretation.\n"
        "  Walk through every Row/Col line and list each entity with its position,\n"
        "  container, and tags — do not skip any.\n"
        "- Since this is a pruned subgrid, the scope for any reasoning covers\n"
        "  ONLY the entities present in this grid.\n\n"
    )
    return preface + build_prompt_grid_interpretation(pruned_grid)


def build_prompt_grid_small_answer(interp_text: str, question: str, candidate_answers: Optional[List[str]]) -> str:
    return build_prompt_grid_answer_from_interpretation(
        interp_text=interp_text,
        question=question,
        candidate_answers=candidate_answers or [],
    )



# -----------------------------
# Entity selection prompt (no relations block)
# -----------------------------

_COREF_WORDS = {
    "this", "that", "it", "one", "something", "anything", "everything",
    "thing", "object", "shape",
}

def _canon(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def build_entity_selection_prompt_no_relblock(question: str, universe_entities: List[str]) -> str:
    U = "\n".join(f"- {e}" for e in universe_entities)
    return f"""You are an information extraction system.

TASK:
Given a QUESTION and a list of ENTITIES (the universe), return ALL entities
the QUESTION could possibly refer to — both the SUBJECT and the TARGET side.
You must handle quantifiers correctly (all/any/a/the, each, every, no, none, both, either).
You MUST NOT output entities that are not in the universe.

ENTITY UNIVERSE (only choose from these exact strings):
{U}

QUESTION:
{question}

══════════════════════════════════════
CRITICAL RULE — CATEGORY / GENERIC REFERENTS
══════════════════════════════════════
A question typically has TWO referent phrases: a SUBJECT and a TARGET.
Example: "Is medium orange apple number two above a medium thing?"
  → SUBJECT = "medium orange apple number two"
  → TARGET  = "a medium thing" (this is a CATEGORY referent!)

BOTH sides MUST produce selected entities. NEVER drop the target side.

A CATEGORY referent is any phrase that does NOT name one specific entity but
instead describes a CLASS of entities. Common patterns:
  • "a medium thing" / "any shape" / "all objects" / "all things"
  • "a/an/any/all + adjective + generic noun" (thing/object/shape/item)
  • "a/an/any/all + color/size + specific type" ("any red hexagon", "all squares")
  • "boxes" / "blocks" / "a block" / "any box" / "all blocks"

When you encounter a category referent:
  1. IDENTIFY the adjective filters (size, color, type) in the phrase.
  2. SELECT ALL universe entities whose name contains those adjectives
     (case-insensitive substring match).
  3. If the category is fully generic ("thing"/"object"/"shape"/"item" with no
     adjective filter), select ALL non-box entities in the universe.
  4. If the category is "box"/"block"/"boxes"/"blocks", select ALL box/block
     entities in the universe (any entity containing "box" or "block").

*** NEVER put a category phrase into ignored_spans. ***
*** NEVER return only one side of the question. ***

══════════════════════════════════════
RULES (follow strictly)
══════════════════════════════════════

1. Output only entities from the universe. Do not paraphrase entity names.

2. CATEGORY MATCHING (substring, case-insensitive):
   * "medium thing"  → select every entity whose name contains "medium"
   * "any shape"     → select ALL non-box entities (every shape in the universe)
   * "all objects"   → select ALL non-box entities
   * "red hexagon"   → select every entity whose name contains "red" AND "hexagon"
   * "square"        → select every entity whose name contains "square"
   * "a block" / "any box" / "all blocks" → select ALL box/block entities

3. QUANTIFIER semantics for entity selection (select generously):
   * "a/an <X>" or "any <X>": select ALL entities matching X.
   * "the <entity>": select that specific entity if present.
   * "all <X>" / "every <X>": select ALL entities matching X.
   * "all X in all boxes": select ALL X entities AND ALL box entities.

4. Always include every EXPLICITLY named entity in the question
   (case-insensitive exact-string match against the universe).

5. Coreference words ("this", "that", "it", "one") are ignored
   ONLY when they do not match any universe entity.
   If a universe entity is literally "grey thing" or "medium grey thing",
   it IS a valid selection.

6. BOXES / BLOCKS:
   * If the question mentions "block" / "box" / "boxes" / "blocks" as a
     referent (subject OR target), select ALL box/block entities in the
     universe — even if they appear as "a block" or "any box."
   * Also select all box/block entities when the question says
     "within any block", "in all boxes", "covered by all blocks", etc.

7. ERROR PREVENTION — the following must NEVER appear in ignored_spans:
   * Category phrases that describe a set of entities
     ("medium thing", "any shape", "all objects", "a block", etc.)
   * Adjective + generic-noun combinations
   Only TRUE coreference filler words ("this", "that", "it") may be ignored,
   and only when they don't match a universe entity.

8. If nothing matches, return an empty list.

OUTPUT JSON ONLY (no extra text):
{{
  "selected_entities": ["..."],
  "quantifiers": {{
    "has_any_or_a": true/false,
    "has_all": true/false,
    "has_all_boxes_scope": true/false
  }},
  "ignored_spans": ["..."],
  "selection_rationale": "explain your reasoning here"
}}
"""
