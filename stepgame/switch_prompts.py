"""
stepgame/switch_prompts.py
==========================
Pure prompt-builder functions for the StepGame switch pipeline.
No LLM calls live here — only string construction.

All constants (REL_OPTIONS_*, SEMANTICS_BLOCK, INVERSE_REL, DIAGONAL_SET) are
defined here and re-exported so that stepgame/switch.py can import them from
this single place.
"""

import re
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

REL_OPTIONS_NO_DK = [
    "left", "right", "above", "below",
    "upper-left", "upper-right", "lower-left", "lower-right",
    "overlap",
]
REL_OPTIONS_WITH_DK = REL_OPTIONS_NO_DK + ["dont know"]
REL_OPTIONS = REL_OPTIONS_WITH_DK[:]

INVERSE_REL = {
    "left": "right", "right": "left",
    "above": "below", "below": "above",
    "upper-left": "lower-right", "lower-right": "upper-left",
    "upper-right": "lower-left", "lower-left": "upper-right",
    "overlap": "overlap",
    "dont know": "dont know",
}

DIAGONAL_SET = {"upper-left", "upper-right", "lower-left", "lower-right"}

SEMANTICS_BLOCK = """
Frame pin:
- "relation of X to Y" means X relative to Y (stand at Y and locate X).

Semantics of Relation(X,Y) (use X and Y as variables):
upper-left:  X is above and to the left of Y.
upper-right: X is above and to the right of Y.
lower-left:  X is below and to the left of Y.
lower-right: X is below and to the right of Y.
above:       X is directly above Y.
below:       X is directly below Y.
left:        X is directly left of Y.
right:       X is directly right of Y.
overlap:     X and Y occupy the same position.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2) HARD-LANGUAGE SCORING PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def prompt_hard_language_scores(text: str) -> str:
    return f"""
Analyze the TEXT below and extract ONLY phrases that make spatial reasoning
genuinely difficult for a small model (Qwen-3-32B).

IMPORTANT: Be VERY strict. Most spatial language is easy. Only flag things
that would actually confuse a 32B model.

== WHAT IS HARD (flag these) ==
1. CLOCK POSITIONS: "4 o'clock", "between 4 and 5 o'clock", "3:00 position",
   "at the 9 position of a clock face". These require mapping clock angles
   to grid directions, which small models fail at.
   difficulty: 0.70 - 0.95

2. COMPOSITIONAL / DIAGONAL DIRECTIONS that combine two axes:
   "lower left", "upper right", "northwest", "southeast",
   "diagonally to the upper left", "bottom right corner",
   "slightly off center to the top left".
   difficulty: 0.50 - 0.80

3. GENUINELY AMBIGUOUS multi-clause sentences where a small model could
   mis-parse which entity relates to which:
   "A and B are next to each other with B on the left and C on the right"
   "X and Y are parallel, and X is on top of Y"
   "D is slightly off center to the top left and K is slightly off center
    to the bottom right" (two composites in one sentence)
   difficulty: 0.50 - 0.80

4. SPATIALLY VAGUE RELATIONAL WORDS that sound simple but are ambiguous
   about the actual grid direction:
   - "parallel" -- does not specify axis or direction. "P and E are
     parallel, and P is over E" is confusing because "parallel" adds
     no clear spatial info yet the model must still resolve the layout.
   - "in a horizontal/vertical line with X on the right/left" -- the
     "in a line" part is vague and combined with a direction clause
     it creates parsing ambiguity for a small model.
   - "next to each other with P at the bottom B on the top" -- combines
     adjacency with vertical ordering in one clause.
   - "on the same vertical/horizontal plane" -- vague about exact
     position. Model may incorrectly assume co-location.
   - "side by side" -- does not specify left/right ordering.
   difficulty: 0.40 - 0.65

5. FILLER / DISTRACTOR phrases that add no spatial information but
   can mislead a small model into over-interpreting or mis-parsing:
   - "with a small gap between them" -- adds adjacency noise.
   - "over there", "both there" -- vague co-presence, model may try
     to infer a direction from it.
   - "on top of" (when combined with other clauses in the same sentence,
     e.g. "U is on top of G" in a multi-sentence support block).
   - "next to each other with P at the bottom B on the top" -- adjacency
     plus ordering, the model must parse carefully.
   difficulty: 0.20 - 0.40

6. UNUSUAL VERB FORMS for spatial relations:
   - "presents right to Y" -- unusual for "is to the right of".
   - "sitting in the left direction of D" -- awkward phrasing.
   - "on the left/right side to B" -- nonstandard preposition.
   - "F and G are vertical and F is above G" -- "are vertical" is unusual.
   - "placed in the right direction of S" -- odd phrasing.
   difficulty: 0.35 - 0.55

== WHAT IS EASY (DO NOT flag these) ==
- Simple cardinal directions: "above", "below", "left of", "right of",
  "directly above", "directly below", "positioned below", "on the right"
  These are TRIVIAL. A 32B model handles these perfectly. difficulty = 0.
- "X is above Y" style simple statements -- NOT hard.
- "to the left of it" -- simple reference. NOT hard.
- 45-degree angle descriptions like "diagonally below X to the right at
  a 45 degree angle" -- these are verbose but clearly mean "lower-right".
  Only flag if the 45-degree phrasing combines with another confusing
  element (e.g. "at a 45 degree angle to K, in the lower lefthand corner"
  -- the "corner" part makes it compositional, so flag that one).

Return ONLY JSON:
{{
  "items": [
    {{"span":"<exact phrase from text>", "type":"clock|direction|ambiguity", "difficulty": 0.0}}
  ]
}}

Rules:
- difficulty MUST be a float in [0,1]
- MOST texts should return 0-2 items. If you find more than 3, you are
  being too generous. Re-check each one against the "EASY" list.
- If no hard phrases exist, return {{ "items": [] }}.
- Do NOT flag simple directions (above, below, left, right, etc.)
- Do NOT invent phrases not in the text.

TEXT:
<<<TEXT>>>
{text}
<<<END TEXT>>>
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 3) Q1 / ANSWER PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_q1_baseline_and_support_sentences(story: str, question: str) -> str:
    options_list = ", ".join(REL_OPTIONS_NO_DK)
    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<think carefully and give your reasoning based on the sentences you think are important to answer the question in story (support_sentences)>",
  "answer": "<one of: {options_list}>",
  "support_sentences": ["<verbatim STORY sentence>", "..."]
}}

Rules:
- USE ONLY the STORY below.
- support_sentences MUST be copied verbatim as full sentences from STORY.
- Choose the MINIMAL set of sentences sufficient to answer.

{SEMANTICS_BLOCK}

STORY:
<<<STORY>>>
{story}
<<<END STORY>>>

QUESTION:
{question}
""".strip()


def prompt_answer_only_allow_dontknow(text_block: str, question: str) -> str:
    options_list = ", ".join(REL_OPTIONS_WITH_DK)
    return f"""
Return EXACTLY this JSON:
{{
  
  "justification": "<think carefully and give your reasoning based on the Support sentences given>",
  "answer": "<one of: {options_list}>"
}}

Rules:
- USE ONLY the given text.
- Dont need to have very specific answers, sometimes vague words like over there, at the center could be used ignore them. Focus on the main relation.
- If insufficient to determine a unique answer, return "dont know".
- Do NOT guess.

{SEMANTICS_BLOCK}

SUPPORT SENTENCES:
<<<TEXT>>>
{text_block}
<<<END TEXT>>>

QUESTION:
{question}
""".strip()


def prompt_q2_answer_from_text(text_block: str, question: str) -> str:
    opts = ", ".join(REL_OPTIONS_WITH_DK)
    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<reasoning using ONLY the text>",
   "answer": "<one of: {opts}>"
}}

Rules:
- USE ONLY the TEXT.
- Dont need to have very specific answers, sometimes vague words like over there, at the center could be used ignore them. Focus on the main relation.
- If insufficient, return "dont know".
- Do NOT guess.

{SEMANTICS_BLOCK}

TEXT:
<<<TEXT>>>
{text_block}
<<<END TEXT>>>

QUESTION:
{question}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4) UNIFIED RELATIONS + VARIANTS PROMPT (Q2 GPT)
# ─────────────────────────────────────────────────────────────────────────────

def prompt_q2_unified_relations_and_variants(
    support_sentences: List[str],
    question: str,
) -> str:
    support_block = "\n".join(f"- {s}" for s in support_sentences)

    return f"""
You are given SUPPORT SENTENCES and a QUESTION.

Your task:
A) Convert each support sentence into atomic relations, but ONLY if explicitly stated.
B) Create 3 paraphrase levels (simple, hinted, canonical).
   - Simple: simple paraphrase, similar difficulty as original.
   - Hinted: add a small hint sentence to interpret the question better.
   - Canonical: rewrite support as normalized relations (left/right/above/below/diagonals/overlap),
                and ask the question explicitly.
C) Create a flip-consistency test question whose PURPOSE is to force using the INVERSE of a stated support relation.

SEMANTICS / NORMALIZATION (USE ONLY IF EXPLICITLY STATED):
You may see relations described using compass directions or clock directions. If a support sentence explicitly uses such a term, normalize it to one of the allowed relations.

Allowed normalized relations:
left, right, above, below, upper-left, upper-right, lower-left, lower-right, overlap

Compass direction normalization (explicit compass words only):
- north / up / upper / top            -> above
- south / down / lower / bottom       -> below
- west / left                         -> left
- east / right                        -> right
- north-west / northwest / NW         -> upper-left  (north=above + west=left)
- north-east / northeast / NE         -> upper-right (north=above + east=right)
- south-west / southwest / SW         -> lower-left  (south=below + west=left)
- south-east / southeast / SE         -> lower-right (south=below + east=right)

Clock direction normalization (treat object X "at <clock> of Y" as a direction from Y to X):
- 12 o'clock  -> above
- 1 o'clock   -> upper-right
- 2 o'clock   -> upper-right
- 3 o'clock   -> right
- 4 o'clock   -> lower-right
- 5 o'clock   -> lower-right
- 6 o'clock   -> below
- 7 o'clock   -> lower-left
- 8 o'clock   -> lower-left
- 9 o'clock   -> left
- 10 o'clock  -> upper-left
- 11 o'clock  -> upper-left

Notes / constraints:
- Only normalize compass/clock terms if they appear explicitly in the support sentence.
- Do NOT infer compass/clock terms from other phrasing unless explicitly stated.
- If a sentence says "X is to the northwest of Y", emit: (X, upper-left, Y).
- If a sentence says "X is at 2 o'clock of Y", emit: (X, upper-right, Y).
- If a sentence says "overlapping", "on top of", "covers", "intersecting", etc., only map to overlap if the sentence explicitly indicates overlap (not mere adjacency).

Return ONLY JSON in this schema:

{{
  "used_order_fallback": false,
  "relations_direct": [
    {{"head":"G","relation":"right","tail":"D","source_sentence":"<verbatim support sentence>"}}
  ],
  "variants": {{
    "simple":   {{"support_text":"...", "question":"..."}},
    "hinted":   {{"support_text":"...", "question":"..."}},
    "canonical":{{
      "support_text":"<ONLY normalized direct relations, one per line: X is <relation> of Y>",
      "question":"<explicit relation query>"
    }}
  }},
  "flip_consistency": {{
    "flipped_question": "<a question that forces inverse of a support-stated relation OR the original question if already inverse / not possible>",
    "flip_applied": true,
    "reference_relation": {{"head":"E","relation":"left","tail":"G"}},
    "reason": "<short explanation of whether/why flip applied>"
  }}
}}

RELATION EXTRACTION RULES (CRITICAL):

PRIMARY RULE (DIRECT-ONLY):
- Add a relation ONLY if explicitly stated in a support sentence.
- DO NOT add inverse relations.
- DO NOT add transitive / inferred relations.
- head/tail must be single capital letters A-Z.
- relation must be exactly one of:
  left, right, above, below, upper-left, upper-right, lower-left, lower-right, overlap
- source_sentence MUST be copied verbatim from SUPPORT SENTENCES.

SECONDARY RULE (ORDER-HEURISTIC FALLBACK; ONLY IF PRIMARY PRODUCES ZERO RELATIONS):
- If and ONLY IF the PRIMARY RULE yields relations_direct = [] (empty),
  then you MUST generate exactly ONE fallback relation using ENTITY ORDER IN THE SUPPORT SENTENCES.
- Procedure:
  1) Scan SUPPORT SENTENCES top-to-bottom.
  2) In the first sentence that contains at least TWO single-letter entities A-Z, take the FIRST appearing entity as E1
     and the SECOND appearing entity as E2 (appearance order, not alphabetical).
  3) Output exactly ONE relation: (E1, right, E2).
     - This is a deterministic fallback when no explicit binary relation is available.
  4) Set source_sentence to that sentence verbatim.
- If no support sentence contains at least two entities, keep relations_direct empty.

ADDITIONAL OUTPUT REQUIREMENT:
- Add a boolean field "used_order_fallback" at the top level of the JSON.
  - used_order_fallback=true iff you used the SECONDARY RULE.
  - otherwise false.


VARIANT RULES:
- Do NOT add facts.
- canonical.support_text MUST contain ONLY normalized direct relations (no prose).

FLIP-CONSISTENCY RULE (CRITICAL):
Goal: Given an explicit support relation (H, r, T) meaning "H is r of T",
construct a question that asks for the relation of T to H. This requires the solver to output INVERSE(r).

Steps:
1) Try to detect the two entities mentioned in the QUESTION (two single capital letters).
2) Choose a reference relation from relations_direct:
   - Prefer a relation that involves BOTH question entities (in any order).
   - Else, if not found, you may pick the FIRST relation in relations_direct.
   - If relations_direct is empty, set flip_applied=false and keep flipped_question=original question.
3) Let reference relation be (H, r, T). Define the target inverse-question pair as (T to H).
4) If the original QUESTION ALREADY asks for the relation of T to H (i.e., it is already the inverse-direction question),
   then DO NOT CHANGE IT: set flipped_question = original QUESTION and flip_applied=false.
5) Otherwise, rewrite a flipped_question that asks for the relation of T to H.
   - Preserve the STYLE of the original question (e.g., keep "agent", "object", etc. if present).
   - Do NOT add new facts; it's only a question rewrite.
   - Set flip_applied=true.

{SEMANTICS_BLOCK}

SUPPORT SENTENCES:
<<<SUPPORT>>>
{support_block}
<<<END SUPPORT>>>

QUESTION:
<<<QUESTION>>>
{question}
<<<END QUESTION>>>
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5) Q2 FLIP / BASELINE PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def flip_question_once(q: str) -> str:
    q = (q or "").strip()
    m = re.search(r"(relation of)\s+([A-Z])\s+(to)\s+([A-Z])", q)
    if m:
        prefix = q[:m.start()]
        suffix = q[m.end():]
        return f"{prefix}{m.group(1)} {m.group(4)} {m.group(3)} {m.group(2)}{suffix}"
    m2 = re.search(r"([A-Z])\s+relative to\s+([A-Z])", q)
    if m2:
        return q[:m2.start()] + f"{m2.group(2)} relative to {m2.group(1)}" + q[m2.end():]
    return q


def build_symbolic_QA_prompt(
    story_text: str,
    question_text: str,
    candidate_answers: List[str],
) -> str:
    options_list = ", ".join(candidate_answers)
    options_json = __import__("json").dumps(candidate_answers, ensure_ascii=False)
    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<think carefully and reason among the relations and story text to answer the given question>",
  "selected_option": "<one of: {options_list}>"
}}

Source of truth:
- USE ONLY the STORY below to answer.

STORY:
<<<STORY>>>
{story_text}
<<<END STORY>>>

Question:
{question_text}

{SEMANTICS_BLOCK}

How to compute BEFORE you output (brief internal steps, do not include in JSON):
1) Build row/col constraints from each relation edge using the text above.
2) Use relational reasoning (transitive closure / chaining) to derive X vs Y.
3) Map to exactly one option in {options_json}.
4) Output ONLY the JSON with "selected_option" and "justification".
""".strip()
