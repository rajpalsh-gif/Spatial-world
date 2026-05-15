"""
spartun/switch_prompts.py
=========================
Pure prompt-builder functions for the SPaRTUN switch pipeline.
No LLM calls live here — only string construction.

SEMANTICS_BLOCK is defined here and re-exported so that spartun/switch.py can
import it from a single place.
"""

import re
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SEMANTICS_BLOCK = r"""
Relations (X relative to Y):
- left/right/above/below/front/behind: directions.
- near/far: proximity.
- dc: disconnected. ec: touching. po: partial overlap.
- ntpp: X strictly inside Y (no boundary contact). tpp: X inside Y AND touching boundary.
- ntppi: X contains Y (no boundary contact). tppi: X contains Y AND touching boundary.
- "X is in Y" / "Y covers X" / "Y has X" without touch cue => ntpp (default: no touch).
- "X is inside and touching Y" / "X is within and touches Y" => tpp (boundary contact).
- ONLY choose tpp/tppi when the text EXPLICITLY says "touching", "inside and touching", or "touching boundary". Otherwise default to ntpp/ntppi.
- "C has O" = "O is in C" (same fact, different direction).
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2) TINY NORMALISATION HELPERS (no external deps, used inside prompt builders)
# ─────────────────────────────────────────────────────────────────────────────

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
    return re.sub(r"\s+", " ", a).strip()


def _relations_to_sentences(relations: List[Dict[str, Any]]) -> str:
    """Format relation list as bullet lines for use in prompts."""
    lines = []
    for r in relations or []:
        if not isinstance(r, dict):
            continue
        h = str(r.get("head", "")).strip()
        t = str(r.get("tail", "")).strip()
        rel = normalize_answer_any(str(r.get("relation", "")).strip())
        if h and t and rel:
            lines.append(f"- {h} is {rel} of {t}")
    return "\n".join(lines) if lines else "- (none)"


# ─────────────────────────────────────────────────────────────────────────────
# 3) Q1 / ANSWER PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_q1_baseline_and_support_sentences_spartun(
    story_text: str,
    question: str,
    candidate_answers: List[str],
) -> str:
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    options_list = ", ".join(sorted(set(cand)))
    return f"""
Return EXACTLY this JSON:
{{
  "support_sentences": ["<verbatim STORY sentence>", "..."]
  "justification": "<reason using ONLY STORY>",
  "answer": "<Choose all the options that fit from here: {options_list}>",
}}

Rules:
- USE ONLY the STORY below.
- support_sentences MUST be copied verbatim as full sentences from STORY.
- Include every sentence needed to fully answer the question, but no more:
  * Sentences that directly state the asked relation or fact.
  * Sentences needed to resolve coreferences (e.g., if a relevant sentence says "this box" or "it", also include the sentence that establishes what it refers to).
  * Sentences establishing containment or nesting (e.g., "X is in Y", "Y covers Z") when the question requires reasoning through containers.
  * For quantifier questions ("all", "every", "any", "no" + noun), include ALL sentences mentioning entities of the quantified type.
  * For multi-hop questions, include every sentence in the reasoning chain even if it only provides an intermediate link.
- Do NOT include sentences about unrelated entities or facts that play no role in answering.
- Do NOT invent sentences.
- The answer MUST be exactly one option from the list.
- You MUST always provide a concrete answer from the options. Do NOT leave answer empty or blank.
- Even if the reasoning is complex, commit to the best-supported option.

{SEMANTICS_BLOCK}

STORY:
<<<STORY>>>
{story_text}
<<<END STORY>>>

QUESTION:
{question}
""".strip()


def prompt_answer_only_allow_dontknow_spartun(
    text_block: str,
    question: str,
    candidate_answers: List[str],
) -> str:
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    opts = sorted(set(cand + ["dont know"]))
    options_list = ", ".join(opts)
    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<short reasoning using ONLY the provided text>",
  "answer": "<Choose all the options that fit of: {options_list}>"
}}

Rules:
- USE ONLY the given text.
- If insufficient to determine a unique answer, return "dont know".
- Do NOT guess.

{SEMANTICS_BLOCK}

TEXT:
<<<TEXT>>>
{text_block}
<<<END TEXT>>>

QUESTION:
{question}
""".strip()


def prompt_answer_forced_spartun(
    text_block: str,
    question: str,
    candidate_answers: List[str],
) -> str:
    """Sufficiency prompt: forces model to pick from candidate answers (no 'dont know').
    Used for Q1-S to test whether support sentences are sufficient."""
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    options_list = ", ".join(sorted(set(cand)))
    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<short reasoning using ONLY the provided text>",
  "answer": "<one of: {options_list}>"
}}

Rules:
- USE ONLY the given text.
- You MUST pick exactly one of the listed options based on the text.
- Reason carefully from what the text states or implies.

{SEMANTICS_BLOCK}

TEXT:
<<<TEXT>>>
{text_block}
<<<END TEXT>>>

QUESTION:
{question}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 4) SYMBOLIC / RELATION-BASED PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def build_symbolic_QA_prompt_spartun(
    story_text: str,
    question_text: str,
    candidate_answers: List[str],
) -> str:
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    options_list = ", ".join(sorted(set(cand)))
    options_json = __import__("json").dumps(sorted(set(cand)), ensure_ascii=False)

    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<give your reason>",
  "selected_option": "<Choose all the valid options here: {options_list}>"
}}

Rules:
- USE ONLY the STORY.
- Containment: "X is in Y" => X is ntpp/tpp of Y. "Y has X" => Y is ntppi/tppi of X.
- If no boundary-touch cue, prefer ntpp/ntppi.
- dc=disconnected, ec=touching, po=partial overlap.
- You MUST select exactly one option. Do NOT leave selected_option empty or blank.
- Even if the reasoning requires multi-hop inference, commit to the best-supported option.

STORY:
{story_text}

Question:
{question_text}


{SEMANTICS_BLOCK}

Options: {options_json}
""".strip()


def build_text_only_with_relations_prompt_from_inst_spartun(inst: Dict[str, Any]) -> str:
    cand = [normalize_answer_any(x) for x in (inst.get("candidate_answers") or []) if str(x).strip()]
    options_list = ", ".join(sorted(set(cand)))
    options_json = __import__("json").dumps(sorted(set(cand)), ensure_ascii=False)

    rels_block = _relations_to_sentences(inst.get("relations", []))
    story = str(inst.get("story", "") or "").strip()
    question = str(inst.get("question", "") or "").strip()

    return f"""
Return EXACTLY this JSON:
{{
  "justification": "<reason using ONLY the RELATIONS list (story only if tie-break)>",
  "selected_option": "<Choose all the valid options from here:  {options_list}>"
}}

Source of truth:
- USE ONLY the RELATIONS list below to answer. Ignore the STORY unless a tie-break is needed.
- If relations are insufficient, use STORY only to break ties; otherwise abstain from guessing and pick the best supported.
- You MUST select exactly one option. Do NOT leave selected_option empty or blank.
- Even if multi-hop reasoning is needed, commit to the best-supported answer.

RELATIONS:
{rels_block}

STORY (ignore unless needed):
{story}

Question:
{question}

{SEMANTICS_BLOCK}

How to compute BEFORE you output (brief internal steps, do not include in JSON):
1) Understand all relations and understand them.
2) Determine the asked relation/answer.
3) Map to exactly one option in {options_json}.
4) Output ONLY the JSON.
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 5) UNIFIED PARAPHRASE + FLIP PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def prompt_unified_paraphrase_and_flip(
    support_sentences: List[str],
    question: str,
    story_text: str = "",
) -> str:
    """Unified GPT prompt: generates 3 paraphrase levels + flipped question in one call."""
    support_text = " ".join([s.strip() for s in (support_sentences or []) if isinstance(s, str) and s.strip()]).strip()
    support_list = "\n".join(
        [f"- {s.strip()}" for s in (support_sentences or []) if isinstance(s, str) and s.strip()]
    ).strip() or "- (none)"
    story_block = (story_text or "").strip()
    return f"""
Rewrite the SUPPORT TEXT and QUESTION into 3 paraphrase levels, generate a flipped variant, choose up to 2 support sentences for answering the flipped question, and provide the expected answer.

Return ONLY JSON:
{{
  "support": {{
    "simple": "...",
    "hinted": "...",
    "canonical": "..."
  }},
  "question": {{
    "simple": "...",
    "hinted": "...",
    "canonical": "..."
  }},
  "flip_support_sentences": ["<verbatim support sentence>", "..."],
  "flipped_question": "<entity-swapped or negated question>",
  "flip_type": "entity_swap | yesno_negation | none",
  "flip_expected_answer": "<the correct answer to the flipped question given ONLY the flip_support_sentences>"
}}

Rules:
- Do NOT add or remove facts.
- Paraphrases must preserve semantics.
- Flipped question must preserve semantics but swap the two queried entities.
- For YES/NO questions: negate the question.

Flip support sentence selection (CRITICAL):
- Choose at most 2 support sentences for the flip question.
- flip_support_sentences MUST be copied verbatim from the support sentence list below.
- Pick the minimal set needed to answer the flipped question.
- The selected sentences MUST be self-contained: a reader seeing ONLY those 1-2 sentences (and nothing else) must be able to answer the flipped question.
- AVOID sentences that rely on coreferences (pronouns like "it", "they", "the object") whose antecedent is in a different sentence you did NOT select. If a sentence uses a pronoun, either pick the antecedent sentence too (within the 2-sentence limit) or pick a different sentence that names entities explicitly.
- If no 2-sentence subset is self-contained enough, pick the best available and note this.

Flip expected answer (CRITICAL FORMAT):
- After selecting the flip_support_sentences and flipped_question, answer the flipped question.
- For entity_swap: invert the spatial relation (left↔right, above↔below, front↔behind, ntpp↔ntppi, tpp↔tppi). Relations dc, ec, po, near, far stay the same.
- For yesno_negation: flip yes↔no.
- flip_expected_answer MUST use ONLY these formal relation terms:
  dc, ec, po, ntpp, ntppi, tpp, tppi, left, right, above, below, front, behind, near, far, yes, no
- Map natural language to formal terms:
  "covered by" / "inside" / "within" / "in" → ntpp
  "covers" / "contains" / "has" → ntppi
  "inside and touching" / "within and touches" → tpp
  "contains and touches boundary" → tppi
  "disconnected" → dc; "touching" → ec; "overlaps" → po
- If TWO or more options are BOTH valid (e.g., "disconnected and far"), return as a JSON list: ["dc", "far"]
- NEVER use natural language phrases like "covered by", "is inside", "within" as the answer.
- Preserve containment direction strictly.

SUPPORT:
<<<SUPPORT>>>
{support_text}
<<<END SUPPORT>>>

SUPPORT SENTENCE LIST:
<<<SUPPORT SENTENCES>>>
{support_list}
<<<END SUPPORT SENTENCES>>>

QUESTION:
<<<QUESTION>>>
{question}
<<<END QUESTION>>>

FULL STORY (use for additional context when determining flip_expected_answer):
<<<STORY>>>
{story_block}
<<<END STORY>>>
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 6) Q2 ANSWER PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_q2_answer_from_text_spartun(
    text_block: str,
    question: str,
    candidate_answers: List[str],
) -> str:
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    opts = sorted(set(cand + ["dont know"]))
    options_list = ", ".join(opts)
    return f"""
Return EXACTLY this JSON:
{{
  "answer": "<one of: {options_list}>",
  "justification": "<short reasoning using ONLY the text>"
}}

Rules:
- USE ONLY the TEXT.
- Think step-by-step: identify every entity mentioned, then trace the spatial chain carefully.
- If the question swaps entities compared to what the text states, invert the relation accordingly (e.g. "A is below B" means "B is above A").
- You MUST commit to a concrete spatial answer whenever the text provides ANY relevant information. Reason through it even if it requires multi-hop inference.
- Only return "dont know" as an absolute last resort when the text contains genuinely NO information about either entity.
- Treat containment paraphrases as equivalent: "C has O" == "O is in C" (keep container/object direction correct).

{SEMANTICS_BLOCK}

TEXT:
<<<TEXT>>>
{text_block}
<<<END TEXT>>>

QUESTION:
{question}
""".strip()


def prompt_q2_answer_no_dontknow_spartun(
    text_block: str,
    question: str,
    candidate_answers: List[str],
) -> str:
    """Q2 paraphrase prompt: forces model to pick from candidate answers (no 'dont know').
    Used to measure paraphrase stability without the model escaping to 'dont know'."""
    cand = [normalize_answer_any(x) for x in (candidate_answers or []) if str(x).strip()]
    opts = sorted(set(cand))
    options_list = ", ".join(opts)
    return f"""
Return EXACTLY this JSON:
{{
  "answer": "<one of: {options_list}>",
  "justification": "<short reasoning using ONLY the text>"
}}

Rules:
- USE ONLY the TEXT.
- You MUST pick exactly one of the listed options.
- Reason carefully and pick the best-supported answer.
- Treat containment paraphrases as equivalent: "C has O" == "O is in C" (keep container/object direction correct).

{SEMANTICS_BLOCK}

TEXT:
<<<TEXT>>>
{text_block}
<<<END TEXT>>>

QUESTION:
{question}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 7) FLIP / NEGATE QUESTION PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

def prompt_flip_question_rewrite_spartun(
    original_question: str,
    entity_a_name: str,
    entity_b_name: str,
) -> str:
    return f"""
You will rewrite a spatial question by swapping the two queried entities.

Return ONLY JSON:
{{
  "flipped_question": "<question with entities swapped>",
  "note": "<short note about what you did>"
}}

Key semantics (especially containment):
{SEMANTICS_BLOCK}

Rules:
- Keep the question type the same (Yes/No remains Yes/No; free-response remains free-response).
- Do NOT change facts or add new facts.
- ONLY swap which entity is the reference vs target.
- Use these exact entity surface forms when possible:
  EntityA: "{entity_a_name}"
  EntityB: "{entity_b_name}"

Containment correctness (MUST follow):
- If the meaning is "X contains Y" (has/contains/covers/includes; ntppi/tppi), keep X as the container and Y as the contained object.
- If the meaning is "Y is in X" (ntpp/tpp), keep Y as the inside object and X as the container.
- Equivalent forms are allowed and should be treated as consistent:
  "Is Y in X?" ⇔ "Does X contain Y?" ⇔ "Does X have Y?"
- Forbidden: accidentally reversing container/object ("Is X in Y?") unless the original said so.

Original question:
<<<Q>>>
{original_question}
<<<END Q>>>
""".strip()


def prompt_negate_yesno_question_spartun(original_question: str) -> str:
    return f"""
You will rewrite a YES/NO question into its NEGATED form so that the correct answer flips (yes <-> no).

Return ONLY JSON:
{{
  "negated_question": "<negated yes/no question>",
  "note": "<short note>"
}}

Key semantics (especially containment):
{SEMANTICS_BLOCK}

Rules:
- Preserve meaning except for adding/removing negation so the answer flips.
- Keep it a yes/no question.
- Do NOT add new facts.
- If the question is "Is there X in Y?" you may rewrite as:
  - "Is there NOT X in Y?" OR "Does Y NOT have X?"
- If the question already contains negation, remove it (double-negation cancellation) so that the answer flips.
- Keep entity mentions and containers the same.
- IMPORTANT: use containment equivalences correctly:
  "Y has X" ⇔ "X is in Y" (do NOT reverse container/object).

Original question:
<<<Q>>>
{original_question}
<<<END Q>>>
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# 8) COMPLEXITY PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def prompt_complexity_spartun_via_gpt(
    story_text: str,
    question: str,
    support_sentences: List[str],
) -> str:
    support_text = " ".join(
        [s.strip() for s in (support_sentences or []) if isinstance(s, str) and s.strip()]
    ).strip()
    return f"""
Analyze the STORY, QUESTION, and SUPPORT TEXT and return features for complexity.

Return ONLY JSON:
{{
  "num_entities": <int>,
  "entities": ["<canonical entity mention>", "..."],
  "coref_difficulty": <float 0..1>,
  "hard_language": {{
    "items": [
      {{"span":"<exact phrase>", "type":"clock|direction|coref", "difficulty": <float 0..1>}}
    ]
  }},
  "notes": "<short note>"
}}

IMPORTANT CALIBRATION (please follow):
- Use *moderate* difficulty values by default.
- Reserve very high scores ONLY for truly hard cases.

Recommended ranges:
- coref spans:
  - Explicit named entity with number (e.g., "medium yellow apple number one"): 0.05–0.20
  - Mild coref (e.g., "this box", "that one", "the medium yellow apple"): 0.25–0.55
  - Heavy / multi-step coref (e.g., "the box above the medium yellow apple", chained anaphora): 0.60–0.85 (rare)
- direction spans:
  - Simple (left/right/above/below/north/south/east/west): 0.10–0.40
  - Chained/composed directions (multi-hop): 0.45–0.70
  - Clock positions (3 o'clock etc.): 0.35–0.70
- coref_difficulty overall:
  - keep typical values in 0.25–0.60; >0.75 should be rare and justified.

Guidelines:
- "entities" should be object mentions (e.g., "box one", "medium yellow apple", ...).
- num_entities should match unique entities you list.
- "coref_difficulty" estimates how much coreference is needed overall.

STORY:
<<<STORY>>>
{story_text}
<<<END STORY>>>

SUPPORT TEXT:
<<<SUPPORT>>>
{support_text}
<<<END SUPPORT>>>

QUESTION:
<<<QUESTION>>>
{question}
<<<END QUESTION>>>
""".strip()
