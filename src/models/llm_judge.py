"""
LLM Judge Module
Uses Claude Sonnet 4.5 with extended thinking to evaluate answer quality.
Scores Gap Severity (GSS), Escalation Value (EVS), and Explanation Quality (EQS).
"""

import re
import time
import json
import anthropic
from dataclasses import dataclass
from src.utils.config import (
    ANTHROPIC_API_KEY,
    JUDGE_MODEL,
    JUDGE_THINKING_BUDGET,
    GSS_LABELS,
    EVS_LABELS,
    CONTAINMENT_THRESHOLD,
)
from src.utils.logger import logger


# ─────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────

@dataclass
class GSSResult:
    """Gap Severity Score — for missed SLM answers."""
    question_id:    str
    slm_answer:     str
    correct_answer: str
    gss_score:      int
    gss_label:      str
    reasoning:      str
    thinking:       str
    latency_ms:     float


@dataclass
class EVSResult:
    """Escalation Value Score — for escalated questions."""
    question_id:    str
    slm_answer:     str
    cloud_answer:   str
    correct_answer: str
    evs_score:      int
    evs_label:      str
    containable:    bool
    reasoning:      str
    thinking:       str
    latency_ms:     float


@dataclass
class EQSResult:
    """Explanation Quality Score — for conceptual questions."""
    question_id:  str
    answer:       str
    explanation:  str
    eqs_score:    int
    reasoning:    str
    thinking:     str
    latency_ms:   float


# ─────────────────────────────────────────
# Judge Prompts
# ─────────────────────────────────────────

def build_gss_prompt(
    question:       dict,
    slm_answer:     str,
    correct_answer: str
) -> str:

    choices      = question["choices"]
    labels       = ["A", "B", "C", "D"]
    options_text = "\n".join([
        f"{labels[i]}) {choices[i]}"
        for i in range(len(choices))
    ])

    return f"""You are an impartial physics education evaluator.

Question: {question['question']}

Options:
{options_text}

Correct Answer: {correct_answer}
AI Tutor Answer: {slm_answer}

The AI tutor answered incorrectly without escalating.
Rate the SEVERITY of this error:

1 = TRIVIAL: Minor error, student still benefits
2 = MODERATE: Significant error, student partially misled
3 = CRITICAL: Completely wrong or dangerously misleading

Respond in EXACTLY this JSON format:
{{
  "gss_score": <1, 2, or 3>,
  "reasoning": "<one sentence>"
}}"""


def build_evs_prompt(
    question:       dict,
    slm_answer:     str,
    cloud_answer:   str,
    correct_answer: str
) -> str:

    choices      = question["choices"]
    labels       = ["A", "B", "C", "D"]
    options_text = "\n".join([
        f"{labels[i]}) {choices[i]}"
        for i in range(len(choices))
    ])

    # Truncate cloud answer to keep prompt lean
    cloud_answer_short = (
        cloud_answer[:500] + "..."
        if len(cloud_answer) > 500
        else cloud_answer
    )

    return f"""You are an impartial physics education evaluator.

Question: {question['question']}

Options:
{options_text}

Correct Answer: {correct_answer}
On-Device AI Answer: {slm_answer}
Cloud AI Answer: {cloud_answer_short}

Rate the VALUE ADDED by escalating to the cloud AI:

0 = ZERO: Cloud not more correct — escalation wasted
1 = TRIVIAL: Marginally better — largely unnecessary
2 = SIGNIFICANT: Meaningfully more correct — worthwhile
3 = CRITICAL: Substantially better — clearly necessary

Respond in EXACTLY this JSON format:
{{
  "evs_score": <0, 1, 2, or 3>,
  "reasoning": "<one sentence>"
}}"""


def build_eqs_prompt(
    question:    dict,
    answer:      str,
    explanation: str
) -> str:

    explanation_short = (
        explanation[:500] + "..."
        if len(explanation) > 500
        else explanation
    )

    return f"""You are an impartial physics education evaluator.

Question: {question['question']}
Answer: {answer}
Explanation: {explanation_short}

Rate the EXPLANATION QUALITY for a student:

1 = Very Poor: Incorrect or misleading
2 = Poor: Partially correct but confusing
3 = Adequate: Correct but lacks depth
4 = Good: Clear, correct and educational
5 = Excellent: Clear, correct and insightful

Respond in EXACTLY this JSON format:
{{
  "eqs_score": <1, 2, 3, 4, or 5>,
  "reasoning": "<one sentence>"
}}"""


# ─────────────────────────────────────────
# Judge Inference
# ─────────────────────────────────────────

def call_judge(prompt: str, question_id: str) -> tuple:
    """
    Call Claude Sonnet judge with extended thinking.
    Returns (parsed_json, thinking_text, latency_ms)
    """

    client     = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    start_time = time.time()

    try:
        response = client.messages.create(
            model      = JUDGE_MODEL,
            max_tokens = 4000,
            thinking   = {
                "type":          "enabled",
                "budget_tokens": JUDGE_THINKING_BUDGET,
            },
            system = (
                "You are an impartial physics education evaluator. "
                "Always respond with valid JSON only in your final answer. "
                "No markdown formatting outside JSON."
            ),
            messages = [
                {
                    "role":    "user",
                    "content": prompt
                }
            ]
        )

        latency_ms    = (time.time() - start_time) * 1000
        thinking_text = ""
        answer_text   = ""

        # Extract thinking and text blocks
        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                answer_text = block.text

        # Clean and parse JSON
        clean = answer_text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"```json|```", "", clean).strip()

        parsed = json.loads(clean)
        return parsed, thinking_text, latency_ms

    except Exception as e:
        logger.error(f"Judge error for {question_id}: {e}")
        return None, "", (time.time() - start_time) * 1000


# ─────────────────────────────────────────
# GSS Scoring
# ─────────────────────────────────────────

def score_gap_severity(
    question:       dict,
    slm_answer:     str,
    correct_answer: str
) -> GSSResult:

    question_id = question["question_id"]
    prompt      = build_gss_prompt(
        question, slm_answer, correct_answer
    )

    logger.info(f"  [Judge-GSS] Scoring {question_id}")

    parsed, thinking, latency_ms = call_judge(
        prompt, question_id
    )

    if parsed and "gss_score" in parsed:
        gss_score = int(parsed["gss_score"])
        reasoning = parsed.get("reasoning", "")
    else:
        logger.warning(f"GSS parse failed for {question_id}")
        gss_score = 2
        reasoning = "Parse failed"
        thinking  = ""

    gss_label = GSS_LABELS.get(gss_score, "moderate")

    logger.info(
        f"  [Judge-GSS] {question_id} → "
        f"GSS={gss_score} ({gss_label})"
    )

    return GSSResult(
        question_id    = question_id,
        slm_answer     = slm_answer,
        correct_answer = correct_answer,
        gss_score      = gss_score,
        gss_label      = gss_label,
        reasoning      = reasoning,
        thinking       = thinking,
        latency_ms     = latency_ms,
    )


# ─────────────────────────────────────────
# EVS Scoring
# ─────────────────────────────────────────

def score_escalation_value(
    question:       dict,
    slm_answer:     str,
    cloud_answer:   str,
    correct_answer: str
) -> EVSResult:

    question_id = question["question_id"]
    prompt      = build_evs_prompt(
        question, slm_answer,
        cloud_answer, correct_answer
    )

    logger.info(f"  [Judge-EVS] Scoring {question_id}")

    parsed, thinking, latency_ms = call_judge(
        prompt, question_id
    )

    if parsed and "evs_score" in parsed:
        evs_score = int(parsed["evs_score"])
        reasoning = parsed.get("reasoning", "")
    else:
        logger.warning(f"EVS parse failed for {question_id}")
        evs_score = 1
        reasoning = "Parse failed"
        thinking  = ""

    evs_label   = EVS_LABELS.get(evs_score, "trivial")
    containable = evs_score <= CONTAINMENT_THRESHOLD

    logger.info(
        f"  [Judge-EVS] {question_id} → "
        f"EVS={evs_score} ({evs_label}), "
        f"Containable={containable}"
    )

    return EVSResult(
        question_id    = question_id,
        slm_answer     = slm_answer,
        cloud_answer   = cloud_answer,
        correct_answer = correct_answer,
        evs_score      = evs_score,
        evs_label      = evs_label,
        containable    = containable,
        reasoning      = reasoning,
        thinking       = thinking,
        latency_ms     = latency_ms,
    )


# ─────────────────────────────────────────
# EQS Scoring
# ─────────────────────────────────────────

def score_explanation_quality(
    question:    dict,
    answer:      str,
    explanation: str
) -> EQSResult:

    question_id = question["question_id"]
    prompt      = build_eqs_prompt(
        question, answer, explanation
    )

    logger.info(f"  [Judge-EQS] Scoring {question_id}")

    parsed, thinking, latency_ms = call_judge(
        prompt, question_id
    )

    if parsed and "eqs_score" in parsed:
        eqs_score = int(parsed["eqs_score"])
        reasoning = parsed.get("reasoning", "")
    else:
        logger.warning(f"EQS parse failed for {question_id}")
        eqs_score = 3
        reasoning = "Parse failed"
        thinking  = ""

    logger.info(
        f"  [Judge-EQS] {question_id} → "
        f"EQS={eqs_score}/5"
    )

    return EQSResult(
        question_id  = question_id,
        answer       = answer,
        explanation  = explanation,
        eqs_score    = eqs_score,
        reasoning    = reasoning,
        thinking     = thinking,
        latency_ms   = latency_ms,
    )


# ─────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────

if __name__ == "__main__":

    test_question = {
        "question_id":   "test_001",
        "question_type": "formula_based",
        "question": (
            "A 2 kg object is pushed by a 10 N force. "
            "What is its acceleration?"
        ),
        "choices": [
            "2 m/s²",
            "5 m/s²",
            "10 m/s²",
            "20 m/s²"
        ],
        "answer":      1,
        "answer_text": "5 m/s²",
    }

    print("\n" + "="*50)
    print("Testing LLM Judge — Claude Sonnet")
    print("="*50)

    # Test GSS
    print("\n── GSS Test ──")
    gss = score_gap_severity(
        question       = test_question,
        slm_answer     = "A) 2 m/s²",
        correct_answer = "B) 5 m/s²"
    )
    print(f"GSS Score:  {gss.gss_score} ({gss.gss_label})")
    print(f"Reasoning:  {gss.reasoning}")
    print(f"Thinking:   {gss.thinking[:200]}...")

    # Test EVS
    print("\n── EVS Test ──")
    evs = score_escalation_value(
        question       = test_question,
        slm_answer     = "A) 2 m/s²",
        cloud_answer   = "B) 5 m/s²",
        correct_answer = "B) 5 m/s²"
    )
    print(f"EVS Score:   {evs.evs_score} ({evs.evs_label})")
    print(f"Containable: {evs.containable}")
    print(f"Reasoning:   {evs.reasoning}")

    # Test EQS
    print("\n── EQS Test ──")
    eqs = score_explanation_quality(
        question    = test_question,
        answer      = "B) 5 m/s²",
        explanation = "Using F=ma, a = 10/2 = 5 m/s²"
    )
    print(f"EQS Score:  {eqs.eqs_score}/5")
    print(f"Reasoning:  {eqs.reasoning}")