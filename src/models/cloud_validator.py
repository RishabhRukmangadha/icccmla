"""
Cloud Validator Module
Sends escalated questions to Claude Opus with extended thinking.
Acts as the cloud LLM in the cascade architecture.
"""

import time
import anthropic
from src.utils.config import VALIDATOR_THINKING_BUDGET
from dataclasses import dataclass
from src.utils.config import (
    ANTHROPIC_API_KEY,
    CLOUD_VALIDATOR_MODEL,
)
from src.utils.logger import logger


# ─────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────

@dataclass
class CloudResponse:
    """Claude Opus validation result."""
    model:          str
    question_id:    str
    question_type:  str
    answer:         str       # A, B, C, or D
    answer_text:    str       # full answer text
    reasoning:      str       # Claude's explanation
    thinking:       str       # extended thinking trace
    latency_ms:     float     # inference time
    parse_success:  bool      # did we parse correctly
    tokens_used:    int       # total tokens consumed


# ─────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────

def build_validator_prompt(question: dict) -> str:
    """
    Build validation prompt for Claude Opus.
    More detailed than SLM prompt — asks for
    thorough reasoning and explanation.
    """

    choices = question["choices"]
    labels  = ["A", "B", "C", "D"]

    options_text = "\n".join([
        f"{labels[i]}) {choices[i]}"
        for i in range(len(choices))
    ])

    prompt = f"""You are an expert physics professor validating a student's physics question.

A student's on-device AI tutor was uncertain about this question and escalated it to you for validation.

Question: {question['question']}

Options:
{options_text}

Instructions:
1. Carefully analyse the question
2. Select the correct option (A, B, C, or D)
3. Provide a clear, educational explanation suitable for a student
4. If this is a formula-based question, show the calculation steps
5. If this is a conceptual question, explain the underlying physics principle

Respond in EXACTLY this format:
ANSWER: <A or B or C or D>
EXPLANATION: <detailed explanation for the student>"""

    return prompt


# ─────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────

def parse_cloud_response(
    raw:         str,
    question:    dict,
    question_id: str
) -> tuple:
    """
    Parse Claude Opus response.
    Returns (answer, explanation, parse_success)
    """

    import re

    answer      = None
    explanation = ""

    try:
        # Extract ANSWER
        answer_match = re.search(
            r'ANSWER:\s*([A-D])',
            raw,
            re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1).upper()

        # Extract EXPLANATION
        explanation_match = re.search(
            r'EXPLANATION:\s*(.+)',
            raw,
            re.IGNORECASE | re.DOTALL
        )
        if explanation_match:
            explanation = explanation_match.group(1).strip()

        parse_success = answer is not None

        if not parse_success:
            logger.warning(
                f"Cloud parse incomplete for {question_id} "
                f"— answer={answer}"
            )
            answer = "A"

    except Exception as e:
        logger.error(f"Cloud parse error for {question_id}: {e}")
        answer        = "A"
        explanation   = f"Parse error: {str(e)}"
        parse_success = False

    return answer, explanation, parse_success


# ─────────────────────────────────────────
# Claude Opus Validator
# ─────────────────────────────────────────

def run_cloud_validation(question: dict) -> CloudResponse:
    """
    Send escalated question to Claude Opus.
    Uses extended thinking for deep reasoning.

    Args:
        question: question dict from processed dataset

    Returns:
        CloudResponse dataclass
    """

    question_id   = question["question_id"]
    question_type = question["question_type"]
    prompt        = build_validator_prompt(question)

    logger.info(
        f"  [Claude Sonnet] Validating {question_id}"
    )

    # Initialise Anthropic client
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    start_time = time.time()

    try:
        # Call Claude Opus with extended thinking
        response = client.messages.create(
            model      = CLOUD_VALIDATOR_MODEL,
            max_tokens = 16000,
            thinking   = {
                "type":          "enabled",
                "budget_tokens": VALIDATOR_THINKING_BUDGET,
            },
            messages = [
                {
                    "role":    "user",
                    "content": prompt
                }
            ]
        )

        latency_ms  = (time.time() - start_time) * 1000
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        # Extract thinking trace and text response
        thinking_text = ""
        answer_text   = ""

        for block in response.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                answer_text = block.text

        # Parse response
        answer, explanation, parse_success = parse_cloud_response(
            answer_text,
            question,
            question_id
        )

        # Map answer to text
        labels     = ["A", "B", "C", "D"]
        answer_idx = labels.index(answer) if answer in labels else 0
        answer_txt = question["choices"][answer_idx]

        logger.info(
            f"  [Claude Sonnet] {question_id} → "
            f"Answer: {answer}, "
            f"Tokens: {tokens_used}, "
            f"Latency: {latency_ms:.0f}ms"
        )

        return CloudResponse(
            model         = CLOUD_VALIDATOR_MODEL,
            question_id   = question_id,
            question_type = question_type,
            answer        = answer,
            answer_text   = answer_txt,
            reasoning     = explanation,
            thinking      = thinking_text,
            latency_ms    = latency_ms,
            parse_success = parse_success,
            tokens_used   = tokens_used,
        )

    except Exception as e:
        logger.error(
            f"Claude Sonnet error for {question_id}: {e}"
        )
        latency_ms = (time.time() - start_time) * 1000

        return CloudResponse(
            model         = CLOUD_VALIDATOR_MODEL,
            question_id   = question_id,
            question_type = question_type,
            answer        = "A",
            answer_text   = question["choices"][0],
            reasoning     = f"Error: {str(e)}",
            thinking      = "",
            latency_ms    = latency_ms,
            parse_success = False,
            tokens_used   = 0,
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
    print("Testing Cloud Validator — Claude Sonnet")
    print("="*50)

    result = run_cloud_validation(test_question)

    print(f"\nModel:         {result.model}")
    print(f"Answer:        {result.answer}")
    print(f"Answer Text:   {result.answer_text}")
    print(f"Tokens Used:   {result.tokens_used}")
    print(f"Latency:       {result.latency_ms:.0f}ms")
    print(f"Parse Success: {result.parse_success}")
    print(f"\nThinking:\n{result.thinking[:500]}...")
    print(f"\nExplanation:\n{result.reasoning}")