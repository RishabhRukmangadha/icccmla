"""
SLM Inference Module
Handles querying on-device SLMs via Ollama.
Implements hybrid escalation trigger:
- Verbalized confidence score
- Answer consistency across 2 runs
"""

import re
import time
import ollama
from dataclasses import dataclass
from src.utils.config import (
    SLM_MODELS,
    ESCALATION_THRESHOLD,
    OLLAMA_BASE_URL,
)
from src.utils.logger import logger


# ─────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────

@dataclass
class SLMResponse:
    """Single SLM inference result."""
    model:          str
    question_id:    str
    question_type:  str
    answer:         str       # A, B, C, or D
    answer_text:    str       # full answer text
    confidence:     int       # 1-5 scale
    reasoning:      str       # one sentence explanation
    latency_ms:     float     # inference time in ms
    raw_response:   str       # full raw model output
    parse_success:  bool      # did we parse correctly


@dataclass
class SLMResult:
    """Combined result from 2 runs — hybrid approach."""
    model:           str
    question_id:     str
    question_type:   str
    run1:            SLMResponse
    run2:            SLMResponse
    final_answer:    str       # consensus answer
    avg_confidence:  float     # average confidence
    consistent:      bool      # both runs same answer?
    should_escalate: bool      # escalation decision
    escalation_reason: str     # why escalated


# ─────────────────────────────────────────
# Prompt Template
# ─────────────────────────────────────────

def build_prompt(question: dict) -> str:
    """
    Build the inference prompt for the SLM.
    Elicits answer + verbalized confidence + reasoning.
    """

    choices = question["choices"]
    labels  = ["A", "B", "C", "D"]

    options_text = "\n".join([
        f"{labels[i]}) {choices[i]}"
        for i in range(len(choices))
    ])

    prompt = f"""You are a physics tutor assistant. Answer the following multiple choice physics question.

Question: {question['question']}

Options:
{options_text}

Instructions:
1. Select the correct option (A, B, C, or D)
2. Rate your confidence on a scale of 1-5:
   1 = Very uncertain — I am guessing
   2 = Uncertain — I have some doubt
   3 = Somewhat confident — I think this is right
   4 = Confident — I am fairly sure
   5 = Very confident — I am certain
3. Provide one sentence explaining your answer

Respond in EXACTLY this format:
ANSWER: <A or B or C or D>
CONFIDENCE: <1 or 2 or 3 or 4 or 5>
REASONING: <one sentence>"""

    return prompt


# ─────────────────────────────────────────
# Response Parser
# ─────────────────────────────────────────

def parse_response(raw: str, model: str, question_id: str) -> tuple:
    """
    Parse SLM response to extract answer, confidence, reasoning.
    Returns (answer, confidence, reasoning, parse_success)
    """

    answer     = None
    confidence = None
    reasoning  = ""

    try:
        # Extract ANSWER
        answer_match = re.search(
            r'ANSWER:\s*([A-D])',
            raw,
            re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1).upper()

        # Extract CONFIDENCE
        confidence_match = re.search(
            r'CONFIDENCE:\s*([1-5])',
            raw
        )
        if confidence_match:
            confidence = int(confidence_match.group(1))

        # Extract REASONING
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?:\n|$)',
            raw,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Validate
        parse_success = (
            answer is not None and
            confidence is not None
        )

        if not parse_success:
            logger.warning(
                f"Parse incomplete for {model} on {question_id} "
                f"— answer={answer}, confidence={confidence}"
            )
            # Defaults for failed parse
            if answer is None:
                answer = "A"
            if confidence is None:
                confidence = 1  # assume low confidence on parse fail

    except Exception as e:
        logger.error(f"Parse error for {model} on {question_id}: {e}")
        answer        = "A"
        confidence    = 1
        reasoning     = "Parse error"
        parse_success = False

    return answer, confidence, reasoning, parse_success


# ─────────────────────────────────────────
# Single Run Inference
# ─────────────────────────────────────────

def run_single_inference(
    model_name: str,
    question:   dict,
    run_number: int = 1
) -> SLMResponse:
    """
    Run a single inference on the SLM via Ollama.
    Returns SLMResponse dataclass.
    """

    question_id   = question["question_id"]
    question_type = question["question_type"]
    prompt        = build_prompt(question)

    logger.debug(
        f"Running {model_name} on {question_id} (run {run_number})"
    )

    start_time = time.time()

    try:
        # Call Ollama API
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role":    "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.1,  # low temp for consistency
                "top_p":       0.9,
            }
        )

        raw_response = response["message"]["content"]
        latency_ms   = (time.time() - start_time) * 1000

        # Parse response
        answer, confidence, reasoning, parse_success = parse_response(
            raw_response,
            model_name,
            question_id
        )

        # Map answer letter to answer text
        labels      = ["A", "B", "C", "D"]
        answer_idx  = labels.index(answer) if answer in labels else 0
        answer_text = question["choices"][answer_idx]

        return SLMResponse(
            model         = model_name,
            question_id   = question_id,
            question_type = question_type,
            answer        = answer,
            answer_text   = answer_text,
            confidence    = confidence,
            reasoning     = reasoning,
            latency_ms    = latency_ms,
            raw_response  = raw_response,
            parse_success = parse_success,
        )

    except Exception as e:
        logger.error(
            f"Ollama error for {model_name} on {question_id}: {e}"
        )
        latency_ms = (time.time() - start_time) * 1000

        return SLMResponse(
            model         = model_name,
            question_id   = question_id,
            question_type = question_type,
            answer        = "A",
            answer_text   = question["choices"][0],
            confidence    = 1,
            reasoning     = f"Error: {str(e)}",
            latency_ms    = latency_ms,
            raw_response  = "",
            parse_success = False,
        )


# ─────────────────────────────────────────
# Hybrid Escalation Decision
# ─────────────────────────────────────────

def hybrid_escalation_decision(
    run1: SLMResponse,
    run2: SLMResponse
) -> tuple:
    """
    Hybrid escalation decision using:
    1. Answer consistency across 2 runs
    2. Average verbalized confidence

    Returns (should_escalate: bool, reason: str)
    """

    consistent     = (run1.answer == run2.answer)
    avg_confidence = (run1.confidence + run2.confidence) / 2

    if not consistent:
        # Inconsistent answers → always escalate
        return True, "inconsistent_answers"

    if avg_confidence <= ESCALATION_THRESHOLD:
        # Consistent but low confidence → escalate
        return True, "low_confidence"

    # Consistent and confident → don't escalate
    return False, "confident_and_consistent"


# ─────────────────────────────────────────
# Main SLM Inference Function
# ─────────────────────────────────────────

def run_slm_inference(
    model_key:  str,
    question:   dict
) -> SLMResult:
    """
    Run hybrid inference on a single question.
    2 runs per question for consistency check.

    Args:
        model_key: 'small', 'medium', or 'large'
        question:  question dict from processed dataset

    Returns:
        SLMResult with escalation decision
    """

    model_name  = SLM_MODELS[model_key]
    question_id = question["question_id"]

    logger.info(f"  [{model_name}] Processing {question_id}")

    # Run 1
    run1 = run_single_inference(model_name, question, run_number=1)
    logger.debug(
        f"  Run 1 → Answer: {run1.answer}, "
        f"Confidence: {run1.confidence}"
    )

    # Run 2
    run2 = run_single_inference(model_name, question, run_number=2)
    logger.debug(
        f"  Run 2 → Answer: {run2.answer}, "
        f"Confidence: {run2.confidence}"
    )

    # Hybrid escalation decision
    should_escalate, reason = hybrid_escalation_decision(run1, run2)

    # Final answer — use run1 if consistent, else mark as uncertain
    final_answer   = run1.answer if run1.answer == run2.answer else "?"
    avg_confidence = (run1.confidence + run2.confidence) / 2
    consistent     = (run1.answer == run2.answer)

    logger.info(
        f"  [{model_name}] {question_id} → "
        f"Answer: {final_answer}, "
        f"Confidence: {avg_confidence:.1f}, "
        f"Consistent: {consistent}, "
        f"Escalate: {should_escalate} ({reason})"
    )

    return SLMResult(
        model             = model_name,
        question_id       = question_id,
        question_type     = question["question_type"],
        run1              = run1,
        run2              = run2,
        final_answer      = final_answer,
        avg_confidence    = avg_confidence,
        consistent        = consistent,
        should_escalate   = should_escalate,
        escalation_reason = reason,
    )


# ─────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Test question
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
    print("Testing SLM Inference — Gemma 3 1B")
    print("="*50)

    result = run_slm_inference("small", test_question)

    print(f"\nModel:             {result.model}")
    print(f"Final Answer:      {result.final_answer}")
    print(f"Avg Confidence:    {result.avg_confidence}")
    print(f"Consistent:        {result.consistent}")
    print(f"Should Escalate:   {result.should_escalate}")
    print(f"Reason:            {result.escalation_reason}")
    print(f"\nRun 1 Raw Response:\n{result.run1.raw_response}")
    print(f"\nRun 2 Raw Response:\n{result.run2.raw_response}")