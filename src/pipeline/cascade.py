"""
Main Cascade Pipeline
Orchestrates the full experiment across all models and questions.
Features:
- Checkpointing — resumes from last completed question
- Retry logic — handles API failures gracefully
- Progress tracking — shows ETA and completion status
- Caffeinate reminder — prevents Mac sleep
"""

import json
import time
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from src.utils.config import (
    SLM_MODELS,
    RESULTS_DIR,
    PROCESSED_DIR,
    FORMULA_QUESTIONS_FILE,
    CONCEPTUAL_QUESTIONS_FILE,
    EXPERIMENT_NAME,
)
from src.utils.logger import logger
from src.models.slm_inference import run_slm_inference
from src.models.cloud_validator import run_cloud_validation
from src.models.llm_judge import (
    score_gap_severity,
    score_escalation_value,
    score_explanation_quality,
)
from src.evaluation.metrics import (
    QuestionResult,
    compute_question_metrics,
    aggregate_metrics,
    generate_summary_report,
)


# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────

CHECKPOINT_FILE  = RESULTS_DIR / "checkpoint.json"
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.json"
METRICS_FILE     = RESULTS_DIR / "metrics_summary.json"


# ─────────────────────────────────────────
# Checkpoint Management
# ─────────────────────────────────────────

def load_checkpoint() -> dict:
    """Load checkpoint if exists, else return empty."""

    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            checkpoint = json.load(f)
        logger.info(
            f"Checkpoint loaded — "
            f"resuming from previous run"
        )
        for model, ids in checkpoint["completed_ids"].items():
            logger.info(
                f"  {model}: {len(ids)} questions completed"
            )
        return checkpoint

    # Fresh start
    return {
        "completed_ids": {k: [] for k in SLM_MODELS.keys()},
        "results":       {k: [] for k in SLM_MODELS.keys()},
        "started_at":    datetime.now().isoformat(),
        "updated_at":    datetime.now().isoformat(),
    }


def save_checkpoint(checkpoint: dict) -> None:
    """Save checkpoint after each question."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = datetime.now().isoformat()

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def clear_checkpoint() -> None:
    """Clear checkpoint for fresh run."""

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("Checkpoint cleared — starting fresh")


# ─────────────────────────────────────────
# Retry Logic
# ─────────────────────────────────────────

def with_retry(func, *args, max_retries=3, delay=5, **kwargs):
    """
    Execute function with retry logic.
    Handles API timeouts and rate limits.
    """

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = delay * (attempt + 1)
                logger.warning(
                    f"Attempt {attempt+1} failed: {e} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"All {max_retries} attempts failed: {e}"
                )
                raise


# ─────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────

def load_questions() -> list:
    """Load and combine all 200 questions."""

    with open(FORMULA_QUESTIONS_FILE, "r") as f:
        formula = json.load(f)

    with open(CONCEPTUAL_QUESTIONS_FILE, "r") as f:
        conceptual = json.load(f)

    all_questions = formula + conceptual
    logger.info(
        f"Loaded {len(all_questions)} questions "
        f"({len(formula)} formula + {len(conceptual)} conceptual)"
    )
    return all_questions


# ─────────────────────────────────────────
# Process Single Question
# ─────────────────────────────────────────

def process_question(
    model_key:  str,
    question:   dict,
) -> dict:
    """
    Full pipeline for one question × one model.
    Returns serializable dict result.
    """

    question_id   = question["question_id"]
    question_type = question["question_type"]
    correct_idx   = question["answer"]
    correct_label = ["A", "B", "C", "D"][correct_idx]
    correct_text  = question["choices"][correct_idx]

    # ── Step 1: SLM Inference ───────────
    slm_result = with_retry(
        run_slm_inference,
        model_key,
        question
    )

    slm_correct = (slm_result.final_answer == correct_label)

    # ── Step 2: Cloud Validation ─────────
    cloud_result   = None
    cloud_answer   = None
    cloud_correct  = None
    cloud_reasoning = None
    cloud_latency  = None

    if slm_result.should_escalate:
        cloud_result    = with_retry(
            run_cloud_validation,
            question
        )
        cloud_answer    = cloud_result.answer
        cloud_correct   = (cloud_answer == correct_label)
        cloud_reasoning = cloud_result.reasoning
        cloud_latency   = cloud_result.latency_ms

    # ── Step 3: Judge Scoring ────────────
    gss_score = gss_label = gss_reasoning = None
    evs_score = evs_label = evs_reasoning = None
    containable = None
    eqs_score = eqs_reasoning = None

    # GSS — SLM wrong and did NOT escalate
    if not slm_result.should_escalate and not slm_correct:
        gss = with_retry(
            score_gap_severity,
            question,
            f"{slm_result.final_answer}) "
            f"{question['choices'][['A','B','C','D'].index(slm_result.final_answer)] if slm_result.final_answer in ['A','B','C','D'] else 'Unknown'}",
            f"{correct_label}) {correct_text}"
        )
        gss_score     = gss.gss_score
        gss_label     = gss.gss_label
        gss_reasoning = gss.reasoning

    # EVS — escalated
    if slm_result.should_escalate and cloud_result:
        evs = with_retry(
            score_escalation_value,
            question,
            f"{slm_result.final_answer}",
            f"{cloud_answer}) {cloud_reasoning[:200] if cloud_reasoning else ''}",
            f"{correct_label}) {correct_text}"
        )
        evs_score     = evs.evs_score
        evs_label     = evs.evs_label
        evs_reasoning = evs.reasoning
        containable   = evs.containable

    # EQS — conceptual questions only
    if question_type == "conceptual":
        answer_text  = (
            cloud_reasoning
            if slm_result.should_escalate and cloud_reasoning
            else slm_result.run1.reasoning
        )
        eqs = with_retry(
            score_explanation_quality,
            question,
            cloud_answer if slm_result.should_escalate else slm_result.final_answer,
            answer_text or ""
        )
        eqs_score     = eqs.eqs_score
        eqs_reasoning = eqs.reasoning

    # ── Step 4: Build Result ─────────────
    result = QuestionResult(
        question_id       = question_id,
        question_type     = question_type,
        model             = SLM_MODELS[model_key],
        correct_answer    = correct_label,
        slm_answer        = slm_result.final_answer,
        slm_correct       = slm_correct,
        avg_confidence    = slm_result.avg_confidence,
        consistent        = slm_result.consistent,
        should_escalate   = slm_result.should_escalate,
        escalation_reason = slm_result.escalation_reason,
        slm_reasoning     = slm_result.run1.reasoning,
        slm_latency_ms    = slm_result.run1.latency_ms,
        cloud_answer      = cloud_answer,
        cloud_correct     = cloud_correct,
        cloud_reasoning   = cloud_reasoning,
        cloud_latency_ms  = cloud_latency,
        gss_score         = gss_score,
        gss_label         = gss_label,
        gss_reasoning     = gss_reasoning,
        evs_score         = evs_score,
        evs_label         = evs_label,
        evs_reasoning     = evs_reasoning,
        containable       = containable,
        eqs_score         = eqs_score,
        eqs_reasoning     = eqs_reasoning,
    )

    result = compute_question_metrics(result)

    # Return serializable dict
    return {
        "question_id":       result.question_id,
        "question_type":     result.question_type,
        "model":             result.model,
        "correct_answer":    result.correct_answer,
        "slm_answer":        result.slm_answer,
        "slm_correct":       result.slm_correct,
        "avg_confidence":    result.avg_confidence,
        "consistent":        result.consistent,
        "should_escalate":   result.should_escalate,
        "escalation_reason": result.escalation_reason,
        "slm_reasoning":     result.slm_reasoning,
        "slm_latency_ms":    result.slm_latency_ms,
        "cloud_answer":      result.cloud_answer,
        "cloud_correct":     result.cloud_correct,
        "cloud_reasoning":   result.cloud_reasoning,
        "cloud_latency_ms":  result.cloud_latency_ms,
        "gss_score":         result.gss_score,
        "gss_label":         result.gss_label,
        "gss_reasoning":     result.gss_reasoning,
        "evs_score":         result.evs_score,
        "evs_label":         result.evs_label,
        "evs_reasoning":     result.evs_reasoning,
        "containable":       result.containable,
        "eqs_score":         result.eqs_score,
        "eqs_reasoning":     result.eqs_reasoning,
        "aes_score":         result.aes_score,
        "final_correct":     result.final_correct,
    }


# ─────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────

def run_cascade(
    pilot_mode:    bool = False,
    pilot_n:       int  = 5,
    fresh_start:   bool = False,
) -> None:
    """
    Run full cascade experiment.

    Args:
        pilot_mode:  if True run only pilot_n questions per model
        pilot_n:     number of questions in pilot mode
        fresh_start: if True clear checkpoint and restart
    """

    logger.info("=" * 60)
    logger.info(f"ICCCMLA Cascade Experiment")
    logger.info(f"Experiment: {EXPERIMENT_NAME}")
    logger.info("=" * 60)

    # Caffeinate reminder
    logger.info(
        "\n⚠️  IMPORTANT: Run with caffeinate to prevent sleep:\n"
        "caffeinate -i python3 -m src.pipeline.cascade\n"
    )

    # Clear checkpoint if fresh start
    if fresh_start:
        clear_checkpoint()

    # Load checkpoint
    checkpoint = load_checkpoint()

    # Load questions
    questions = load_questions()

    if pilot_mode:
        questions = questions[:pilot_n]
        logger.info(f"PILOT MODE — running {pilot_n} questions")

    # ── Main Loop ───────────────────────
    for model_key, model_name in SLM_MODELS.items():

        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        # Get already completed question IDs
        completed_ids = set(
            checkpoint["completed_ids"][model_key]
        )

        # Filter remaining questions
        remaining = [
            q for q in questions
            if q["question_id"] not in completed_ids
        ]

        logger.info(
            f"Completed: {len(completed_ids)} | "
            f"Remaining: {len(remaining)}"
        )

        if not remaining:
            logger.info(f"All questions completed for {model_name}")
            continue

        # Progress bar
        pbar = tqdm(
            remaining,
            desc        = f"{model_name}",
            unit        = "q",
            initial     = len(completed_ids),
            total       = len(questions),
            file        = sys.stdout,
            colour      = "green",
        )

        for question in pbar:

            question_id = question["question_id"]

            try:
                # Process question
                result = process_question(model_key, question)

                # Save to checkpoint
                checkpoint["results"][model_key].append(result)
                checkpoint["completed_ids"][model_key].append(
                    question_id
                )
                save_checkpoint(checkpoint)

                # Update progress bar
                pbar.set_postfix({
                    "escalated": result["should_escalate"],
                    "correct":   result["final_correct"],
                    "conf":      f"{result['avg_confidence']:.1f}",
                })

            except Exception as e:
                logger.error(
                    f"Failed on {question_id} "
                    f"for {model_name}: {e}"
                )
                logger.info("Checkpoint saved — safe to restart")
                continue

        pbar.close()

    # ── Save Final Results ───────────────
    logger.info("\nSaving final results...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw results
    with open(RAW_RESULTS_FILE, "w") as f:
        json.dump(checkpoint["results"], f, indent=2)
    logger.info(f"Raw results saved to {RAW_RESULTS_FILE}")

    # ── Compute Aggregate Metrics ────────
    logger.info("\nComputing aggregate metrics...")

    all_metrics = {}

    for model_key, model_name in SLM_MODELS.items():

        raw = checkpoint["results"][model_key]
        if not raw:
            continue

        # Convert dicts back to QuestionResult objects
        results = [
            QuestionResult(**{
                k: v for k, v in r.items()
            })
            for r in raw
        ]

        all_metrics[model_name] = {}

        for qtype in ["all", "formula_based", "conceptual"]:
            m = aggregate_metrics(results, model_name, qtype)
            all_metrics[model_name][qtype] = {
                "escalation_rate":  m.escalation_rate,
                "miss_rate":        m.miss_rate,
                "containment_rate": m.containment_rate,
                "consistency_rate": m.consistency_rate,
                "cascade_accuracy": m.cascade_accuracy,
                "avg_gss":          m.avg_gss,
                "avg_evs":          m.avg_evs,
                "avg_eqs":          m.avg_eqs,
                "avg_aes":          m.avg_aes,
                "avg_slm_latency":  m.avg_slm_latency,
            }

    # Save metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Metrics saved to {METRICS_FILE}")

    # Print summary
    print("\n" + generate_summary_report(
        {k: {
            qt: type('M', (), v)()
            for qt, v in vm.items()
        } for k, vm in all_metrics.items()}
    ))

    logger.info("\nExperiment complete!")


# ─────────────────────────────────────────
# Entry Points
# ─────────────────────────────────────────

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="ICCCMLA Cascade Experiment"
    )
    parser.add_argument(
        "--pilot",
        action  = "store_true",
        help    = "Run pilot with 5 questions per model"
    )
    parser.add_argument(
        "--pilot-n",
        type    = int,
        default = 5,
        help    = "Number of questions in pilot mode"
    )
    parser.add_argument(
        "--fresh",
        action  = "store_true",
        help    = "Clear checkpoint and start fresh"
    )

    args = parser.parse_args()

    run_cascade(
        pilot_mode  = args.pilot,
        pilot_n     = args.pilot_n,
        fresh_start = args.fresh,
    )