"""
Sample and prepare 200 MMLU Physics questions.
Split into formula-based and conceptual categories.
"""

import json
import random
from src.utils.config import (
    RAW_DIR,
    PROCESSED_DIR,
    FORMULA_QUESTIONS_FILE,
    CONCEPTUAL_QUESTIONS_FILE,
    NUM_FORMULA_QUESTIONS,
    NUM_CONCEPTUAL_QUESTIONS,
    RANDOM_SEED,
)
from src.utils.logger import logger

# ─────────────────────────────────────────
# Subset categorisation
# ─────────────────────────────────────────

# Formula-based: calculation and numerical
FORMULA_SUBSETS = [
    "high_school_physics",
    "college_physics",
]

# Conceptual: understanding and explanation
CONCEPTUAL_SUBSETS = [
    "conceptual_physics",
    "astronomy",
]


def load_subset(subset: str) -> list:
    """Load raw questions for a subset."""
    raw_file = RAW_DIR / f"{subset}.json"
    with open(raw_file, "r") as f:
        return json.load(f)


def add_metadata(questions: list, question_type: str) -> list:
    """Add question type and ID metadata."""
    for i, q in enumerate(questions):
        q["question_type"] = question_type
        q["question_id"]   = f"{question_type}_{i+1:03d}"
    return questions


def prepare_dataset() -> tuple:
    """
    Sample and prepare 200 questions:
    - 100 formula-based (high school + college physics)
    - 100 conceptual (conceptual physics + astronomy)
    """

    random.seed(RANDOM_SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Formula-based questions ──────────────
    logger.info("Preparing formula-based questions...")

    formula_pool = []
    for subset in FORMULA_SUBSETS:
        questions = load_subset(subset)
        formula_pool.extend(questions)
        logger.info(f"  {subset}: {len(questions)} questions loaded")

    logger.info(f"  Total formula pool: {len(formula_pool)} questions")

    # Sample 100
    formula_sample = random.sample(formula_pool, NUM_FORMULA_QUESTIONS)
    formula_sample = add_metadata(formula_sample, "formula_based")
    logger.info(f"  Sampled: {len(formula_sample)} formula-based questions")

    # ── Conceptual questions ─────────────────
    logger.info("Preparing conceptual questions...")

    conceptual_pool = []
    for subset in CONCEPTUAL_SUBSETS:
        questions = load_subset(subset)
        conceptual_pool.extend(questions)
        logger.info(f"  {subset}: {len(questions)} questions loaded")

    logger.info(f"  Total conceptual pool: {len(conceptual_pool)} questions")

    # Sample 100
    conceptual_sample = random.sample(
        conceptual_pool,
        NUM_CONCEPTUAL_QUESTIONS
    )
    conceptual_sample = add_metadata(conceptual_sample, "conceptual")
    logger.info(f"  Sampled: {len(conceptual_sample)} conceptual questions")

    # ── Save to processed files ──────────────
    with open(FORMULA_QUESTIONS_FILE, "w") as f:
        json.dump(formula_sample, f, indent=2)
    logger.info(f"Formula questions saved to {FORMULA_QUESTIONS_FILE}")

    with open(CONCEPTUAL_QUESTIONS_FILE, "w") as f:
        json.dump(conceptual_sample, f, indent=2)
    logger.info(f"Conceptual questions saved to {CONCEPTUAL_QUESTIONS_FILE}")

    # ── Summary ──────────────────────────────
    logger.info("\nDataset preparation complete:")
    logger.info(f"  Formula-based : {len(formula_sample)} questions")
    logger.info(f"  Conceptual    : {len(conceptual_sample)} questions")
    logger.info(f"  Total         : {len(formula_sample) + len(conceptual_sample)} questions")

    return formula_sample, conceptual_sample


if __name__ == "__main__":
    formula, conceptual = prepare_dataset()

    # Show sample question from each type
    print("\n── Sample Formula Question ──")
    print(json.dumps(formula[0], indent=2))

    print("\n── Sample Conceptual Question ──")
    print(json.dumps(conceptual[0], indent=2))