"""
Download and explore MMLU Physics subsets from HuggingFace.
"""

import json
from datasets import load_dataset
from src.utils.config import MMLU_SUBSETS, RAW_DIR
from src.utils.logger import logger


def download_mmlu_physics() -> dict:
    """
    Download all MMLU physics subsets from HuggingFace.
    Saves raw data to data/raw/
    Returns dict of all questions by subset.
    """

    # Create raw directory if needed
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_data = {}

    for subset in MMLU_SUBSETS:
        logger.info(f"Downloading MMLU subset: {subset}")

        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(
                "cais/mmlu",
                subset,
                trust_remote_code=True
            )

            # Combine all splits (train, validation, test)
            questions = []

            for split in ["test", "validation", "dev"]:
                if split in dataset:
                    for item in dataset[split]:
                        questions.append({
                            "subset":   subset,
                            "split":    split,
                            "question": item["question"],
                            "choices":  item["choices"],
                            "answer":   int(item["answer"]),
                            "answer_text": item["choices"][int(item["answer"])]
                        })

            all_data[subset] = questions
            logger.info(f"  → {len(questions)} questions loaded")

            # Save raw subset to file
            raw_file = RAW_DIR / f"{subset}.json"
            with open(raw_file, "w") as f:
                json.dump(questions, f, indent=2)
            logger.info(f"  → Saved to {raw_file}")

        except Exception as e:
            logger.error(f"Failed to download {subset}: {e}")
            raise

    # Summary
    total = sum(len(v) for v in all_data.values())
    logger.info(f"\nDownload complete — {total} total questions across {len(MMLU_SUBSETS)} subsets")

    for subset, questions in all_data.items():
        logger.info(f"  {subset}: {len(questions)} questions")

    return all_data


if __name__ == "__main__":
    data = download_mmlu_physics()