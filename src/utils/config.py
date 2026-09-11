import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ─────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────

# On-device SLMs (via Ollama)
SLM_MODELS = {
    "small":  os.getenv("SLM_SMALL",  "gemma3:1b"),
    "medium": os.getenv("SLM_MEDIUM", "gemma3:4b"),
    "large":  os.getenv("SLM_LARGE",  "phi3:mini"),
}

# Cloud Validator (Anthropic Claude Sonnet 4.5)
CLOUD_VALIDATOR_MODEL = os.getenv(
    "CLOUD_VALIDATOR_MODEL",
    "claude-sonnet-4-5-20250929"
)



# LLM Judge (Anthropic Claude Sonnet 4.5)
JUDGE_MODEL = os.getenv(
    "JUDGE_MODEL",
    "claude-sonnet-4-5-20250929"
)
# Thinking budgets
VALIDATOR_THINKING_BUDGET = int(
    os.getenv("VALIDATOR_THINKING_BUDGET", 3000)
)
JUDGE_THINKING_BUDGET = int(
    os.getenv("JUDGE_THINKING_BUDGET", 1024)
)

# ─────────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────────

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# Ollama runs locally — no API key needed
OLLAMA_BASE_URL = "http://localhost:11434"

# ─────────────────────────────────────────
# ESCALATION CONFIGURATION
# ─────────────────────────────────────────

# Verbal confidence threshold (1-5 scale)
# If SLM confidence <= threshold → escalate
ESCALATION_THRESHOLD = int(
    os.getenv("ESCALATION_THRESHOLD", 3)
)

# ─────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────

# MMLU Physics subsets to use
MMLU_SUBSETS = [
    "high_school_physics",
    "college_physics",
    "conceptual_physics",
    "astronomy",
]

# Question type labels
QUESTION_TYPES = {
    "formula":     "formula_based",
    "conceptual":  "conceptual",
}

# Sampling configuration
NUM_FORMULA_QUESTIONS     = int(os.getenv("FORMULA_QUESTIONS",    100))
NUM_CONCEPTUAL_QUESTIONS  = int(os.getenv("CONCEPTUAL_QUESTIONS", 100))
TOTAL_QUESTIONS           = NUM_FORMULA_QUESTIONS + NUM_CONCEPTUAL_QUESTIONS

# Random seed for reproducibility
RANDOM_SEED = 42

# ─────────────────────────────────────────
# EVALUATION CONFIGURATION
# ─────────────────────────────────────────

# Gap Severity Score scale (1-3)
GSS_LABELS = {
    1: "trivial",    # Minor error — student still benefits
    2: "moderate",   # Significant error — some understanding shown
    3: "critical",   # Completely wrong or misleading
}

# Escalation Value Score scale (0-3)
EVS_LABELS = {
    0: "zero",       # Cloud added no correctness value
    1: "trivial",    # Cloud marginally better — trivial gain
    2: "significant",# Cloud meaningfully more correct
    3: "critical",   # Cloud completely correct where SLM was wrong
}

# Containment threshold
# EVS <= this value → counted in Containment Rate
CONTAINMENT_THRESHOLD = 1

# ─────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────

import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.parent

DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = DATA_DIR / "results"
OUTPUTS_DIR   = BASE_DIR / "outputs"
LOGS_DIR      = OUTPUTS_DIR / "logs"
TABLES_DIR    = OUTPUTS_DIR / "tables"
FIGURES_DIR   = OUTPUTS_DIR / "figures"

# Processed dataset files
FORMULA_QUESTIONS_FILE    = PROCESSED_DIR / "formula_questions.json"
CONCEPTUAL_QUESTIONS_FILE = PROCESSED_DIR / "conceptual_questions.json"

# Results files
RAW_RESULTS_FILE      = RESULTS_DIR / "raw_results.json"
METRICS_SUMMARY_FILE  = RESULTS_DIR / "metrics_summary.csv"

# ─────────────────────────────────────────
# LOGGING CONFIGURATION
# ─────────────────────────────────────────

LOG_LEVEL  = "INFO"
LOG_FILE   = LOGS_DIR / "experiment.log"

# ─────────────────────────────────────────
# EXPERIMENT METADATA
# ─────────────────────────────────────────

EXPERIMENT_NAME = "ICCCMLA-cascaded-slm-llm-physics"
PAPER_TITLE     = (
    "Towards On-Device STEM Tutoring: Measuring Implicit "
    "Escalation in Cascaded SLM-LLM Inference for Physics QA"
)
CONFERENCE      = "ICCCMLA 2026"
HARDWARE        = "Apple M1 (Metal GPU via Ollama)"
DATASET         = "MMLU-Physics"