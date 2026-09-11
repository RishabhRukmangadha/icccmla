# Towards On-Device STEM Tutoring: Measuring Implicit Escalation in Cascaded SLM–LLM Inference for Physics Question Answering

**ICCCMLA 2026 — 8th IEEE International Conference on Cybernetics, Cognition and Machine Learning Applications**

> Rukmangadha Peddapalli Venkatappa (Digit88 Technologies, India)  
> Rishabh Rukmangadha (Independent Researcher, India)

---

## Overview

This repository contains the complete experiment code, dataset, raw results, and human evaluation data for the ICCCMLA 2026 paper.

We propose a cascaded SLM→LLM architecture for physics question answering where on-device Small Language Models (SLMs) serve as the primary inference layer and implicitly escalate uncertain queries to a Claude Sonnet 4.5 cloud validator. We evaluate three SLMs on MMLU-Physics and introduce five novel metrics including **Containment Rate** — the proportion of escalations that added zero or negligible correctness improvement.

---

## Key Findings

| Model | Escalation Rate | Miss Rate | Containment Rate |
|-------|----------------|-----------|-----------------|
| Gemma 3 1B | 89.0% | 8.0% | 23.6% |
| Gemma 3 4B | 0.0% | 55.5% | — |
| Phi-3 Mini | 14.5% | 37.5% | 24.1% |

- **Size ≠ Safety** — Gemma 3 4B (largest) exhibited the most dangerous profile with 55.5% miss rate and zero escalations due to systematic overconfidence
- **Hybrid trigger validated** — 18 of 29 Phi-3 Mini escalations were triggered by inconsistency, not confidence — these would have been missed by confidence-only triggers
- **~76/24 ratio is stable** — approximately 24% of escalations are consistently containable across models (χ²(1) = 0.004, p = 0.949)

---

## Repository Structure

```
icccmla/
├── src/
│   ├── dataset/
│   │   ├── download_mmlu.py          # Download MMLU Physics from HuggingFace
│   │   └── prepare_dataset.py        # Sample 200 questions (100 formula + 100 conceptual)
│   ├── models/
│   │   ├── slm_inference.py          # Ollama SLM inference + hybrid escalation trigger
│   │   ├── cloud_validator.py        # Claude Sonnet 4.5 cloud validator
│   │   └── llm_judge.py              # Claude Sonnet 4.5 LLM judge (GSS, EVS, EQS)
│   ├── pipeline/
│   │   └── cascade.py                # Main experiment pipeline with checkpointing
│   ├── evaluation/
│   │   ├── metrics.py                # All evaluation metrics (ER, MR, CR, AES etc.)
│   │   └── kappa_analysis.py         # Cohen's Kappa inter-rater reliability
│   └── utils/
│       ├── config.py                 # All configuration and constants
│       └── logger.py                 # Logging setup
├── data/
│   ├── processed/
│   │   ├── formula_questions.json    # 100 formula-based MMLU-Physics questions
│   │   └── conceptual_questions.json # 100 conceptual MMLU-Physics questions
│   ├── results/
│   │   ├── raw_results.json          # All 600 experiment records (200 × 3 models)
│   │   ├── metrics_summary.json      # Aggregate metrics per model
│   │   └── checkpoint.json           # Experiment checkpoint
│   └── human_evaluation/
│       ├── evaluation_instructions.csv # Scoring rubric provided to human evaluators
│       ├── human_judge_T1.xlsx        # Teacher 1 scores (GSS, EVS, EQS)
│       ├── human_judge_T2.xlsx        # Teacher 2 scores
│       └── human_judge_T3.xlsx        # Teacher 3 scores (Kappa validation)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running
- Anthropic API key

### Installation

```bash
# Clone the repository
git clone https://github.com/RishabhRukmangadha/icccmla.git
cd icccmla

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Pull SLM Models via Ollama

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
ollama pull phi3:mini
```

---

## Running the Experiment

### Step 1 — Download MMLU dataset

```bash
python -m src.dataset.download_mmlu
```

Downloads 4 MMLU Physics subsets from HuggingFace (`cais/mmlu`):
- `high_school_physics`
- `college_physics`
- `conceptual_physics`
- `astronomy`

### Step 2 — Prepare dataset

```bash
python -m src.dataset.prepare_dataset
```

Samples 100 formula-based + 100 conceptual questions with seed 42 for reproducibility.

### Step 3 — Run cascade experiment

```bash
# Recommended: prevent Mac sleep during long experiment
caffeinate -i python -m src.pipeline.cascade

# Pilot mode (5 questions per model — for testing)
python -m src.pipeline.cascade --pilot

# Fresh start (clears checkpoint)
python -m src.pipeline.cascade --fresh
```

⚠️ **Cost warning:** The full experiment makes ~800+ Claude Sonnet 4.5 API calls. Estimated cost: USD 15-25 depending on question complexity.

### Step 4 — Compute Kappa scores

```bash
python src/evaluation/kappa_analysis.py
```

Reproduces the inter-rater reliability scores reported in the paper.

---

## Experiment Configuration

All parameters are in `src/utils/config.py` and can be overridden via `.env`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SLM_SMALL` | `gemma3:1b` | Small SLM |
| `SLM_MEDIUM` | `gemma3:4b` | Medium SLM |
| `SLM_LARGE` | `phi3:mini` | Large SLM |
| `CLOUD_VALIDATOR_MODEL` | `claude-sonnet-4-5-20250929` | Cloud validator |
| `JUDGE_MODEL` | `claude-sonnet-4-5-20250929` | LLM judge |
| `ESCALATION_THRESHOLD` | `3` | Confidence threshold τ (1-5 scale) |
| `VALIDATOR_THINKING_BUDGET` | `3000` | Extended thinking tokens |
| `JUDGE_THINKING_BUDGET` | `1024` | Judge thinking tokens |
| `NUM_FORMULA_QUESTIONS` | `100` | Formula-based questions |
| `NUM_CONCEPTUAL_QUESTIONS` | `100` | Conceptual questions |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## Escalation Logic

```
For each question, the SLM is queried twice:

IF run₁.answer ≠ run₂.answer  → escalate (inconsistent)
ELSE IF avg(confidence) ≤ 3   → escalate (low confidence)
ELSE                           → answer locally
```

For consistent responses, confidence is averaged across both runs. For inconsistent responses, confidence is set to minimum (1/5), guaranteeing escalation.

---

## Metrics

| Metric | Definition |
|--------|-----------|
| Escalation Rate (ER) | % queries escalated to cloud |
| Miss Rate (MR) | % SLM confidently wrong without escalating |
| Containment Rate (CR) | % escalations with EVS ≤ 1 (wasted) |
| Gap Severity Score (GSS) | Judge score 1-3: severity of missed answers |
| Escalation Value Score (EVS) | Judge score 0-3: value added by escalation |
| Explanation Quality Score (EQS) | Judge score 1-5: explanation depth (conceptual only) |
| Answer Efficiency Score (AES) | Correctness per unit explanation length |
| Cascade Accuracy (CA) | Full system accuracy |
| Consistency Rate (CS) | % questions answered same across both runs |

---

## Human Evaluation

Three human physics educators independently scored a 30-question sample on GSS, EVS, and EQS to validate the Claude Sonnet 4.5 judge scores.

**Inter-rater agreement (Cohen's Kappa, linear weighted):**

| Metric | Human-Human κ | Human-Claude κ |
|--------|--------------|----------------|
| GSS | 0.672 (Substantial) | 0.569 (Moderate) |
| EVS | 0.673 (Substantial) | 0.449 (Moderate) |
| EQS | 0.667 (Substantial) | 0.032 (Slight) |

To reproduce:
```bash
python src/evaluation/kappa_analysis.py
```

See `data/human_evaluation/evaluation_instructions.csv` for the scoring rubric provided to evaluators.

---

## Statistical Tests

Key findings are statistically validated:

| Test | Result |
|------|--------|
| ER differences across models | Fisher's exact, p < 0.001 |
| MR: Gemma 4B vs Phi-3 Mini | Fisher's exact, OR = 2.08, p < 0.001 |
| 76/24 ratio consistency | χ²(1) = 0.004, p = 0.949, V = 0.004 |
| Formula vs conceptual MR (Gemma 4B) | Fisher's exact, OR = 4.02, p < 0.001 |
| Formula vs conceptual MR (Phi-3 Mini) | Fisher's exact, OR = 2.50, p = 0.003 |

---

## Hardware

All SLM inference was conducted on Apple M1 (MacBook) via Ollama using Q4_K_M quantization. This simulates edge deployment — not a physical mobile device.

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{venkatappa2026stemtutoring,
  title     = {Towards On-Device STEM Tutoring: Measuring Implicit Escalation 
               in Cascaded SLM--LLM Inference for Physics Question Answering},
  author    = {Venkatappa, Rukmangadha Peddapalli and Rukmangadha, Rishabh},
  booktitle = {2026 IEEE 8th International Conference on Cybernetics, Cognition 
               and Machine Learning Applications (ICCCMLA)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## Acknowledgements

Three human physics educators provided independent evaluation of judge scores. Claude Sonnet 4.5 (Anthropic) was used as cloud validator, LLM judge, and for language refinement and formatting assistance in preparing the manuscript.

---

## License

Code: MIT License  
Data: MMLU dataset is subject to its original license ([cais/mmlu](https://huggingface.co/datasets/cais/mmlu))
