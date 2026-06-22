"""
Metrics Module
Calculates all evaluation metrics for the cascade experiment.

Metrics:
- Escalation Rate (ER)
- Miss Rate (MR)
- Containment Rate (CR)
- Gap Severity Score (GSS)
- Escalation Value Score (EVS)
- Explanation Quality Score (EQS)
- Answer Efficiency Score (AES)
- Consistency Rate (CS)
- Cascade Accuracy (CA)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from src.utils.config import CONTAINMENT_THRESHOLD
from src.utils.logger import logger


# ─────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────

@dataclass
class QuestionResult:
    """Complete result for one question across all pipeline stages."""

    # Identity
    question_id:    str
    question_type:  str   # formula_based or conceptual
    model:          str

    # Ground truth
    correct_answer: str

    # SLM results
    slm_answer:       str
    slm_correct:      bool
    avg_confidence:   float
    consistent:       bool
    should_escalate:  bool
    escalation_reason: str
    slm_reasoning:    str
    slm_latency_ms:   float

    # Cloud results (None if not escalated)
    cloud_answer:     Optional[str]  = None
    cloud_correct:    Optional[bool] = None
    cloud_reasoning:  Optional[str]  = None
    cloud_latency_ms: Optional[float] = None

    # Judge scores
    gss_score:    Optional[int]   = None
    gss_label:    Optional[str]   = None
    gss_reasoning: Optional[str]  = None

    evs_score:    Optional[int]   = None
    evs_label:    Optional[str]   = None
    evs_reasoning: Optional[str]  = None
    containable:  Optional[bool]  = None

    eqs_score:    Optional[int]   = None
    eqs_reasoning: Optional[str]  = None

    # Computed metrics
    aes_score:    Optional[float] = None
    final_correct: Optional[bool] = None


@dataclass
class ModelMetrics:
    """Aggregate metrics for one model across all questions."""

    model:          str
    total:          int = 0
    question_type:  str = "all"

    # Escalation metrics
    escalated:      int = 0
    not_escalated:  int = 0

    # Correctness metrics
    slm_correct:    int = 0
    cloud_correct:  int = 0
    final_correct:  int = 0

    # Miss metrics
    misses:         int = 0  # not escalated + wrong

    # Containment metrics
    containable:    int = 0  # escalated + EVS <= threshold

    # Consistency metrics
    consistent:     int = 0

    # Score lists for averaging
    gss_scores:     List[int]   = field(default_factory=list)
    evs_scores:     List[int]   = field(default_factory=list)
    eqs_scores:     List[int]   = field(default_factory=list)
    aes_scores:     List[float] = field(default_factory=list)

    # Latency
    slm_latencies:   List[float] = field(default_factory=list)
    cloud_latencies: List[float] = field(default_factory=list)

    # ── Computed Properties ──────────────

    @property
    def escalation_rate(self) -> float:
        """ER: % questions escalated to cloud."""
        return (self.escalated / self.total * 100
                if self.total > 0 else 0.0)

    @property
    def miss_rate(self) -> float:
        """MR: % questions SLM answered confidently but wrongly."""
        return (self.misses / self.total * 100
                if self.total > 0 else 0.0)

    @property
    def containment_rate(self) -> float:
        """CR: % escalations that added zero/trivial value."""
        return (self.containable / self.escalated * 100
                if self.escalated > 0 else 0.0)

    @property
    def consistency_rate(self) -> float:
        """CS: % questions SLM answered same twice."""
        return (self.consistent / self.total * 100
                if self.total > 0 else 0.0)

    @property
    def slm_accuracy(self) -> float:
        """Local accuracy — SLM alone."""
        return (self.slm_correct / self.not_escalated * 100
                if self.not_escalated > 0 else 0.0)

    @property
    def cascade_accuracy(self) -> float:
        """CA: Full system accuracy."""
        return (self.final_correct / self.total * 100
                if self.total > 0 else 0.0)

    @property
    def avg_gss(self) -> float:
        """Average Gap Severity Score."""
        return (float(np.mean(self.gss_scores))
                if self.gss_scores else 0.0)

    @property
    def avg_evs(self) -> float:
        """Average Escalation Value Score."""
        return (float(np.mean(self.evs_scores))
                if self.evs_scores else 0.0)

    @property
    def avg_eqs(self) -> float:
        """Average Explanation Quality Score."""
        return (float(np.mean(self.eqs_scores))
                if self.eqs_scores else 0.0)

    @property
    def avg_aes(self) -> float:
        """Average Answer Efficiency Score."""
        return (float(np.mean(self.aes_scores))
                if self.aes_scores else 0.0)

    @property
    def avg_slm_latency(self) -> float:
        """Average SLM inference latency in ms."""
        return (float(np.mean(self.slm_latencies))
                if self.slm_latencies else 0.0)

    @property
    def avg_cloud_latency(self) -> float:
        """Average cloud latency in ms."""
        return (float(np.mean(self.cloud_latencies))
                if self.cloud_latencies else 0.0)


# ─────────────────────────────────────────
# Answer Efficiency Score
# ─────────────────────────────────────────

def calculate_aes(
    is_correct:  bool,
    explanation: str,
    max_words:   int = 150
) -> float:
    """
    Answer Efficiency Score (AES).
    Measures correctness per unit of explanation length.

    Rationale:
    - For advanced students, concise correct answers
      are more valuable than verbose correct answers
    - Wrong answers score 0 regardless of length
    - Short correct answers score highest

    Scale: 0.0 (wrong or very verbose) to 1.0 (correct + minimal)

    Args:
        is_correct:  whether the answer is correct
        explanation: the explanation/reasoning text
        max_words:   baseline word count (above this = minimum score)

    Returns:
        AES score between 0.0 and 1.0
    """

    if not is_correct:
        return 0.0

    word_count = len(explanation.split()) if explanation else 0

    if word_count == 0:
        return 1.0  # correct + no explanation = maximum efficiency

    # Normalize: shorter = higher score
    normalized = min(word_count, max_words) / max_words
    aes = 1.0 - normalized

    # Floor at 0.1 for correct answers regardless of length
    aes = max(aes, 0.1)

    return round(aes, 3)


# ─────────────────────────────────────────
# Per-Question Metric Computation
# ─────────────────────────────────────────

def compute_question_metrics(result: QuestionResult) -> QuestionResult:
    """
    Compute derived metrics for a single question result.
    Fills in AES and final_correct fields.
    """

    # Final correctness
    if result.should_escalate:
        result.final_correct = result.cloud_correct
    else:
        result.final_correct = result.slm_correct

    # AES — for the answer that was actually delivered to student
    if result.should_escalate and result.cloud_reasoning:
        # Cloud answered — score cloud efficiency
        result.aes_score = calculate_aes(
            is_correct  = result.cloud_correct or False,
            explanation = result.cloud_reasoning
        )
    elif not result.should_escalate:
        # SLM answered locally — score SLM efficiency
        result.aes_score = calculate_aes(
            is_correct  = result.slm_correct,
            explanation = result.slm_reasoning
        )

    return result


# ─────────────────────────────────────────
# Aggregate Metrics
# ─────────────────────────────────────────

def aggregate_metrics(
    results:       List[QuestionResult],
    model:         str,
    question_type: str = "all"
) -> ModelMetrics:
    """
    Aggregate all question results into model-level metrics.

    Args:
        results:       list of QuestionResult for one model
        model:         model name
        question_type: 'all', 'formula_based', or 'conceptual'

    Returns:
        ModelMetrics with all computed metrics
    """

    # Filter by question type if needed
    if question_type != "all":
        results = [
            r for r in results
            if r.question_type == question_type
        ]

    metrics = ModelMetrics(
        model         = model,
        total         = len(results),
        question_type = question_type
    )

    if metrics.total == 0:
        logger.warning(f"No results for {model} / {question_type}")
        return metrics

    for r in results:

        # Escalation
        if r.should_escalate:
            metrics.escalated += 1
        else:
            metrics.not_escalated += 1

        # Consistency
        if r.consistent:
            metrics.consistent += 1

        # SLM correctness (non-escalated only)
        if not r.should_escalate and r.slm_correct:
            metrics.slm_correct += 1

        # Miss rate (not escalated + wrong)
        if not r.should_escalate and not r.slm_correct:
            metrics.misses += 1

        # Cloud correctness
        if r.should_escalate and r.cloud_correct:
            metrics.cloud_correct += 1

        # Final correctness
        if r.final_correct:
            metrics.final_correct += 1

        # Containment
        if r.should_escalate and r.containable:
            metrics.containable += 1

        # GSS scores
        if r.gss_score is not None:
            metrics.gss_scores.append(r.gss_score)

        # EVS scores
        if r.evs_score is not None:
            metrics.evs_scores.append(r.evs_score)

        # EQS scores
        if r.eqs_score is not None:
            metrics.eqs_scores.append(r.eqs_score)

        # AES scores
        if r.aes_score is not None:
            metrics.aes_scores.append(r.aes_score)

        # Latencies
        metrics.slm_latencies.append(r.slm_latency_ms)
        if r.cloud_latency_ms:
            metrics.cloud_latencies.append(r.cloud_latency_ms)

    logger.info(
        f"\nMetrics [{model}] [{question_type}]:\n"
        f"  Total:             {metrics.total}\n"
        f"  Escalation Rate:   {metrics.escalation_rate:.1f}%\n"
        f"  Miss Rate:         {metrics.miss_rate:.1f}%\n"
        f"  Containment Rate:  {metrics.containment_rate:.1f}%\n"
        f"  Consistency Rate:  {metrics.consistency_rate:.1f}%\n"
        f"  Cascade Accuracy:  {metrics.cascade_accuracy:.1f}%\n"
        f"  Avg GSS:           {metrics.avg_gss:.2f}\n"
        f"  Avg EVS:           {metrics.avg_evs:.2f}\n"
        f"  Avg EQS:           {metrics.avg_eqs:.2f}\n"
        f"  Avg AES:           {metrics.avg_aes:.2f}\n"
        f"  Avg SLM Latency:   {metrics.avg_slm_latency:.0f}ms"
    )

    return metrics


# ─────────────────────────────────────────
# Summary Report
# ─────────────────────────────────────────

def generate_summary_report(
    all_metrics: dict  # {model_name: {type: ModelMetrics}}
) -> str:
    """
    Generate a formatted summary report for all models.
    Returns string suitable for printing or saving.
    """

    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT RESULTS SUMMARY")
    lines.append("=" * 80)

    header = (
        f"{'Model':<20} {'Type':<12} "
        f"{'ER%':>6} {'MR%':>6} {'CR%':>6} "
        f"{'CA%':>6} {'GSS':>6} {'EVS':>6} "
        f"{'EQS':>6} {'AES':>6}"
    )
    lines.append(header)
    lines.append("-" * 80)

    for model, type_metrics in all_metrics.items():
        for qtype, m in type_metrics.items():
            row = (
                f"{model:<20} {qtype:<12} "
                f"{m.escalation_rate:>6.1f} "
                f"{m.miss_rate:>6.1f} "
                f"{m.containment_rate:>6.1f} "
                f"{m.cascade_accuracy:>6.1f} "
                f"{m.avg_gss:>6.2f} "
                f"{m.avg_evs:>6.2f} "
                f"{m.avg_eqs:>6.2f} "
                f"{m.avg_aes:>6.3f}"
            )
            lines.append(row)
        lines.append("-" * 80)

    return "\n".join(lines)


# ─────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*50)
    print("Testing Metrics Module")
    print("="*50)

    # Test AES
    print("\n── AES Tests ──")

    # Wrong answer
    aes = calculate_aes(False, "This is wrong")
    print(f"Wrong answer:          AES = {aes}")

    # Correct + short
    aes = calculate_aes(True, "F=ma so a=5")
    print(f"Correct + short:       AES = {aes}")

    # Correct + medium
    aes = calculate_aes(
        True,
        "Using Newton's second law F=ma, "
        "we rearrange to get a=F/m=10/2=5 m/s²"
    )
    print(f"Correct + medium:      AES = {aes}")

    # Correct + very long
    aes = calculate_aes(
        True,
        " ".join(["word"] * 200)  # 200 words
    )
    print(f"Correct + very long:   AES = {aes}")

    # Test QuestionResult
    print("\n── QuestionResult Test ──")

    r = QuestionResult(
        question_id      = "test_001",
        question_type    = "formula_based",
        model            = "gemma3:1b",
        correct_answer   = "B",
        slm_answer       = "A",
        slm_correct      = False,
        avg_confidence   = 2.0,
        consistent       = True,
        should_escalate  = True,
        escalation_reason = "low_confidence",
        slm_reasoning    = "I think it is A",
        slm_latency_ms   = 4500.0,
        cloud_answer     = "B",
        cloud_correct    = True,
        cloud_reasoning  = "Using F=ma, a=10/2=5 m/s²",
        cloud_latency_ms = 3200.0,
        evs_score        = 3,
        evs_label        = "critical",
        containable      = False,
    )

    r = compute_question_metrics(r)
    print(f"Final correct: {r.final_correct}")
    print(f"AES score:     {r.aes_score}")

    # Test aggregate
    print("\n── Aggregate Metrics Test ──")

    metrics = aggregate_metrics([r], "gemma3:1b", "all")
    print(f"ER:  {metrics.escalation_rate:.1f}%")
    print(f"CA:  {metrics.cascade_accuracy:.1f}%")
    print(f"AES: {metrics.avg_aes:.3f}")