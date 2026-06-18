"""
Escalation engine.

After the async verifier detects a routing failure, the escalation
engine decides whether to re-run the request with the highest-quality
model — and if the latency budget allows, returns the better output.

The escalation chain:
  verifier detects failure
    → escalate_if_needed()
      → checks latency budget
      → if budget allows: calls expensive model, returns better output
      → logs escalation event to DB
      → logs routing failure to DB (for flywheel)
      → returns EscalationResult

Latency budget:
  Total budget (from routing.yaml): 10 000 ms
  Reserve for escalation call:       2 000 ms
  If elapsed > (budget - reserve): skip escalation, just log failure.
"""

from __future__ import annotations
import logging
import time
from typing import Optional

from src.config import load_registry, load_routing, get_highest_quality_model
from src.database import log_routing_failure, update_escalation, update_quality_score
from src.models import EscalationResult, QualityScore, ModelConfig

logger = logging.getLogger(__name__)

# ── Singleton: expensive model config ─────────────────────────────────────────
_expensive_model: Optional[ModelConfig] = None


def _get_expensive_model() -> ModelConfig:
    global _expensive_model
    if _expensive_model is None:
        _expensive_model = get_highest_quality_model(load_registry())
    return _expensive_model


def _get_thresholds() -> tuple[float, float, int, int]:
    """Load quality + latency thresholds from routing.yaml."""
    routing             = load_routing()
    gap_threshold       = routing["quality"]["gap_threshold"]
    min_cheap_score     = routing["quality"]["min_cheap_score"]
    budget_ms           = routing["latency"]["budget_ms"]
    escalation_reserve  = routing["latency"]["escalation_reserve_ms"]
    return gap_threshold, min_cheap_score, budget_ms, escalation_reserve


def is_routing_failure(quality_score: QualityScore,
                       gap_threshold: float,
                       min_cheap_score: float) -> bool:
    """
    Return True if the routing decision was bad.
    Two failure conditions (either triggers):
      1. quality_gap > gap_threshold  (cheap model significantly worse)
      2. cheap_score < min_cheap_score (cheap model flat-out bad)
    """
    return (
        quality_score.quality_gap > gap_threshold or
        quality_score.cheap_score < min_cheap_score
    )


async def escalate_if_needed(
    request_id: str,
    prompt: str,
    cheap_output: str,
    quality_score: QualityScore,
    classified_tier: int,
    elapsed_ms: float,
) -> EscalationResult:
    """
    Decide whether to escalate and run the expensive model.

    Called by the background verifier task after scoring is complete.

    Args:
        request_id:      UUID of the original request row in the DB.
        prompt:          The original user prompt.
        cheap_output:    The cheap model's output (already returned to user).
        quality_score:   Scores from the judge.
        classified_tier: What tier the classifier assigned.
        elapsed_ms:      Time already spent (API call + verification).

    Returns:
        EscalationResult — always returned, escalated=True only if we
        re-ran the expensive model within budget.
    """
    gap_threshold, min_cheap_score, budget_ms, reserve_ms = _get_thresholds()

    # Always update quality score in DB regardless of escalation decision
    update_quality_score(request_id, quality_score)

    # Case 1: routing was correct — nothing to do
    if not is_routing_failure(quality_score, gap_threshold, min_cheap_score):
        return EscalationResult(
            escalated      = False,
            original_model = "",
            escalated_model= None,
            cost_delta_usd = 0.0,
            output         = None,
            reason         = "routing_correct",
        )

    # Always log the failure for the flywheel, regardless of escalation
    log_routing_failure(request_id, prompt, classified_tier, quality_score)

    # Case 2: routing failed but latency budget exhausted
    remaining_ms = budget_ms - elapsed_ms
    if remaining_ms < reserve_ms:
        logger.warning(
            "Routing failure detected but latency budget exhausted "
            "(remaining=%.0fms < reserve=%dms). Logged for retrain only.",
            remaining_ms, reserve_ms,
        )
        result = EscalationResult(
            escalated      = False,
            original_model = "",
            escalated_model= None,
            cost_delta_usd = 0.0,
            output         = None,
            reason         = "latency_budget_exhausted",
        )
        update_escalation(request_id, result)
        return result

    # Case 3: escalate — re-run with expensive model
    from src.providers.dispatcher import send_request
    expensive_model = _get_expensive_model()

    logger.info(
        "Escalating request %s: gap=%.1f cheap_score=%.1f → %s",
        request_id[:8], quality_score.quality_gap,
        quality_score.cheap_score, expensive_model.display_name,
    )

    try:
        escalated_response = send_request(prompt, expensive_model)
        result = EscalationResult(
            escalated       = True,
            original_model  = "",       # caller fills this in
            escalated_model = expensive_model.model_id,
            cost_delta_usd  = escalated_response.cost_usd,
            output          = escalated_response.output_text,
            reason          = (
                f"quality_gap={quality_score.quality_gap:.1f} "
                f"cheap_score={quality_score.cheap_score:.1f}"
            ),
        )
    except Exception as e:
        logger.error("Escalation API call failed: %s", e)
        result = EscalationResult(
            escalated      = False,
            original_model = "",
            escalated_model= None,
            cost_delta_usd = 0.0,
            output         = None,
            reason         = f"escalation_api_failed: {e}",
        )

    update_escalation(request_id, result)
    return result