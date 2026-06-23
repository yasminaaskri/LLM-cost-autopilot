"""
Async verifier — orchestrates the full quality verification pipeline.

Called as a FastAPI BackgroundTask immediately after the user
receives their response. Runs entirely after response delivery
so it never adds to the user-facing latency.

Pipeline per request:
  1. Call expensive model on the same prompt (get reference output)
  2. Call judge.score_quality(prompt, cheap_output, expensive_output)
  3. Call escalate_if_needed(...)
  4. All DB writes happen inside judge / escalation modules

The verify_response() function is the single entry point.
The API layer calls it like:
    background_tasks.add_task(
        verify_response,
        request_id, prompt, cheap_output, cheap_model_id, classified_tier, start_time
    )
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime

from src.config import load_registry, get_highest_quality_model
from src.database import update_quality_score
from src.models import ModelConfig, QualityScore
from src.verifier.judge import score_quality
from src.verifier.escalation import escalate_if_needed

logger = logging.getLogger(__name__)

# ── Singleton: expensive reference model ──────────────────────────────────────
_expensive_model: ModelConfig | None = None


def _get_expensive_model() -> ModelConfig:
    global _expensive_model
    if _expensive_model is None:
        _expensive_model = get_highest_quality_model()
    return _expensive_model


async def verify_response(
    request_id: str,
    prompt: str,
    cheap_output: str,
    cheap_model_id: str,          # FIX 3: new param — the model that served the user
    classified_tier: int,
    request_start_time: float,    # time.monotonic() from before the original API call
) -> None:
    """
    Full async verification pipeline for a single completed request.

    This function is designed to be called via FastAPI BackgroundTasks.
    Exceptions are handled in two layers:
      - Step 1+2 failures (reference call or judge): write a sentinel
        quality score of -1.0 so the dashboard shows the row as
        unverified rather than leaving quality_score=NULL forever.
      - Step 3 (escalation): its own internal exception handling.
    A verification failure must never crash the main application.

    Args:
        request_id:         UUID from the original requests DB row.
        prompt:             The original user prompt.
        cheap_output:       The cheap model's output (already sent to user).
        cheap_model_id:     model_id of the model that served the user.
        classified_tier:    The tier the classifier assigned (for failure logging).
        request_start_time: time.monotonic() stamp from before the original call.
    """
    verify_start = time.monotonic()

    try:
        # Step 1: Get reference output from expensive model
        from src.providers.dispatcher import send_request
        expensive_model    = _get_expensive_model()
        expensive_response = send_request(prompt, expensive_model)
        expensive_output   = expensive_response.output_text

        # Step 2: Judge both outputs
        quality_score = score_quality(
            original_prompt  = prompt,
            cheap_output     = cheap_output,
            expensive_output = expensive_output,
        )

        logger.info(
            "Verified request %s: cheap=%.1f expensive=%.1f gap=%.1f correct=%s",
            request_id[:8],
            quality_score.cheap_score,
            quality_score.expensive_score,
            quality_score.quality_gap,
            quality_score.routing_correct,
        )

        # Step 3: Escalate if needed
        # FIX 3: pass cheap_model_id (the model that served the user),
        # not expensive_model.model_id (the reference model).
        # The old code always wrote "" or the reference model's id to
        # original_model, making traceability impossible.
        elapsed_ms = (time.monotonic() - request_start_time) * 1000
        await escalate_if_needed(
            request_id      = request_id,
            prompt          = prompt,
            cheap_output    = cheap_output,
            quality_score   = quality_score,
            classified_tier = classified_tier,
            elapsed_ms      = elapsed_ms,
            original_model  = cheap_model_id,   # ← was expensive_model.model_id
        )

    except Exception as e:
        # FIX 3 (continued): the old broad except block silently swallowed
        # errors here and never called escalate_if_needed at all, leaving
        # escalated=NULL in the DB for any request where the reference call
        # or judge call failed. Now we log the error clearly and write a
        # sentinel so the row is visibly "failed to verify" on the dashboard.
        logger.error(
            "Verification failed for request %s: %s",
            request_id[:8], e, exc_info=True,
        )
        # Write a sentinel quality score so the dashboard can distinguish
        # "not yet verified" (NULL) from "verification attempted but failed" (-1)
        try:
            sentinel = QualityScore(
                cheap_score     = -1.0,
                expensive_score = -1.0,
                quality_gap     = 0.0,
                routing_correct = False,
                failure_reason  = f"verification_error: {type(e).__name__}: {e}",
                judge_cost_usd  = 0.0,
            )
            update_quality_score(request_id, sentinel)
        except Exception as db_err:
            logger.error(
                "Failed to write sentinel quality score for %s: %s",
                request_id[:8], db_err,
            )

    verify_elapsed = (time.monotonic() - verify_start) * 1000
    logger.debug("Verification completed in %.0fms for request %s",
                 verify_elapsed, request_id[:8])


def verify_response_sync(
    request_id: str,
    prompt: str,
    cheap_output: str,
    cheap_model_id: str,
    classified_tier: int,
    request_start_time: float,
) -> None:
    """
    Synchronous wrapper around verify_response for use in non-async contexts
    (e.g. the background worker process, tests).
    """
    asyncio.run(verify_response(
        request_id         = request_id,
        prompt             = prompt,
        cheap_output       = cheap_output,
        cheap_model_id     = cheap_model_id,
        classified_tier    = classified_tier,
        request_start_time = request_start_time,
    ))
