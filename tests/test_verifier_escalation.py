"""
tests/test_verifier_escalation.py — Day 14 unit tests.

5 targeted tests for the verifier and escalation paths.
All provider calls are mocked — no real API calls.

Run with:  pytest tests/test_verifier_escalation.py -v
"""

from __future__ import annotations
import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Isolated SQLite DB for each test."""
    import src.database as db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    return tmp_path / "test.db"


def _make_llm_response(model_id="llama-3.1-8b-instant", cost=0.000005,
                        text="Mock output.", input_tokens=50, output_tokens=20):
    from src.models import LLMResponse
    from src.config import calculate_cost_if_highest_quality
    return LLMResponse(
        output_text             = text,
        input_tokens            = input_tokens,
        output_tokens           = output_tokens,
        latency_ms              = 300.0,
        cost_usd                = cost,
        cost_if_highest_quality = calculate_cost_if_highest_quality(input_tokens, output_tokens),
        model_id                = model_id,
        provider                = "groq",
        timestamp               = datetime.now(timezone.utc),
    )


def _make_quality_score(cheap=4.0, expensive=4.5, correct=True, reason=None):
    from src.models import QualityScore
    return QualityScore(
        cheap_score     = cheap,
        expensive_score = expensive,
        quality_gap     = expensive - cheap,
        routing_correct = correct,
        failure_reason  = reason,
        judge_cost_usd  = 0.0001,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — is_routing_failure() logic
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsRoutingFailure:
    """Unit tests for the routing failure detection function."""

    def test_passes_when_gap_below_threshold_and_score_above_min(self):
        from src.verifier.escalation import is_routing_failure
        score = _make_quality_score(cheap=3.5, expensive=4.5)   # gap=1.0 < 1.5
        assert is_routing_failure(score, gap_threshold=1.5, min_cheap_score=3.0) is False

    def test_fails_when_gap_exceeds_threshold(self):
        from src.verifier.escalation import is_routing_failure
        score = _make_quality_score(cheap=2.0, expensive=4.0)   # gap=2.0 > 1.5
        assert is_routing_failure(score, gap_threshold=1.5, min_cheap_score=3.0) is True

    def test_fails_when_cheap_score_below_minimum(self):
        from src.verifier.escalation import is_routing_failure
        score = _make_quality_score(cheap=2.5, expensive=3.0)   # cheap < 3.0
        assert is_routing_failure(score, gap_threshold=1.5, min_cheap_score=3.0) is True

    def test_fails_when_both_conditions_trigger(self):
        from src.verifier.escalation import is_routing_failure
        score = _make_quality_score(cheap=1.0, expensive=5.0)   # gap=4.0, cheap<3.0
        assert is_routing_failure(score, gap_threshold=1.5, min_cheap_score=3.0) is True

    def test_boundary_exact_threshold(self):
        """Gap exactly equal to threshold should NOT trigger failure (strictly greater)."""
        from src.verifier.escalation import is_routing_failure
        score = _make_quality_score(cheap=3.0, expensive=4.5)   # gap=1.5, exactly at threshold
        # quality_gap > 1.5 is the condition, so 1.5 == 1.5 should NOT trigger
        assert is_routing_failure(score, gap_threshold=1.5, min_cheap_score=3.0) is False


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — escalate_if_needed: routing_correct path
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalateRoutingCorrect:
    """When routing is correct, escalated=False must be written to the DB."""

    def test_no_escalation_when_routing_correct(self, temp_db):
        import src.database as db
        from src.models import ClassifierResult

        # Insert a fake request row first
        response = _make_llm_response()
        classifier = ClassifierResult(tier=1, confidence=0.9,
                                      features={}, low_confidence=False)
        request_id = db.log_request(response, classifier,
                                    prompt_preview="Test prompt",
                                    output_preview="Test output")

        quality = _make_quality_score(cheap=4.0, expensive=4.5, correct=True)

        with patch("src.verifier.escalation.load_routing") as mock_routing, \
             patch("src.verifier.escalation.get_highest_quality_model") as mock_hq:
            mock_routing.return_value = {
                "quality":  {"gap_threshold": 1.5, "min_cheap_score": 3.0},
                "latency":  {"budget_ms": 10000, "escalation_reserve_ms": 2000},
            }
            mock_hq.return_value = MagicMock(
                model_id="llama-3.3-70b-versatile",
                display_name="Llama 3.3 70B",
            )

            from src.verifier.escalation import escalate_if_needed
            result = asyncio.run(escalate_if_needed(
                request_id      = request_id,
                prompt          = "Test prompt",
                cheap_output    = "Test output",
                quality_score   = quality,
                classified_tier = 1,
                elapsed_ms      = 500.0,
                original_model  = "llama-3.1-8b-instant",
            ))

        assert result.escalated is False
        assert result.reason == "routing_correct"

        # CRITICAL: escalated must be written as 0, not left NULL
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT escalated FROM requests WHERE id=?", (request_id,)
            ).fetchone()
        assert row is not None
        assert row["escalated"] == 0, (
            "escalated column is NULL — update_escalation was not called on routing_correct path"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — escalate_if_needed: routing failure + successful escalation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalateOnFailure:
    """When routing fails and budget allows, escalated=True must be written."""

    def test_escalation_fires_on_routing_failure(self, temp_db):
        import src.database as db
        from src.models import ClassifierResult

        response = _make_llm_response(cost=0.000005)
        classifier = ClassifierResult(tier=2, confidence=0.75,
                                      features={}, low_confidence=False)
        request_id = db.log_request(response, classifier,
                                    prompt_preview="Write a React component...",
                                    output_preview="Here is some output")

        # Quality failure: cheap=1.0 << expensive=4.5, gap=3.5 > 1.5
        bad_quality = _make_quality_score(cheap=1.0, expensive=4.5,
                                          correct=False, reason="Incomplete output")

        escalated_response = _make_llm_response(
            model_id="llama-3.3-70b-versatile",
            cost=0.0005,
            text="Better output from expensive model",
        )

        # ✅ FIXED: Patch the dispatcher where send_request is IMPORTED FROM
        with patch("src.verifier.escalation.load_routing") as mock_routing, \
             patch("src.verifier.escalation.get_highest_quality_model") as mock_hq, \
             patch("src.providers.dispatcher.send_request",
                   return_value=escalated_response):
            mock_routing.return_value = {
                "quality":  {"gap_threshold": 1.5, "min_cheap_score": 3.0},
                "latency":  {"budget_ms": 30000, "escalation_reserve_ms": 2000},
            }
            expensive_mc = MagicMock()
            expensive_mc.model_id     = "llama-3.3-70b-versatile"
            expensive_mc.display_name = "Llama 3.3 70B"
            mock_hq.return_value = expensive_mc

            from src.verifier.escalation import escalate_if_needed
            result = asyncio.run(escalate_if_needed(
                request_id      = request_id,
                prompt          = "Write a React component...",
                cheap_output    = "Here is some output",
                quality_score   = bad_quality,
                classified_tier = 2,
                elapsed_ms      = 1500.0,
                original_model  = "gemini-2.5-flash",
            ))

        assert result.escalated is True
        assert result.escalated_model == "llama-3.3-70b-versatile"
        assert result.original_model  == "gemini-2.5-flash"
        assert result.output == "Better output from expensive model"

        # DB must reflect escalated=1
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT escalated, escalated_model FROM requests WHERE id=?",
                (request_id,)
            ).fetchone()
        assert row["escalated"] == 1
        assert row["escalated_model"] == "llama-3.3-70b-versatile"

        # Routing failure must be logged for flywheel
        with db.get_connection() as conn:
            failures = conn.execute(
                "SELECT * FROM routing_failures WHERE request_id=?", (request_id,)
            ).fetchall()
        assert len(failures) == 1
        assert failures[0]["classified_tier"] == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — escalate_if_needed: latency budget exhausted
# ═══════════════════════════════════════════════════════════════════════════════

class TestEscalateLatencyBudget:
    """When budget is exhausted, failure is logged but escalation is skipped."""

    def test_no_escalation_when_budget_exhausted(self, temp_db):
        import src.database as db
        from src.models import ClassifierResult

        response = _make_llm_response()
        classifier = ClassifierResult(tier=3, confidence=0.55,
                                      features={}, low_confidence=True)
        request_id = db.log_request(response, classifier,
                                    prompt_preview="Complex reasoning task",
                                    output_preview="Short answer")

        bad_quality = _make_quality_score(cheap=1.5, expensive=4.8,
                                          correct=False, reason="Very incomplete")

        with patch("src.verifier.escalation.load_routing") as mock_routing, \
             patch("src.verifier.escalation.get_highest_quality_model") as mock_hq:
            mock_routing.return_value = {
                "quality":  {"gap_threshold": 1.5, "min_cheap_score": 3.0},
                "latency":  {"budget_ms": 10000, "escalation_reserve_ms": 2000},
            }
            mock_hq.return_value = MagicMock(model_id="llama-3.3-70b-versatile")

            from src.verifier.escalation import escalate_if_needed
            result = asyncio.run(escalate_if_needed(
                request_id      = request_id,
                prompt          = "Complex reasoning task",
                cheap_output    = "Short answer",
                quality_score   = bad_quality,
                classified_tier = 3,
                elapsed_ms      = 9000.0,  # only 1000ms left < 2000ms reserve
                original_model  = "llama-3.1-8b-instant",
            ))

        assert result.escalated is False
        assert result.reason == "latency_budget_exhausted"

        # Failure must still be logged for the flywheel even though no escalation
        with db.get_connection() as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM routing_failures WHERE request_id=?",
                (request_id,)
            ).fetchone()["n"]
        assert n == 1, "Routing failure must be logged even when budget is exhausted"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — verify_response: sentinel written on verification failure
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerifyResponseSentinel:
    """
    When the reference model call fails during verification,
    a sentinel quality_score of -1 must be written so the dashboard
    shows 'failed' instead of leaving quality_score=NULL forever.
    """

    def test_sentinel_written_when_reference_call_fails(self, temp_db):
        import src.database as db
        from src.models import ClassifierResult, ProviderError

        response = _make_llm_response()
        classifier = ClassifierResult(tier=1, confidence=0.92,
                                      features={}, low_confidence=False)
        request_id = db.log_request(response, classifier,
                                    prompt_preview="Simple extraction task",
                                    output_preview="output@email.com")

        # ✅ FIXED: Patch the dispatcher where send_request is IMPORTED FROM
        with patch("src.providers.dispatcher.send_request",
                   side_effect=ProviderError("llama-3.3-70b-versatile", "rate limited")), \
             patch("src.verifier.verifier.get_highest_quality_model") as mock_hq:
            mock_hq.return_value = MagicMock(model_id="llama-3.3-70b-versatile")

            from src.verifier.verifier import verify_response
            asyncio.run(verify_response(
                request_id         = request_id,
                prompt             = "Simple extraction task",
                cheap_output       = "output@email.com",
                cheap_model_id     = "llama-3.1-8b-instant",
                classified_tier    = 1,
                request_start_time = __import__("time").monotonic() - 0.5,
            ))

        # quality_score must be -1.0 (sentinel), not NULL
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT quality_score, verified_at FROM requests WHERE id=?",
                (request_id,)
            ).fetchone()

        assert row["quality_score"] is not None, (
            "quality_score is NULL — sentinel was not written after verification failure"
        )
        assert row["quality_score"] == -1.0, (
            f"Expected sentinel -1.0, got {row['quality_score']}"
        )
        assert row["verified_at"] is not None, "verified_at should be set alongside sentinel"