"""
Day 1 test suite.
Run with: pytest tests/test_day1.py -v

Tests every piece of Day 1 work:
  - ModelConfig dataclass and cost calculations
  - Registry YAML loading
  - Routing YAML loading
  - LLMResponse cost helpers
  - SQLite schema init
  - DB log_request and read back
  - cost_if_gpt4o is never None (the cardinal rule)
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime

# ── Allow imports from src/ ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ModelConfig, LLMResponse, ClassifierResult, ProviderError
from src.config import (
    load_registry, load_routing,
    calculate_cost, calculate_cost_if_highest_quality,
    get_model_by_tier,
)
import src.database as db


# ═══════════════════════════════════════════════════════════════════════════════
# ModelConfig tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelConfig:

    def test_cost_for_tokens_basic(self):
        model = ModelConfig(
            provider="groq",
            model_id="llama-3.1-8b-instant",
            cost_per_1k_input_usd=0.000035,
            cost_per_1k_output_usd=0.000055,
            quality_tier="low",
            avg_latency_ms=500,
            display_name="Llama 3.1 8B",
        )
        # 1000 input + 500 output tokens
        cost = model.cost_for_tokens(1000, 500)
        expected = (1.0 * 0.000035) + (0.5 * 0.000055)
        assert abs(cost - expected) < 1e-10

    def test_cost_zero_tokens(self):
        model = ModelConfig(
            provider="groq",
            model_id="test",
            cost_per_1k_input_usd=0.001,
            cost_per_1k_output_usd=0.002,
            quality_tier="high",
            avg_latency_ms=300,
            display_name="Test",
        )
        assert model.cost_for_tokens(0, 0) == 0.0

    def test_all_three_registry_models_load(self):
        registry = load_registry()
        assert "llama-3.3-70b-versatile" in registry
        assert "gemini-2.5-flash" in registry
        assert "llama-3.1-8b-instant" in registry

    def test_registry_model_fields_correct(self):
        registry = load_registry()
        haiku = registry["llama-3.1-8b-instant"]
        assert haiku.provider == "groq"
        assert haiku.quality_tier == "low"
        assert haiku.cost_per_1k_input_usd == 0.000035
        assert haiku.cost_per_1k_output_usd == 0.000055
        assert haiku.avg_latency_ms == 500

    def test_registry_providers_are_correct(self):
        registry = load_registry()
        assert registry["llama-3.3-70b-versatile"].provider == "groq"
        assert registry["gemini-2.5-flash"].provider == "google"
        assert registry["llama-3.1-8b-instant"].provider == "groq"

    def test_registry_key_set_after_load(self):
        registry = load_registry()
        for key, config in registry.items():
            assert config.registry_key == key, \
                f"registry_key not set for {key}"

    def test_quality_tiers_valid(self):
        registry = load_registry()
        valid_tiers = {"high", "medium", "low"}
        for key, config in registry.items():
            assert config.quality_tier in valid_tiers, \
                f"{key} has invalid quality_tier: {config.quality_tier}"

    def test_model_to_dict(self):
        registry = load_registry()
        d = registry["gemini-2.5-flash"].to_dict()
        assert "provider" in d
        assert "cost_per_1k_input_usd" in d
        assert d["provider"] == "google"


# ═══════════════════════════════════════════════════════════════════════════════
# Cost calculation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCostCalculation:

    def test_calculate_cost_if_highest_quality_known_values(self):
        # 1000 input + 1000 output at  Llama 3.3 70B pricing
        # input: 1 * $0.005 = $0.005
        # output: 1 * $0.015 = $0.015
        # total: $0.020
        cost = calculate_cost_if_highest_quality(1000, 1000)
        assert abs(cost - 0.00111) < 1e-8

    def test_calculate_cost_if_highest_quality_zero(self):
        assert calculate_cost_if_highest_quality(0, 0) == 0.0

    def test_calculate_cost_if_highest_quality_output_heavy(self):
        # Output tokens are 3x more expensive in Llama 3.3 70B
        cost_input_only  = calculate_cost_if_highest_quality(1000, 0)
        cost_output_only = calculate_cost_if_highest_quality(0, 1000)
        assert cost_output_only > cost_input_only

    def test_llama_8b_is_cheaper_than_llama33_70b_baseline(self):
        """The whole point of the project — cheap model costs less."""
        registry = load_registry()
        model = registry["llama-3.1-8b-instant"]
        actual  = calculate_cost(model, 500, 200)
        baseline = calculate_cost_if_highest_quality(500, 200)
        assert actual < baseline

    def test_savings_are_significant_for_tier1_model(self):
        registry = load_registry()
        model = registry["llama-3.1-8b-instant"]
        actual   = calculate_cost(model, 500, 200)
        baseline = calculate_cost_if_highest_quality(500, 200)
        savings_pct = (1 - actual / baseline) * 100
        # Llama 3.1 8B should save at least 90% vs Llama 3.3 70B
        assert savings_pct > 90, \
            f"Expected >90% savings, got {savings_pct:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# Routing config tests
# ════════════════════════
    def test_routing_config_loads(self):
        routing = load_routing()
        assert "routing" in routing
        assert "quality" in routing
        assert "latency" in routing

    def test_all_tier_models_defined(self):
        routing = load_routing()
        r = routing["routing"]
        assert "tier_1_model" in r
        assert "tier_2_model" in r
        assert "tier_3_model" in r
        assert "fallback_model" in r

    def test_tier_models_exist_in_registry(self):
        routing = load_routing()
        registry = load_registry()
        r = routing["routing"]
        for key in ["tier_1_model", "tier_2_model", "tier_3_model", "fallback_model"]:
            model_key = r[key]
            assert model_key in registry, \
                f"{key}='{model_key}' not found in registry"

    def test_get_model_by_tier(self):
        routing = load_routing()
        assert get_model_by_tier(1, routing) == "llama-3.1-8b-instant"
        assert get_model_by_tier(2, routing) == "gemini-2.5-flash"
        assert get_model_by_tier(3, routing) == "llama-3.3-70b-versatile"

    def test_quality_thresholds_present(self):
        routing = load_routing()
        q = routing["quality"]
        assert "gap_threshold" in q
        assert "min_cheap_score" in q
        assert q["gap_threshold"] > 0
        assert q["min_cheap_score"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# LLMResponse tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMResponse:

    def _make_response(self, cost_usd=0.001, cost_if_highest_quality=0.010) -> LLMResponse:
        return LLMResponse(
            output_text="Test output",
            input_tokens=100,
            output_tokens=50,
            latency_ms=350.0,
            cost_usd=cost_usd,
            cost_if_highest_quality=cost_if_highest_quality,
            model_id="llama-3.1-8b-instant",
            provider="groq",
            timestamp=datetime.utcnow(),
        )

    def test_response_has_request_id(self):
        r = self._make_response()
        assert r.request_id is not None
        assert len(r.request_id) == 36  # UUID format

    def test_total_tokens(self):
        r = self._make_response()
        assert r.total_tokens == 150

    def test_savings_usd(self):
        r = self._make_response(cost_usd=0.001, cost_if_highest_quality=0.010)
        assert abs(r.savings_usd - 0.009) < 1e-9

    def test_savings_pct(self):
        r = self._make_response(cost_usd=0.001, cost_if_highest_quality=0.010)
        assert abs(r.savings_pct - 90.0) < 0.01

    def test_cost_if_highest_quality_never_none(self):
        """The cardinal rule — cost_if_highest_quality must always be set."""
        r = self._make_response()
        assert r.cost_if_highest_quality is not None
        assert r.cost_if_highest_quality > 0

    def test_to_dict_contains_savings(self):
        r = self._make_response()
        d = r.to_dict()
        assert "savings_usd" in d
        assert "savings_pct" in d
        assert "total_tokens" in d


# ═══════════════════════════════════════════════════════════════════════════════
# Database tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabase:

    @pytest.fixture(autouse=True)
    def use_temp_db(self, tmp_path, monkeypatch):
        """Redirect DB_PATH to a temp file for each test."""
        temp_db = tmp_path / "test_autopilot.db"
        monkeypatch.setattr(db, "DB_PATH", temp_db)
        db.init_db()

    def _make_response(self) -> LLMResponse:
        return LLMResponse(
            output_text="Hello from Llama",
            input_tokens=80,
            output_tokens=40,
            latency_ms=420.0,
            cost_usd=calculate_cost_if_highest_quality(80, 40) * 0.01,  # very cheap
            cost_if_highest_quality=calculate_cost_if_highest_quality(80, 40),
            model_id="llama-3.1-8b-instant",
            provider="groq",
            timestamp=datetime.utcnow(),
        )

    def _make_classifier(self) -> ClassifierResult:
        return ClassifierResult(
            tier=1,
            confidence=0.91,
            features={"token_count": 80, "is_extraction": True},
            low_confidence=False,
        )

    def test_init_db_creates_tables(self):
        conn = db.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "requests" in table_names
        assert "routing_failures" in table_names
        assert "retrain_log" in table_names
        assert "baseline_runs" in table_names

    def test_log_request_inserts_row(self):
        response   = self._make_response()
        classifier = self._make_classifier()
        request_id = db.log_request(
            response, classifier,
            prompt_preview="Extract the email from: john@test.com",
            output_preview="john@test.com",
        )
        assert request_id == response.request_id

        conn = db.get_connection()
        row = conn.execute(
            "SELECT * FROM requests WHERE id = ?", (request_id,)
        ).fetchone()

        assert row is not None
        assert row["routed_model"] == "llama-3.1-8b-instant"
        assert row["complexity_tier"] == 1
        assert row["cost_if_highest_quality"] is not None
        assert row["cost_if_highest_quality"] > 0

    def test_cost_if_highest_quality_always_populated(self):
        """The cardinal rule — verified in the DB layer."""
        response   = self._make_response()
        request_id = db.log_request(
            response, None,
            prompt_preview="Test prompt",
            output_preview="Test output",
        )
        conn = db.get_connection()
        row = conn.execute(
            "SELECT cost_if_highest_quality FROM requests WHERE id = ?",
            (request_id,)
        ).fetchone()
        assert row["cost_if_highest_quality"] is not None, \
            "FATAL: cost_if_highest_quality is NULL — savings metric will be broken"

    def test_summary_stats_returns_savings(self):
        # Insert 3 requests and check savings calc
        for _ in range(3):
            resp = self._make_response()
            db.log_request(resp, self._make_classifier(),
                           prompt_preview="test", output_preview="out")

        stats = db.get_summary_stats()
        assert stats["total_requests"] == 3
        assert stats["total_baseline_cost"] > stats["total_cost_usd"]
        assert stats["savings_pct"] > 0

    def test_get_recent_requests(self):
        for _ in range(5):
            resp = self._make_response()
            db.log_request(resp, self._make_classifier(),
                           prompt_preview="test", output_preview="out")

        rows = db.get_recent_requests(limit=3)
        assert len(rows) == 3

    def test_get_cost_timeseries(self):
        resp = self._make_response()
        db.log_request(resp, self._make_classifier(),
                       prompt_preview="test", output_preview="out")
        rows = db.get_cost_timeseries(days=7)
        assert len(rows) >= 1
        assert "cost_actual" in rows[0]
        assert "cost_baseline" in rows[0]

    def test_init_db_is_idempotent(self):
        """Calling init_db() twice must not raise or duplicate user tables."""
        db.init_db()
        db.init_db()
        conn = db.get_connection()
        # Filter out SQLite internal tables (sqlite_sequence etc.)
        tables = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
        names = {r["name"] for r in tables}
        # Exactly our 4 tables — no duplicates
        assert names == {"requests", "routing_failures",
                         "retrain_log", "baseline_runs"}

    def test_log_baseline_run(self):
        db.log_baseline_run(
            run_timestamp="2026-06-17T09:00:00",
            prompt_index=0,
            prompt_preview="Extract the name",
            model_key="llama-3.1-8b-instant",
            output_preview="John",
            input_tokens=20,
            output_tokens=5,
            cost_usd=0.000001,
            latency_ms=380.0,
        )
        conn = db.get_connection()
        row = conn.execute("SELECT * FROM baseline_runs").fetchone()
        assert row["model_key"] == "llama-3.1-8b-instant"
        assert row["cost_usd"] == 0.000001
