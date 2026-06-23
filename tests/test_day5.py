"""
Day 5 test suite — FastAPI skeleton + integration.
Run with: pytest tests/test_day5.py -v

Uses FastAPI TestClient with mocked providers — no real API calls.
"""

from __future__ import annotations
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mock_llm_response(cost=0.000028):
    from src.models import LLMResponse
    from src.config import load_registry, calculate_cost_if_highest_quality
    registry = load_registry()
    return LLMResponse(
        output_text             = "Mocked model output.",
        input_tokens            = 80,
        output_tokens           = 25,
        latency_ms              = 320.0,
        cost_usd                = cost,
        cost_if_highest_quality = calculate_cost_if_highest_quality(80, 25),
        model_id                = "llama-3.1-8b-instant",
        provider                = "groq",
        timestamp               = datetime.now(timezone.utc),
    )


def _mock_classifier(tier=1, conf=0.88):
    from src.models import ClassifierResult
    return ClassifierResult(
        tier=tier, confidence=conf,
        features={"token_count": 20.0},
        low_confidence=conf < 0.60,
    )


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    import src.database as db
    temp = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", temp)
    db.init_db()
    return temp


@pytest.fixture
def client(temp_db):
    from src.api.main import app
    from src.config import load_registry
    registry  = load_registry()
    low_model = registry["llama-3.1-8b-instant"]

    # Use side_effect to return a NEW response each time
    with patch("src.api.main.get_router") as mgr, \
         patch("src.api.main.send_request", side_effect=lambda *args, **kwargs: _mock_llm_response()), \
         patch("src.api.main._run_verification", new_callable=AsyncMock):

        mock_router = MagicMock()
        mock_router.route.return_value = (low_model, _mock_classifier())
        mock_router.get_routing_config.return_value = {
            "tier_1_model": "llama-3.1-8b-instant",
            "tier_2_model": "gemini-2.5-flash",
            "tier_3_model": "llama-3.3-70b-versatile",
            "fallback_model": "gemini-2.5-flash",
            "low_confidence_threshold": 0.60,
        }
        mock_router._registry = registry
        mgr.return_value = mock_router

        from fastapi.testclient import TestClient
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


def _post(client, content="Extract emails from: hello@test.com"):
    return client.post("/v1/completions", json={
        "messages": [{"role": "user", "content": content}],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# POST /v1/completions
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompletions:

    def test_200_ok(self, client):
        assert _post(client).status_code == 200

    def test_content_present(self, client):
        assert len(_post(client).json()["content"]) > 0

    def test_cost_metadata_present(self, client):
        data = _post(client).json()
        for f in ["cost_usd", "cost_if_highest_quality", "savings_usd", "savings_pct", "latency_ms"]:
            assert f in data

    def test_routing_metadata_present(self, client):
        data = _post(client).json()
        for f in ["tier", "confidence", "low_confidence", "model_used", "provider"]:
            assert f in data

    def test_request_id_is_uuid(self, client):
        rid = _post(client).json()["request_id"]
        assert len(rid) == 36

    def test_cost_if_highest_quality_nonzero(self, client):
        assert _post(client).json()["cost_if_highest_quality"] > 0

    def test_savings_pct_positive(self, client):
        assert _post(client).json()["savings_pct"] > 0

    def test_request_logged_to_db(self, client, temp_db):
        import src.database as db
        _post(client)
        assert db.get_summary_stats()["total_requests"] >= 1

    def test_cost_if_highest_quality_in_db(self, client, temp_db):
        import src.database as db
        _post(client)
        with db.get_connection() as conn:
            row = conn.execute(
                "SELECT cost_if_highest_quality FROM requests LIMIT 1"
            ).fetchone()
        assert row["cost_if_highest_quality"] is not None
        assert row["cost_if_highest_quality"] > 0

    def test_empty_messages_rejected(self, client):
        r = client.post("/v1/completions", json={"messages": []})
        assert r.status_code == 422

    def test_bad_temperature_rejected(self, client):
        r = client.post("/v1/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 5.0,
        })
        assert r.status_code == 422

    def test_provider_error_returns_502(self, temp_db):
        from src.api.main import app
        from src.config import load_registry
        from src.models import ProviderError
        registry = load_registry()

        with patch("src.api.main.get_router") as mgr, \
             patch("src.api.main.send_request",
                   side_effect=ProviderError("llama-3.1-8b-instant", "down")):
            mock_router = MagicMock()
            mock_router.route.return_value = (
                registry["llama-3.1-8b-instant"], _mock_classifier())
            mock_router._registry = registry
            mgr.return_value = mock_router
            from fastapi.testclient import TestClient
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.post("/v1/completions", json={
                    "messages": [{"role": "user", "content": "test"}]
                })
        assert r.status_code == 502


# ═══════════════════════════════════════════════════════════════════════════════
# GET /v1/models
# ═══════════════════════════════════════════════════════════════════════════════

class TestModels:

    def test_200_ok(self, client):
        assert client.get("/v1/models").status_code == 200

    def test_returns_3_models(self, client):
        assert len(client.get("/v1/models").json()) == 3

    def test_model_fields_present(self, client):
        m = client.get("/v1/models").json()[0]
        for f in ["registry_key", "model_id", "provider", "display_name",
                  "quality_tier", "cost_per_1k_input_usd"]:
            assert f in m


# ═══════════════════════════════════════════════════════════════════════════════
# GET /v1/stats
# ═══════════════════════════════════════════════════════════════════════════════

class TestStats:

    def test_200_ok(self, client):
        assert client.get("/v1/stats").status_code == 200

    def test_required_fields(self, client):
        data = client.get("/v1/stats").json()
        for f in ["total_requests", "total_cost_usd", "savings_pct",
                  "total_baseline_cost", "escalation_rate_pct"]:
            assert f in data, f"Missing: {f}"

    def test_total_baseline_cost_alias(self, client):
        """routing_test.py uses this key — must be present."""
        assert "total_baseline_cost" in client.get("/v1/stats").json()

    def test_savings_after_request(self, client, temp_db):
        import src.database as db
        _post(client)
        assert db.get_summary_stats()["total_requests"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Routing config
# ═══════════════════════════════════════════════════════════════════════════════

class TestRoutingConfig:

    def test_get_200(self, client):
        assert client.get("/v1/routing-config").status_code == 200

    def test_get_has_tier_keys(self, client):
        data = client.get("/v1/routing-config").json()
        for k in ["tier_1_model", "tier_2_model", "tier_3_model"]:
            assert k in data

    def test_put_effective_immediately(self, client):
        r = client.put("/v1/routing-config",
                       json={"tier_1_model": "gemini-2.5-flash"})
        assert r.status_code == 200
        assert r.json()["effective_immediately"] is True

    def test_put_empty_rejected(self, client):
        assert client.put("/v1/routing-config", json={}).status_code == 422

    def test_put_invalid_model_rejected(self, client):
        from src.api.main import app
        from src.config import load_registry
        registry = load_registry()

        with patch("src.api.main.get_router") as mgr:
            mock_router = MagicMock()
            mock_router.update_routing.side_effect = ValueError("not found in registry")
            mock_router.get_routing_config.return_value = {}
            mock_router._registry = registry
            mgr.return_value = mock_router
            from fastapi.testclient import TestClient
            from src.api.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                r = c.put("/v1/routing-config", json={"tier_1_model": "gpt-fake"})
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:

    def test_200_ok(self, client):
        assert client.get("/health").status_code == 200

    def test_has_status(self, client):
        assert "status" in client.get("/health").json()

    def test_has_providers(self, client):
        data = client.get("/health").json()
        assert "groq" in data["providers"]
        assert "google" in data["providers"]

    def test_has_db_ok(self, client):
        assert "db_ok" in client.get("/health").json()


# ═══════════════════════════════════════════════════════════════════════════════
# Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_three_requests_accumulate_savings(self, client, temp_db):
        import src.database as db
        for p in ["Extract emails.", "Reformat date.", "Yes or no?"]:
            response = _post(client, p)
            # Ensure each request succeeded
            assert response.status_code == 200
        stats = db.get_summary_stats()
        assert stats["total_requests"] == 3
        assert stats["savings_usd"] > 0

    def test_stats_api_matches_db(self, client, temp_db):
        import src.database as db
        response = _post(client)
        assert response.status_code == 200
        api  = client.get("/v1/stats").json()
        db_s = db.get_summary_stats()
        assert api["total_requests"] == db_s["total_requests"]

    def test_all_rows_have_cost_if_highest_quality(self, client, temp_db):
        import src.database as db
        for i in range(4):
            response = _post(client, f"Prompt {i}")
            assert response.status_code == 200
        with db.get_connection() as conn:
            nulls = conn.execute(
                "SELECT COUNT(*) AS n FROM requests WHERE cost_if_highest_quality IS NULL"
            ).fetchone()["n"]
        assert nulls == 0

    def test_unique_request_ids(self, client):
        ids = set()
        for _ in range(3):
            response = _post(client)
            assert response.status_code == 200
            ids.add(response.json()["request_id"])
        assert len(ids) == 3