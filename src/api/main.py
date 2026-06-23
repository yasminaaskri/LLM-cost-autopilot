
"""
FastAPI application — Day 5.

Endpoints:
  POST /v1/completions        Main routing endpoint
  GET  /v1/models             List all models in registry
  GET  /v1/stats              Cost savings summary
  PUT  /v1/routing-config     Hot-reload tier→model mapping
  GET  /v1/routing-config     Current routing config
  GET  /v1/admin/model-info   Classifier metadata
  POST /v1/admin/retrain      Trigger flywheel retrain
  GET  /health                Provider + DB health check

Design decisions:
  - BackgroundTasks for async verification: user gets response immediately,
    verifier runs after. This is the architectural contract — never block.
  - Global exception handler: ProviderError never crashes the API.
  - CORS enabled for the Streamlit dashboard on a different port.
  - Lifespan context: DB init and model warmup happen at startup, not
    on first request (eliminates first-request latency spike).
"""

from __future__ import annotations
import logging
import os
import time
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    CompletionRequest, CompletionResponse,
    ModelInfo, StatsResponse,
    RoutingConfigUpdate, RoutingConfigResponse,
    ClassifierInfoResponse, RetrainResponse,
    HealthResponse,
)
from src.config import load_registry
from src.database import init_db, log_request, get_summary_stats, get_connection
from src.models import ProviderError
from src.providers.dispatcher import send_request
from src.router.router import get_router, reset_router

ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: startup + shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run at startup: init DB, warm up router and classifier."""
    logger.info("Starting LLM Cost Autopilot API…")
    init_db()

    # Warm up router singleton (loads registry + routing.yaml)
    try:
        router = get_router()
        logger.info("Router ready — %d models loaded", len(router._registry))
    except Exception as e:
        logger.warning("Router warm-up failed: %s", e)

    # Warm up classifier (loads pkl into memory)
    try:
        from src.classifier.predict import get_model_info
        info = get_model_info()
        logger.info("Classifier ready: %s (%.1f%% accuracy)",
                    info["model_name"], (info["test_accuracy"] or 0) * 100)
    except Exception as e:
        logger.warning("Classifier warm-up failed (train first): %s", e)

    yield   # app runs here

    logger.info("Shutting down LLM Cost Autopilot API")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "LLM Cost Autopilot",
    description = "Intelligent routing layer that sends each prompt to the cheapest capable model.",
    version     = "0.1.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # tighten in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Global exception handlers ──────────────────────────────────────────────────

@app.exception_handler(ProviderError)
async def provider_error_handler(request, exc: ProviderError):
    logger.error("ProviderError: %s", exc)
    return JSONResponse(
        status_code = 502,
        content     = {
            "error":    "provider_error",
            "model_id": exc.model_id,
            "message":  exc.message,
        },
    )

@app.exception_handler(Exception)
async def generic_error_handler(request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code = 500,
        content     = {"error": "internal_error", "message": str(exc)},
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_prompt(messages: list) -> str:
    """Extract the last user message as the prompt string."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return messages[-1].content


async def _run_verification(
    request_id: str,
    prompt: str,
    cheap_output: str,
    cheap_model_id: str,           # FIX: model that served the user
    classified_tier: int,
    request_start_time: float,
) -> None:
    """
    Background task: verify quality AFTER the user gets their response.
    All exceptions caught — verification failure must never affect users.
    """
    try:
        from src.verifier.verifier import verify_response
        await verify_response(
            request_id         = request_id,
            prompt             = prompt,
            cheap_output       = cheap_output,
            cheap_model_id     = cheap_model_id,   # FIX: pass through
            classified_tier    = classified_tier,
            request_start_time = request_start_time,
        )
    except Exception as e:
        logger.error("Background verification failed for %s: %s",
                     request_id[:8], e, exc_info=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/v1/completions",
    response_model = CompletionResponse,
    summary        = "Route and complete a prompt",
    description    = "Send a chat completion request. The router selects the cheapest capable model automatically.",
)
async def complete(
    req: CompletionRequest,
    background_tasks: BackgroundTasks,
) -> CompletionResponse:
    """
    Main endpoint. Flow:
      1. Extract prompt from messages
      2. Classify complexity → pick cheapest model
      3. Call model → get response
      4. Log to DB (includes cost_if_highest_quality on every row)
      5. Return response to user IMMEDIATELY
      6. Queue async verifier as background task (runs after response)
    """
    request_start = time.monotonic()
    prompt = _extract_prompt(req.messages)

    # ── 1. Route ───────────────────────────────────────────────────────────────
    router = get_router()
    model_config, classifier_result = router.route(prompt)

    logger.info(
        "Routing: tier=%d conf=%.2f low_conf=%s → %s",
        classifier_result.tier,
        classifier_result.confidence,
        classifier_result.low_confidence,
        model_config.display_name,
    )

    # ── 2. Call model ──────────────────────────────────────────────────────────
    response = send_request(
        prompt       = prompt,
        model_config = model_config,
        max_tokens   = req.max_tokens,
        temperature  = req.temperature,
    )

    # ── 3. Log to DB ───────────────────────────────────────────────────────────
    request_id = log_request(
        response       = response,
        classifier     = classifier_result,
        user_id        = req.user_id,
        prompt_preview = prompt,
        output_preview = response.output_text,
    )

    # ── 4. Queue async verifier (non-blocking) ─────────────────────────────────
    background_tasks.add_task(
        _run_verification,
        request_id         = request_id,
        prompt             = prompt,
        cheap_output       = response.output_text,
        cheap_model_id     = response.model_id,    # FIX: cheap model that served user
        classified_tier    = classifier_result.tier,
        request_start_time = request_start,
    )

    # ── 5. Return to user immediately ─────────────────────────────────────────
    return CompletionResponse(
        content                  = response.output_text,
        request_id               = request_id,
        model_used               = response.model_id,
        provider                 = response.provider,
        tier                     = classifier_result.tier,
        confidence               = classifier_result.confidence,
        low_confidence           = classifier_result.low_confidence,
        cost_usd                 = response.cost_usd,
        cost_if_highest_quality  = response.cost_if_highest_quality,
        savings_usd              = response.savings_usd,
        savings_pct              = response.savings_pct,
        latency_ms               = response.latency_ms,
        tokens_in                = response.input_tokens,
        tokens_out               = response.output_tokens,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STATS + MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/v1/stats",
    response_model = StatsResponse,
    summary        = "Cost savings summary",
)
async def get_stats() -> StatsResponse:
    """Return cumulative cost savings and routing stats from the DB."""
    stats = get_summary_stats()
    return StatsResponse(**stats)


@app.get(
    "/v1/models",
    response_model = list[ModelInfo],
    summary        = "List all models in the registry",
)
async def list_models() -> list[ModelInfo]:
    """Return every model from registry.yaml with costs and tier info."""
    registry = load_registry()
    return [
        ModelInfo(
            registry_key           = key,
            model_id               = cfg.model_id,
            provider               = cfg.provider,
            display_name           = cfg.display_name,
            quality_tier           = cfg.quality_tier,
            cost_per_1k_input_usd  = cfg.cost_per_1k_input_usd,
            cost_per_1k_output_usd = cfg.cost_per_1k_output_usd,
            avg_latency_ms         = cfg.avg_latency_ms,
        )
        for key, cfg in registry.items()
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING CONFIG (hot-reload)
# ═══════════════════════════════════════════════════════════════════════════════

@app.put(
    "/v1/routing-config",
    response_model = RoutingConfigResponse,
    summary        = "Hot-reload tier→model mapping",
    description    = "Change which model handles each tier without restarting the server.",
)
async def update_routing_config(update: RoutingConfigUpdate) -> RoutingConfigResponse:
    """
    Live demo showstopper: swap Tier 1 from Llama 8B to Gemini with one API call.
    Takes effect on the very next request.
    """
    router = get_router()

    # Build patch dict — only include fields that were actually set
    patch = {k: v for k, v in update.model_dump().items() if v is not None}
    if not patch:
        raise HTTPException(
            status_code = 422,
            detail      = "No fields provided. Send at least one of: tier_1_model, tier_2_model, tier_3_model, fallback_model.",
        )

    try:
        router.update_routing(patch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    current = router.get_routing_config()
    return RoutingConfigResponse(
        tier_1_model             = current.get("tier_1_model", ""),
        tier_2_model             = current.get("tier_2_model", ""),
        tier_3_model             = current.get("tier_3_model", ""),
        fallback_model           = current.get("fallback_model", ""),
        low_confidence_threshold = current.get("low_confidence_threshold", 0.60),
        effective_immediately    = True,
    )


@app.get(
    "/v1/routing-config",
    response_model = RoutingConfigResponse,
    summary        = "Current routing config",
)
async def get_routing_config() -> RoutingConfigResponse:
    """Return the current tier→model mapping."""
    router  = get_router()
    current = router.get_routing_config()
    return RoutingConfigResponse(
        tier_1_model             = current.get("tier_1_model", ""),
        tier_2_model             = current.get("tier_2_model", ""),
        tier_3_model             = current.get("tier_3_model", ""),
        fallback_model           = current.get("fallback_model", ""),
        low_confidence_threshold = current.get("low_confidence_threshold", 0.60),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/v1/admin/model-info",
    response_model = ClassifierInfoResponse,
    summary        = "Classifier metadata",
)
async def get_model_info() -> ClassifierInfoResponse:
    """Return classifier accuracy, training date, and pending failure count."""
    from src.classifier.predict import get_model_info as _get_info

    try:
        info = _get_info()
    except FileNotFoundError:
        raise HTTPException(
            status_code = 503,
            detail      = "Classifier not trained yet. Run: python -m src.classifier.train",
        )

    # Count pending failures
    with get_connection() as conn:
        pending = conn.execute(
            "SELECT COUNT(*) AS n FROM routing_failures WHERE used_in_retrain=0"
        ).fetchone()["n"]

    return ClassifierInfoResponse(
        model_name       = info["model_name"] or "Unknown",
        test_accuracy    = info["test_accuracy"] or 0.0,
        model_path       = info["model_path"],
        loaded           = info["loaded"],
        pending_failures = pending,
    )


@app.post(
    "/v1/admin/retrain",
    response_model = RetrainResponse,
    summary        = "Trigger flywheel retrain",
    description    = "Absorb accumulated routing failures into the classifier. The flywheel.",
)
async def trigger_retrain(dry_run: bool = False) -> RetrainResponse:
    """
    Retrain the classifier using accumulated routing failures.
    Pass ?dry_run=true to evaluate without saving the model.
    """
    try:
        from scripts.retrain import retrain
        result = retrain(dry_run=dry_run)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain failed: {e}")

    # Fill in defaults for fields retrain() might not return
    return RetrainResponse(
        status         = result.get("status", "unknown"),
        old_accuracy   = result.get("old_accuracy", 0.0),
        new_accuracy   = result.get("new_accuracy", 0.0),
        n_original     = result.get("n_original", 0),
        n_failures     = result.get("n_failures", 0),
        n_total        = result.get("n_total", 0),
        model_replaced = result.get("model_replaced", False),
        best_model     = result.get("best_model", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
)
async def health() -> HealthResponse:
    """
    Ping all providers, check DB, check classifier.
    Returns 200 even if providers fail — status per provider is in the body.
    """
    providers_status: dict = {}

    # Check Groq
    try:
        import os
        groq_key = os.getenv("GROQ_API_KEY", "")
        providers_status["groq"] = "ok" if groq_key else "missing_api_key"
    except Exception as e:
        providers_status["groq"] = f"error: {e}"

    # Check Google
    try:
        google_key = os.getenv("GOOGLE_API_KEY", "")
        providers_status["google"] = "ok" if google_key else "missing_api_key"
    except Exception as e:
        providers_status["google"] = f"error: {e}"

    # Check DB
    db_ok = False
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        db_ok = True
    except Exception:
        pass

    # Check classifier
    classifier_loaded = False
    try:
        from src.classifier.predict import get_model_info
        info = get_model_info()
        classifier_loaded = info["loaded"]
    except Exception:
        pass

    overall = "ok" if db_ok and classifier_loaded else "degraded"

    return HealthResponse(
        status            = overall,
        providers         = providers_status,
        classifier_loaded = classifier_loaded,
        db_ok             = db_ok,
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        reload_dirs = [str(ROOT / "src")],
    )