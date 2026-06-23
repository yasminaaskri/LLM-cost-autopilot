"""
Pydantic v2 request/response schemas for the FastAPI API.

All endpoint I/O is defined here — never inline in main.py.
This keeps main.py clean and makes the schemas importable by tests.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., examples=["user"])
    content: str = Field(..., min_length=1)


class CompletionRequest(BaseModel):
    """
    POST /v1/completions request body.
    OpenAI-compatible shape so any client that speaks OpenAI can use this.
    """
    messages: list[Message] = Field(..., min_length=1)
    user_id:  str           = Field(default="default")
    max_tokens: int         = Field(default=1024, ge=1, le=8192)
    temperature: float      = Field(default=0.1,  ge=0.0, le=2.0)


class RoutingConfigUpdate(BaseModel):
    """PUT /v1/routing-config — update tier→model mapping live."""
    tier_1_model: Optional[str] = None
    tier_2_model: Optional[str] = None
    tier_3_model: Optional[str] = None
    fallback_model: Optional[str] = None
    low_confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ── Response models ────────────────────────────────────────────────────────────

class CompletionResponse(BaseModel):
    """
    POST /v1/completions response.
    Returns the model output PLUS routing metadata so clients can see
    which model was used and how much was saved.
    """
    content:        str
    request_id:     str
    model_used:     str
    provider:       str
    tier:           int
    confidence:     float
    low_confidence: bool
    cost_usd:       float
    cost_if_highest_quality: float
    savings_usd:    float
    savings_pct:    float
    latency_ms:     float
    tokens_in:      int
    tokens_out:     int


class ModelInfo(BaseModel):
    registry_key:           str
    model_id:               str
    provider:               str
    display_name:           str
    quality_tier:           str
    cost_per_1k_input_usd:  float
    cost_per_1k_output_usd: float
    avg_latency_ms:         int


class StatsResponse(BaseModel):
    total_requests:              int
    total_cost_usd:              float
    total_baseline_cost:         float   # alias — same value, used by dashboard
    savings_usd:                 float
    savings_pct:                 float
    avg_quality_score:           Optional[float]
    avg_latency_ms:              float
    escalation_rate_pct:         float
    requests_by_tier:            dict
    requests_by_model:           dict


class RoutingConfigResponse(BaseModel):
    tier_1_model:              str
    tier_2_model:              str
    tier_3_model:              str
    fallback_model:            str
    low_confidence_threshold:  float
    effective_immediately:     bool = True


class ClassifierInfoResponse(BaseModel):
    model_name:     str
    test_accuracy:  float
    model_path:     str
    loaded:         bool
    pending_failures: int


class RetrainResponse(BaseModel):
    status:         str
    old_accuracy:   float
    new_accuracy:   float
    n_original:     int
    n_failures:     int
    n_total:        int
    model_replaced: bool
    best_model:     str


class HealthResponse(BaseModel):
    status:    str
    providers: dict
    classifier_loaded: bool
    db_ok:     bool