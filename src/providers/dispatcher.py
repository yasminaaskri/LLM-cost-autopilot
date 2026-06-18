"""
Dispatcher — the single function the entire codebase calls.
Never import groq_provider or google_provider directly anywhere else.

send_request(prompt, model_config) → LLMResponse
"""

from __future__ import annotations
import time
import uuid
from datetime import datetime

from src.models import ModelConfig, LLMResponse, ProviderError
from src.config import calculate_cost, calculate_cost_if_highest_quality


def send_request(
    prompt: str,
    model_config: ModelConfig,
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> LLMResponse:
    """
    Route a prompt to the correct provider and return a standardised LLMResponse.

    This is the ONLY function the router, verifier, and escalation engine call.
    All provider-specific logic is isolated inside the provider modules.

    Args:
        prompt:       The user prompt string.
        model_config: ModelConfig loaded from the registry.
        max_tokens:   Max output tokens (default 1024).
        temperature:  Sampling temperature (default 0.1 for consistency).

    Returns:
        LLMResponse with all fields populated including cost_if_highest_quality.

    Raises:
        ProviderError if the API call fails after retries.
    """
    # Pre-compute the highest quality model baseline BEFORE the API call.
    # We estimate input tokens from prompt length (actual count returned after).
    # We'll recompute cost_if_highest_quality after we have real token counts.
    estimated_input = len(prompt.split()) * 4 // 3  # rough pre-call estimate

    start = time.monotonic()

    raw = _dispatch(prompt, model_config, max_tokens, temperature)

    latency_ms = (time.monotonic() - start) * 1000

    # Now compute costs with REAL token counts from the API response
    cost_usd       = calculate_cost(
        model_config, raw["input_tokens"], raw["output_tokens"])
    cost_if_highest_quality  = calculate_cost_if_highest_quality(
        raw["input_tokens"], raw["output_tokens"])

    return LLMResponse(
        output_text   = raw["text"],
        input_tokens  = raw["input_tokens"],
        output_tokens = raw["output_tokens"],
        latency_ms    = round(latency_ms, 2),
        cost_usd      = cost_usd,
        cost_if_highest_quality = cost_if_highest_quality,   # ALWAYS set — never None
        model_id      = model_config.model_id,
        provider      = model_config.provider,
        timestamp     = datetime.utcnow(),
        request_id    = str(uuid.uuid4()),
    )


def _dispatch(
    prompt: str,
    model_config: ModelConfig,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Route to the correct provider module based on model_config.provider."""

    if model_config.provider == "groq":
        from src.providers import groq_provider
        return groq_provider.send(prompt, model_config, max_tokens, temperature)

    elif model_config.provider == "google":
        from src.providers import google_provider
        return google_provider.send(prompt, model_config, max_tokens, temperature)

    else:
        raise ProviderError(
            model_id=model_config.model_id,
            message=f"Unknown provider '{model_config.provider}'. "
                    f"Supported: groq, google",
        )
