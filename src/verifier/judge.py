"""
LLM-as-judge quality scorer.

After the cheap model responds to a user, the async verifier calls
score_quality() to compare that output against what the highest-quality
model produces for the same prompt.

The judge is ALWAYS the highest-quality model in the registry
(llama-3.3-70b-versatile via Groq). We force JSON output so the
score is always parseable.

Design decisions:
  - We call the judge via our own dispatcher, so the cost is tracked.
  - We require JSON mode by prefixing the response with a brace in the
    system prompt — Groq doesn't support OpenAI-style response_format yet.
  - judge_cost_usd is tracked so the dashboard can show total verification cost.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from src.models import ModelConfig, QualityScore, ProviderError
from src.config import load_registry, get_highest_quality_model, calculate_cost

logger = logging.getLogger(__name__)

# ── Singleton: high-quality model config used for judging ──────────────────────
_judge_model: Optional[ModelConfig] = None


def _get_judge_model() -> ModelConfig:
    global _judge_model
    if _judge_model is None:
        registry     = load_registry()
        _judge_model = get_highest_quality_model()
        logger.info("Judge model: %s", _judge_model.display_name)
    return _judge_model


# ── Judge prompt ───────────────────────────────────────────────────────────────
_JUDGE_SYSTEM = """You are a strict, impartial quality evaluator for LLM outputs.

You will receive:
1. ORIGINAL_PROMPT — the task given to both models
2. RESPONSE_A — output from the cheap/routed model
3. RESPONSE_B — output from the reference high-quality model

Score each response on a scale of 1–5:
  5 = perfect, complete, accurate
  4 = good, minor gaps
  3 = acceptable, some missing detail
  2 = incomplete or partially wrong
  1 = wrong, off-topic, or refused

Then determine if routing to the cheap model was CORRECT:
  routing_correct = true  if  (score_A >= 3.0) AND (score_B - score_A < 1.5)
  routing_correct = false otherwise

Respond with ONLY valid JSON, no other text:
{
  "cheap_score": <float 1-5>,
  "expensive_score": <float 1-5>,
  "quality_gap": <float, expensive_score minus cheap_score>,
  "routing_correct": <true or false>,
  "failure_reason": <string explaining the gap, or null if routing_correct is true>
}"""


def score_quality(
    original_prompt: str,
    cheap_output: str,
    expensive_output: str,
) -> QualityScore:
    """
    Call the judge model to score both outputs.

    Args:
        original_prompt:  The prompt both models received.
        cheap_output:     Output from the cheap routed model.
        expensive_output: Output from the highest-quality model.

    Returns:
        QualityScore with all fields populated.

    Raises:
        ProviderError if the judge API call fails.
        ValueError if the judge returns unparseable JSON.
    """
    judge_model = _get_judge_model()

    user_message = (
        f"ORIGINAL_PROMPT:\n{original_prompt}\n\n"
        f"RESPONSE_A (cheap model):\n{cheap_output}\n\n"
        f"RESPONSE_B (reference model):\n{expensive_output}"
    )

    # Import here to avoid circular imports
    from src.providers.dispatcher import send_request

    response = send_request(
        prompt      = f"{_JUDGE_SYSTEM}\n\n{user_message}",
        model_config = judge_model,
        max_tokens  = 256,
        temperature = 0.0,   # deterministic judge
    )

    judge_cost = calculate_cost(
        judge_model, response.input_tokens, response.output_tokens)

    parsed = _parse_judge_output(response.output_text)

    return QualityScore(
        cheap_score     = float(parsed["cheap_score"]),
        expensive_score = float(parsed["expensive_score"]),
        quality_gap     = float(parsed["quality_gap"]),
        routing_correct = bool(parsed["routing_correct"]),
        failure_reason  = parsed.get("failure_reason"),
        judge_cost_usd  = judge_cost,
    )


def _parse_judge_output(text: str) -> dict:
    """
    Extract JSON from judge output. Handles:
      - Clean JSON response
      - JSON wrapped in markdown code block
      - Leading/trailing whitespace
    """
    # Strip markdown fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find first { ... } block in case of preamble text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Judge returned unparseable JSON: {e}\nRaw output: {text[:300]}"
        )

    required = {"cheap_score", "expensive_score", "quality_gap", "routing_correct"}
    missing  = required - set(data.keys())
    if missing:
        raise ValueError(f"Judge JSON missing keys: {missing}")

    # Enforce numeric quality_gap (sometimes the model computes it wrong)
    data["quality_gap"] = float(data["expensive_score"]) - float(data["cheap_score"])

    return data