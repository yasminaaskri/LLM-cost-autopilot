"""
Router — maps every incoming prompt to the cheapest capable model.

Flow:
  1. classify_prompt(prompt) → ClassifierResult (tier + confidence)
  2. If confidence < threshold → use fallback_model from routing.yaml
  3. Else → look up tier_N_model from routing.yaml
  4. Return (ModelConfig, ClassifierResult)

The Router is a singleton. update_routing() hot-reloads the routing
config without restarting the process — called by PUT /v1/routing-config.

Fix vs your original:
  - routing.yaml opened via ROOT-anchored Path, not a bare relative
    string, so it works regardless of the working directory.
  - update_routing() also calls save_routing() to persist the change.
  - route() returns (ModelConfig, ClassifierResult) — the full
    ClassifierResult, not just (config, tier, confidence), so the
    API layer has access to the raw features for logging.
"""

from __future__ import annotations
import logging
from pathlib import Path

from src.classifier.predict import classify_prompt, reset_singleton as reset_classifier
from src.config import load_registry, load_routing, save_routing, get_model_by_tier
from src.models import ModelConfig, ClassifierResult

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent


class Router:
    """
    Stateful router singleton.
    Holds the registry and routing config in memory for fast access.
    """

    def __init__(self) -> None:
        self._registry = load_registry()
        self._routing  = load_routing()
        logger.info("Router initialised — %d models in registry", len(self._registry))

    # ── Public API ─────────────────────────────────────────────────────────────

    def route(self, prompt: str) -> tuple[ModelConfig, ClassifierResult]:
        """
        Classify a prompt and return the cheapest capable model.

        Args:
            prompt: The raw user prompt string.

        Returns:
            (ModelConfig, ClassifierResult)
            ModelConfig  — the model to call via send_request()
            ClassifierResult — tier, confidence, features (for logging)
        """
        threshold = self._routing["routing"].get("low_confidence_threshold", 0.60)
        result    = classify_prompt(prompt, low_confidence_threshold=threshold)

        if result.low_confidence:
            model_key = self._routing["routing"]["fallback_model"]
            logger.debug(
                "Low confidence (%.2f) → fallback model: %s",
                result.confidence, model_key,
            )
        else:
            model_key = get_model_by_tier(result.tier, self._routing)
            logger.debug(
                "Tier %d (%.2f confidence) → model: %s",
                result.tier, result.confidence, model_key,
            )

        model_config = self._registry.get(model_key)
        if model_config is None:
            # Safety net: if routing.yaml references a key not in registry
            logger.error("Model key '%s' not found in registry — falling back to cheapest", model_key)
            model_config = min(
                self._registry.values(),
                key=lambda m: m.cost_per_1k_input_usd,
            )

        return model_config, result

    def update_routing(self, new_routing_section: dict) -> None:
        """
        Hot-reload the tier→model mapping without restarting.
        Persists to routing.yaml AND updates the in-memory config.

        Called by PUT /v1/routing-config.

        Args:
            new_routing_section: Dict with keys like 'tier_1_model',
                                 'tier_2_model', 'tier_3_model', etc.
        """
        # Validate: all referenced models must exist in registry
        for key, model_key in new_routing_section.items():
            if key.endswith("_model") and model_key not in self._registry:
                raise ValueError(
                    f"Model '{model_key}' (from key '{key}') not found in registry. "
                    f"Available: {list(self._registry.keys())}"
                )

        save_routing(new_routing_section)
        # Reload full config to pick up any changes outside routing section too
        self._routing = load_routing()
        logger.info("Routing config updated: %s", new_routing_section)

    def get_routing_config(self) -> dict:
        """Return the current routing section for GET /v1/routing-config."""
        return self._routing["routing"].copy()

    def reload_registry(self) -> None:
        """Force reload registry.yaml. Useful after adding models."""
        self._registry = load_registry()
        logger.info("Registry reloaded — %d models", len(self._registry))


# ── Module-level singleton ─────────────────────────────────────────────────────

_router: Router | None = None


def get_router() -> Router:
    """Return the module-level Router singleton, creating it on first call."""
    global _router
    if _router is None:
        _router = Router()
    return _router


def reset_router() -> None:
    """Force re-creation of the router singleton. Used in tests."""
    global _router
    _router = None
    reset_classifier()


def route(prompt: str) -> tuple[ModelConfig, ClassifierResult]:
    """Convenience function — calls get_router().route(prompt)."""
    return get_router().route(prompt)