"""
Loads and validates config/registry.yaml and config/routing.yaml.
Single source of truth for all model configs and routing rules.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from src.models import ModelConfig

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
REGISTRY_PATH = ROOT / "config" / "registry.yaml"
ROUTING_PATH  = ROOT / "config" / "routing.yaml"
TIERS_PATH    = ROOT / "config" / "tiers.yaml"

# ── GPT-4o baseline costs (for savings calculation) ────────────────────────────
# We use GPT-4o pricing as the "what you would have paid" baseline.
# These are hardcoded intentionally — they are the benchmark, not a provider.
GPT4O_COST_PER_1K_INPUT  = 0.005   # USD
GPT4O_COST_PER_1K_OUTPUT = 0.015   # USD


def load_registry() -> dict[str, ModelConfig]:
    """
    Parse config/registry.yaml into a dict of {registry_key: ModelConfig}.
    Called once at startup; result is cached by the Router singleton.
    """
    with open(REGISTRY_PATH, "r") as f:
        raw = yaml.safe_load(f)

    registry: dict[str, ModelConfig] = {}
    for key, data in raw["models"].items():
        config = ModelConfig(
            provider=data["provider"],
            model_id=data["model_id"],
            cost_per_1k_input_usd=float(data["cost_per_1k_input_usd"]),
            cost_per_1k_output_usd=float(data["cost_per_1k_output_usd"]),
            quality_tier=data["quality_tier"],
            avg_latency_ms=int(data["avg_latency_ms"]),
            display_name=data["display_name"],
            registry_key=key,
        )
        registry[key] = config

    return registry


def load_routing() -> dict:
    """
    Parse config/routing.yaml.
    Returns the full routing section as a plain dict.
    """
    with open(ROUTING_PATH, "r") as f:
        raw = yaml.safe_load(f)
    return raw  # keep nested structure (routing, quality, latency)


def save_routing(new_routing_section: dict) -> None:
    """
    Persist an updated routing section back to routing.yaml.
    Used by PUT /v1/routing-config for hot-reloads.
    Only overwrites the 'routing' key — quality and latency sections preserved.
    """
    with open(ROUTING_PATH, "r") as f:
        current = yaml.safe_load(f)

    current["routing"].update(new_routing_section)

    with open(ROUTING_PATH, "w") as f:
        yaml.dump(current, f, default_flow_style=False, sort_keys=False)


def load_tiers() -> dict:
    """Load tier definitions for labeling UI and documentation."""
    with open(TIERS_PATH, "r") as f:
        return yaml.safe_load(f)


def calculate_cost(model_config: ModelConfig,
                   input_tokens: int,
                   output_tokens: int) -> float:
    """Cost in USD for actual routed model."""
    return model_config.cost_for_tokens(input_tokens, output_tokens)


def get_highest_quality_model():
    registry = load_registry()
    high_quality = [m for m in registry.values() if m.quality_tier == "high"]
    return max(high_quality, key=lambda m: m.cost_per_1k_input_usd + m.cost_per_1k_output_usd)

def calculate_cost_if_highest_quality(input_tokens, output_tokens):
    baseline_model = get_highest_quality_model()
    return baseline_model.cost_for_tokens(input_tokens, output_tokens)

def get_model_by_tier(tier: int, routing: dict) -> str:
    """
    Return the registry key for a given tier from routing config.
    e.g. tier=1 → 'llama-3.1-8b-instant'
    """
    key = f"tier_{tier}_model"
    return routing["routing"][key]
