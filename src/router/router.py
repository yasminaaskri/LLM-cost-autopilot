"""
Router - decides which model to use for each prompt.
"""

import yaml
from pathlib import Path

from src.classifier.predict import classify_prompt
from src.config import load_registry, get_model_by_tier


class Router:
    def __init__(self):
        self.registry = load_registry()
        self.routing = self._load_routing()
        self.threshold = self.routing["routing"]["low_confidence_threshold"]
        self.fallback = self.routing["routing"]["fallback_model"]
    
    def _load_routing(self):
        with open("config/routing.yaml", "r") as f:
            return yaml.safe_load(f)
    
    def route(self, prompt: str):
        """
        Route a prompt to the right model.
        
        How it works:
        1. Classify the prompt (get tier and confidence)
        2. If confidence is low, use fallback model
        3. Otherwise, use the model for that tier
        4. Return ModelConfig, tier, confidence
        """
        # 1. Classify
        tier, confidence = classify_prompt(prompt)
        
        # 2. Check confidence
        if confidence < self.threshold:
            model_key = self.fallback
            print(f"⚠️ Low confidence ({confidence:.2%}) - using fallback")
        else:
            model_key = get_model_by_tier(tier, self.routing)
        
        # 3. Get model config
        model_config = self.registry.get(model_key)
        if not model_config:
            # Fallback to cheapest
            model_config = min(self.registry.values(), key=lambda m: m.cost_per_1k_input_usd)
        
        return model_config, tier, confidence


# Singleton
_router = None


def get_router():
    global _router
    if _router is None:
        _router = Router()
    return _router


def route(prompt: str):
    """Convenience function to route a prompt."""
    return get_router().route(prompt)