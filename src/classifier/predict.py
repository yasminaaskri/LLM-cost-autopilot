"""
Predict — use the trained calibrated model to classify new prompts.

Returns (tier: int, confidence: float) for every prompt.
The model is loaded once (singleton) and reused across all calls.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.classifier.features import extract_features, features_to_list
from src.models import ClassifierResult

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT / "models" / "classifier.pkl"

# ── Singleton state ────────────────────────────────────────────────────────────
_model       = None
_scaler      = None
_model_name  = None
_accuracy    = None


def _load_model(model_path: Path = MODEL_PATH) -> tuple:
    """Lazy-load the calibrated model. Thread-safe enough for single-worker use."""
    global _model, _scaler, _model_name, _accuracy
    if _model is None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {model_path}.\n"
                f"Run: python -m src.classifier.train"
            )
        data        = joblib.load(model_path)
        _model      = data["model"]
        _scaler     = data.get("scaler")
        _model_name = data.get("model_name", "Unknown")
        _accuracy   = data.get("test_accuracy", 0.0)
        logger.info("Classifier loaded: %s (accuracy=%.2f%%)",
                    _model_name, (_accuracy or 0) * 100)
    return _model, _scaler


def reset_singleton() -> None:
    """Force reload on next call. Used by retrain.py after saving a new pkl."""
    global _model, _scaler, _model_name, _accuracy
    _model = _scaler = _model_name = _accuracy = None


def get_model_info() -> dict:
    """Return metadata about the loaded model. Used by GET /v1/admin/model-info."""
    _load_model()
    return {
        "model_name":    _model_name,
        "test_accuracy": _accuracy,
        "model_path":    str(MODEL_PATH),
        "loaded":        _model is not None,
    }


def classify_prompt(
    prompt: str,
    low_confidence_threshold: float = 0.60,
) -> ClassifierResult:
    """
    Classify a single prompt and return a ClassifierResult.

    Args:
        prompt:                    The raw prompt string.
        low_confidence_threshold:  Below this confidence the router
                                   uses the safe fallback model.

    Returns:
        ClassifierResult with tier (1/2/3), confidence (0–1),
        raw features dict, and low_confidence flag.
    """
    model, scaler = _load_model()

    features     = extract_features(prompt)
    feature_list = features_to_list(features)
    X = np.array(feature_list).reshape(1, -1)

    if scaler is not None:
        X = scaler.transform(X)

    tier          = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence    = float(max(probabilities))

    return ClassifierResult(
        tier           = tier,
        confidence     = confidence,
        features       = features,
        low_confidence = confidence < low_confidence_threshold,
    )


def classify_batch(
    prompts: list[str],
    low_confidence_threshold: float = 0.60,
) -> list[ClassifierResult]:
    """Classify a list of prompts. More efficient than calling classify_prompt in a loop."""
    model, scaler = _load_model()
    feature_names = [features_to_list(extract_features(p)) for p in prompts]
    X = np.array(feature_names)

    if scaler is not None:
        X = scaler.transform(X)

    tiers         = model.predict(X)
    probabilities = model.predict_proba(X)

    return [
        ClassifierResult(
            tier           = int(tiers[i]),
            confidence     = float(max(probabilities[i])),
            features       = extract_features(prompts[i]),
            low_confidence = float(max(probabilities[i])) < low_confidence_threshold,
        )
        for i in range(len(prompts))
    ]


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── IMPORTANT: use realistic full-length prompts ───────────────────────────
    # Short truncated prompts (ending in '...') look identical to Tier 1
    # to the classifier because token_count is the strongest signal.
    # Always test with the actual length you'll use in production.
    test_prompts = [
        # Tier 1 — short, direct, no reasoning required
        ("Extract all email addresses from this text: support@company.com or sales@firm.io", 1),
        ("Reformat this date from MM/DD/YYYY to DD-MM-YYYY: 06/17/2026", 1),
        ("Is the following sentence grammatically correct? Reply yes or no. Sentence: 'She don't like coffee.'", 1),
        # Tier 2 — moderate context, structured output
        ("Summarize the following paragraph in exactly 2 sentences: Artificial intelligence has transformed how companies operate, enabling automation of repetitive tasks, faster data analysis, and more personalised customer experiences.", 2),
        ("Classify this customer message into one of: complaint, question, compliment, or feature_request. Message: 'Your app keeps crashing whenever I try to export a PDF. This is really frustrating — I have lost work twice now.'", 2),
        ("Translate the following sentence to French: 'The meeting has been rescheduled to Thursday at 3pm. Please update your calendars accordingly.'", 2),
        # Tier 3 — complex reasoning, long output, multi-step
        ("Analyze the competitive landscape for EV batteries and recommend a market entry strategy. Consider: current players, cost trends, and geographic opportunities.", 3),
        ("Write a Python function called flatten_dict that: 1) Accepts a nested dict of arbitrary depth, 2) Flattens it using dot-notation keys, 3) Handles lists by indexing them. Include type hints and a docstring.", 3),
        ("What are the second-order economic consequences of widespread LLM adoption in knowledge work? Consider effects on wage levels for different skill brackets and how this differs from previous automation waves.", 3),
    ]

    print("\n" + "=" * 62)
    print("CLASSIFIER TEST")
    print("=" * 62)
    print(f"{'Status':<6} {'Predicted':>9} {'Expected':>9} {'Conf':>7}  Prompt")
    print("─" * 62)

    correct = 0
    for prompt, expected in test_prompts:
        result = classify_prompt(prompt)
        match  = result.tier == expected
        correct += int(match)
        icon   = "✅" if match else "❌"
        warn   = " ⚠ low-conf" if result.low_confidence else ""
        print(f"{icon}  Tier {result.tier:>1} (pred)  "
              f"Tier {expected:>1} (exp)  "
              f"{result.confidence:>6.1%}  "
              f"{prompt[:42]}...{warn}")

    print("─" * 62)
    print(f"Accuracy on test set: {correct}/{len(test_prompts)} "
          f"({correct/len(test_prompts):.0%})\n")
