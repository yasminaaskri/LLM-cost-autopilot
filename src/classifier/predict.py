"""
Predict - use trained calibrated model to classify new prompts.
"""

import joblib
import numpy as np
from pathlib import Path

from src.classifier.features import extract_features, features_to_list

# Singleton
_model = None
_scaler = None
_model_name = None


def _load_model():
    """Lazy load the calibrated model."""
    global _model, _scaler, _model_name
    if _model is None:
        model_data = joblib.load("models/classifier.pkl")
        _model = model_data["model"]
        _scaler = model_data.get("scaler")
        _model_name = model_data.get("model_name", "Unknown")
        print(f"✅ Calibrated model loaded: {_model_name}")
    return _model, _scaler


def classify_prompt(prompt: str):
    """
    Classify a prompt and return (tier, confidence).
    
    With calibration, confidence scores will be realistic:
    - Not 100% for everything
    - Confidence reflects actual uncertainty
    """
    model, scaler = _load_model()
    
    # Extract features
    features = extract_features(prompt)
    feature_list = features_to_list(features)
    
    # Convert to numpy
    X = np.array(feature_list).reshape(1, -1)
    
    # Scale if scaler exists
    if scaler is not None:
        X = scaler.transform(X)
    
    # Predict
    tier = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    
    return tier, confidence


def classify_batch(prompts):
    """Classify multiple prompts."""
    return [classify_prompt(p) for p in prompts]


if __name__ == "__main__":
    test_prompts = [
        "Extract all email addresses from this text: support@company.com",
        "Summarize this article in 3 bullet points: AI is transforming business...",
        "Analyze the competitive landscape for EV batteries and recommend a strategy",
        "What is 2+2?",
        "Write a Python function to parse JSON with error handling",
        "Is this sentence correct? 'He go to school.'",
        "Classify this sentiment: 'The product is good but expensive'",
        "Explain how transformer attention mechanisms work"
    ]
    
    print("\n" + "=" * 50)
    print("CALIBRATED CLASSIFIER TEST")
    print("=" * 50)
    
    for prompt in test_prompts:
        tier, confidence = classify_prompt(prompt)
        status = "✅" if confidence > 0.60 else "⚠️"
        print(f"{status} Tier {tier} ({confidence:.2%}) - {prompt[:50]}...")