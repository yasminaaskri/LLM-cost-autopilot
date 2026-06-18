"""
Train the complexity classifier.

Trains two models (Logistic Regression + Random Forest), both with
Platt-scaling calibration so predict_proba() returns realistic
confidence scores rather than overconfident 0/1 values.

Saves ONLY what predict.py needs — no extra result objects bloating
the pkl. The comparison table is printed but not persisted.

Fix vs your original:
  - model_data no longer contains the full 'results' dict (which
    embeds both trained model objects and bloats the pkl by ~5x).
  - Accuracy is stored as a scalar for the retrain guard in retrain.py.
  - Uses ROOT-anchored paths via pathlib so it works from any CWD.

Run:
    python -m src.classifier.train
    # or
    python scripts/train_classifier.py
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from src.classifier.features import extract_features, features_to_list, get_feature_names

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = ROOT / "data" / "labeled_prompts.csv"
MODEL_PATH = ROOT / "models" / "classifier.pkl"


def load_and_featurise(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled CSV and return (X feature matrix, y labels)."""
    df = pd.read_csv(csv_path)

    required = {"prompt", "tier"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"labeled_prompts.csv is missing columns: {missing}. "
            f"Expected at minimum: prompt, tier"
        )

    print(f"✓ Loaded {len(df)} labeled prompts")
    dist = df["tier"].value_counts().sort_index().to_dict()
    print(f"  Distribution: {dist}")

    if len(df) < 60:
        raise ValueError(
            f"Only {len(df)} prompts found. Need at least 60 for a "
            f"meaningful train/test split. Target is 200+."
        )

    X = np.array([features_to_list(extract_features(p)) for p in df["prompt"]])
    y = df["tier"].values
    return X, y


def train(csv_path: Path = DATA_PATH,
          model_path: Path = MODEL_PATH) -> dict:
    """
    Full training pipeline. Returns a summary dict with accuracies.

    Steps:
      1. Load + featurise labeled data
      2. Scale features (required for Logistic Regression)
      3. 80/20 stratified split
      4. Train LR + RF, both with 5-fold Platt-scaling calibration
      5. Pick best model by test accuracy
      6. Save MINIMAL pkl: model, scaler, feature_names, accuracy, model_name
      7. Print comparison table
    """
    print("=" * 60)
    print("CLASSIFIER TRAINING")
    print("=" * 60)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    X, y = load_and_featurise(csv_path)
    feature_names = get_feature_names()
    print(f"✓ {len(feature_names)} features per prompt")

    # ── 2. Scale ───────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. Split ───────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size   = 0.20,
        random_state = 42,
        stratify    = y,
    )
    print(f"✓ Train: {len(X_train)}  Test: {len(X_test)}")

    # ── 4. Train both models ───────────────────────────────────────────────────
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter     = 5000,
            random_state = 42,
            C            = 1.0,
            class_weight = "balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators = 100,
            max_depth    = 10,
            random_state = 42,
            class_weight = "balanced",
        ),
    }

    results: dict[str, dict] = {}

    for name, base_model in candidates.items():
        print(f"\n{'─' * 50}")
        print(f"  Training {name} …")

        calibrated = CalibratedClassifierCV(
            base_model,
            method = "sigmoid",   # Platt scaling
            cv     = 5,
        )
        calibrated.fit(X_train, y_train)

        y_pred   = calibrated.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Cross-val for a more honest accuracy estimate
        cv_scores = cross_val_score(calibrated, X_scaled, y, cv=5)

        print(f"  Test accuracy:       {accuracy:.2%}")
        print(f"  CV accuracy (5-fold): {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
        print()
        print(classification_report(y_test, y_pred,
                                    target_names=["Tier 1", "Tier 2", "Tier 3"]))
        print("  Confusion matrix:")
        cm = confusion_matrix(y_test, y_pred)
        for row in cm:
            print("   ", row)

        results[name] = {
            "calibrated_model": calibrated,
            "test_accuracy":    accuracy,
            "cv_mean":          float(cv_scores.mean()),
        }

    # ── 5. Pick best model ─────────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["test_accuracy"])
    best      = results[best_name]

    print(f"\n{'=' * 60}")
    print(f"  Best model: {best_name}  ({best['test_accuracy']:.2%} test accuracy)")
    print("=" * 60)

    if best["test_accuracy"] < 0.80:
        print(
            f"\n⚠  Accuracy {best['test_accuracy']:.2%} is below the 80% target.\n"
            f"   Add more labeled examples — especially for the confused tiers.\n"
            f"   Current dataset size: {len(y)} prompts.\n"
        )

    # ── 6. Save minimal pkl ────────────────────────────────────────────────────
    # Only save what predict.py needs. Do NOT include the 'results' dict —
    # it embeds both model objects and bloats the file ~5x for no benefit.
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model":         best["calibrated_model"],
        "scaler":        scaler,
        "feature_names": feature_names,
        "model_name":    best_name,
        "test_accuracy": best["test_accuracy"],
        "cv_accuracy":   best["cv_mean"],
    }
    joblib.dump(model_data, model_path)
    print(f"\n✓ Model saved → {model_path}")

    # ── 7. Comparison table ────────────────────────────────────────────────────
    print("\n  MODEL COMPARISON")
    print("  " + "─" * 42)
    for name, r in results.items():
        marker = " ← best" if name == best_name else ""
        print(f"  {name:<25}  {r['test_accuracy']:.2%}{marker}")
    print()

    return {
        "best_model_name": best_name,
        "test_accuracy":   best["test_accuracy"],
        "cv_accuracy":     best["cv_mean"],
        "n_train":         len(X_train),
        "n_test":          len(X_test),
    }


if __name__ == "__main__":
    train()