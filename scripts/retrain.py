"""
scripts/retrain.py — the feedback flywheel.

Every routing failure logged in the routing_failures table is a
new training example. This script:

  1. Loads the original hand-labeled dataset
  2. Exports unprocessed failures from the DB as new examples
  3. Combines them
  4. Retrains the classifier
  5. Only saves the new model if accuracy doesn't regress > 2%
  6. Logs the retrain event to retrain_log table
  7. Resets the predict.py singleton so new model is used immediately

Schedule: runs every Sunday at 02:00 via the background worker cron.
Can also be triggered manually: python scripts/retrain.py
Or via API: POST /v1/admin/retrain

The retrain guard (accuracy must not drop > 2%) prevents a batch of
mislabeled failure examples from accidentally degrading the classifier.
"""

from __future__ import annotations
import sys
import hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from src.classifier.features import extract_features, features_to_list, get_feature_names
from src.classifier.predict import reset_singleton as reset_classifier
from src.database import init_db, export_failures_for_retrain, get_connection

ORIGINAL_DATA_PATH = ROOT / "data" / "labeled_prompts.csv"
MODEL_PATH         = ROOT / "models" / "classifier.pkl"
ACCURACY_GUARD     = 0.02   # allow up to 2% accuracy regression


def _get_current_accuracy() -> float:
    """Read the accuracy stored in the current pkl."""
    if not MODEL_PATH.exists():
        return 0.0
    try:
        data = joblib.load(MODEL_PATH)
        return float(data.get("test_accuracy", 0.0))
    except Exception:
        return 0.0


def _model_hash(path: Path) -> str:
    """SHA-256 of the pkl file for the audit log."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _log_retrain_event(old_acc: float, new_acc: float,
                       n_original: int, n_failures: int,
                       replaced: bool, model_hash: str) -> None:
    """Insert a row into retrain_log."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO retrain_log (
                timestamp, old_accuracy, new_accuracy,
                num_original, num_failures, total_examples,
                model_replaced, model_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            old_acc, new_acc,
            n_original, n_failures, n_original + n_failures,
            int(replaced), model_hash,
        ))


def retrain(
    original_data_path: Path = ORIGINAL_DATA_PATH,
    model_path: Path = MODEL_PATH,
    dry_run: bool = False,
) -> dict:
    """
    Full retrain pipeline.

    Args:
        original_data_path: Path to hand-labeled labeled_prompts.csv
        model_path:         Where to save the new classifier.pkl
        dry_run:            If True, trains and evaluates but does NOT
                            save the model or mark failures as used.

    Returns:
        dict with keys: old_accuracy, new_accuracy, n_failures,
                        model_replaced, status
    """
    WIDTH = 60
    print("=" * WIDTH)
    print("FLYWHEEL RETRAIN")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * WIDTH)

    init_db()
    old_accuracy = _get_current_accuracy()
    print(f"  Current classifier accuracy: {old_accuracy:.2%}")

    # ── 1. Load original labeled dataset ──────────────────────────────────────
    if not original_data_path.exists():
        print(f"  ✗ Original data not found at {original_data_path}")
        return {"status": "error", "message": "labeled_prompts.csv not found"}

    df_original = pd.read_csv(original_data_path)[["prompt", "tier"]]
    n_original  = len(df_original)
    print(f"  Original examples:  {n_original}")

    # ── 2. Export failure examples ─────────────────────────────────────────────
    if dry_run:
        # In dry_run, peek at failures without marking them used
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT prompt, correct_tier FROM routing_failures WHERE used_in_retrain=0"
            ).fetchall()
        failures = [{"prompt": r["prompt"], "tier": r["correct_tier"]} for r in rows]
    else:
        failures = export_failures_for_retrain()

    n_failures = len(failures)
    print(f"  New failure examples: {n_failures}")

    if n_failures == 0 and old_accuracy > 0:
        print("  No new failure examples — skipping retrain (model is current)")
        return {
            "status":         "skipped",
            "reason":         "no_new_failures",
            "old_accuracy":   old_accuracy,
            "new_accuracy":   old_accuracy,
            "n_failures":     0,
            "model_replaced": False,
        }

    # ── 3. Combine datasets ────────────────────────────────────────────────────
    df_failures = pd.DataFrame(failures) if failures else pd.DataFrame(columns=["prompt", "tier"])
    df_combined = pd.concat([df_original, df_failures], ignore_index=True)
    n_total     = len(df_combined)
    print(f"  Total training examples: {n_total}")

    dist = df_combined["tier"].value_counts().sort_index().to_dict()
    print(f"  Tier distribution: {dist}")

    # ── 4. Featurise ──────────────────────────────────────────────────────────
    X = np.array([
        features_to_list(extract_features(p)) for p in df_combined["prompt"]
    ])
    y = df_combined["tier"].values

    # ── 5. Scale + split ──────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Keep 20% held-out — same random_state as original training for consistency
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size    = 0.20,
        random_state = 42,
        stratify     = y,
    )

    # ── 6. Train best of LR + RF (same as train.py) ───────────────────────────
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=42, C=1.0, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"),
    }

    best_model    = None
    best_name     = ""
    best_accuracy = 0.0

    for name, base in candidates.items():
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        calibrated.fit(X_train, y_train)
        acc = accuracy_score(y_test, calibrated.predict(X_test))
        print(f"  {name:<25}: {acc:.2%}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model    = calibrated
            best_name     = name

    new_accuracy = best_accuracy
    print(f"\n  Best: {best_name}  →  {new_accuracy:.2%}")

    # ── 7. Retrain guard ──────────────────────────────────────────────────────
    replaced = False
    if old_accuracy > 0 and new_accuracy < (old_accuracy - ACCURACY_GUARD):
        print(
            f"\n  ⚠ Accuracy dropped from {old_accuracy:.2%} → {new_accuracy:.2%} "
            f"(>{ACCURACY_GUARD*100:.0f}% regression). Keeping old model."
        )
        status = "rejected_regression"
    else:
        if dry_run:
            print("\n  [DRY RUN] Would save new model — skipping file write.")
            status = "dry_run_ok"
        else:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_data = {
                "model":         best_model,
                "scaler":        scaler,
                "feature_names": get_feature_names(),
                "model_name":    best_name,
                "test_accuracy": new_accuracy,
                "cv_accuracy":   new_accuracy,   # retrain uses same test set
            }
            joblib.dump(model_data, model_path)
            reset_classifier()   # force predict.py to reload on next call
            replaced = True
            status   = "replaced"
            h        = _model_hash(model_path)
            _log_retrain_event(old_accuracy, new_accuracy,
                               n_original, n_failures, replaced, h)
            print(f"\n  ✓ Model saved → {model_path}")
            print(f"  ✓ Accuracy: {old_accuracy:.2%} → {new_accuracy:.2%}")
            print(f"  ✓ New examples absorbed: {n_failures}")

    print("=" * WIDTH)

    return {
        "status":         status,
        "old_accuracy":   old_accuracy,
        "new_accuracy":   new_accuracy,
        "n_original":     n_original,
        "n_failures":     n_failures,
        "n_total":        n_total,
        "model_replaced": replaced,
        "best_model":     best_name,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrain the complexity classifier")
    parser.add_argument("--dry-run", action="store_true",
                        help="Train and evaluate but do not save the model")
    args = parser.parse_args()
    result = retrain(dry_run=args.dry_run)
    print("\nResult:", result)