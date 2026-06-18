"""
Train the classifier with calibration for both models.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from src.classifier.features import extract_features, features_to_list, get_feature_names


def main():
    print("=" * 60)
    print("CLASSIFIER TRAINING (WITH CALIBRATION)")
    print("=" * 60)
    
    # 1. Load labeled data
    df = pd.read_csv("data/labeled_prompts.csv")
    print(f"✅ Loaded {len(df)} labeled prompts")
    print(f"   Distribution: {df['tier'].value_counts().sort_index().to_dict()}")
    
    # 2. Extract features
    X = []
    for prompt in df['prompt']:
        features = extract_features(prompt)
        X.append(features_to_list(features))
    X = np.array(X)
    y = df['tier'].values
    feature_names = get_feature_names()
    
    print(f"✅ Extracted {len(feature_names)} features per prompt")
    
    # 3. Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training: {len(X_train)} prompts, Testing: {len(X_test)} prompts")
    
    # 5. Train both models with calibration
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000,
            random_state=42,
            C=1.0,
            class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    results = {}
    
    for name, base_model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        
        # Calibrate the model
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method='sigmoid',  # Platt scaling
            cv=5
        )
        
        calibrated_model.fit(X_train, y_train)
        y_pred = calibrated_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        results[name] = {
            "model": calibrated_model,
            "accuracy": accuracy,
            "base_model": base_model
        }
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = calibrated_model
            best_name = name
    
    print(f"\n{'='*50}")
    print(f"🏆 Best model: {best_name} with {best_accuracy:.2%} accuracy")
    print("=" * 50)
    
    # 6. Save best model
    Path("models").mkdir(exist_ok=True)
    model_data = {
        "model": best_model,
        "feature_names": feature_names,
        "scaler": scaler,
        "model_name": best_name,
        "results": results
    }
    joblib.dump(model_data, "models/classifier.pkl")
    print(f"✅ Model saved to models/classifier.pkl")
    
    # 7. Show comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name:<25}: {result['accuracy']:.2%}")
    
    print(f"\n🎉 Best accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()