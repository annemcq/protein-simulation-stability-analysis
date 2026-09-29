# ============================================================
# TRAIN ML MODELS FOR SIMULATION STABILITY PREDICTION
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    average_precision_score,
    classification_report
)

import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "project3_final_dataset.csv"

FEATURE_COLS = [
    "mean_deltaV",
    "std_deltaV",
    "min_deltaV",
    "max_deltaV",
    "range_deltaV",
    "slope_deltaV",
    "drift_deltaV",
    "contrast_deltaV",
    "fluct_deltaV",
    "mean_rmsd",
    "std_rmsd",
    "slope_rmsd"
]


def build_models():
    """Fresh, unfitted instances of the three models -- reused by the
    single-split pipeline here and by the repeated-evaluation script."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }


def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    return {
        "roc_auc": roc_auc_score(y_test, probs),
        "avg_precision": average_precision_score(y_test, probs),
        "accuracy": accuracy_score(y_test, preds)
    }


def main():
    # ============================================================
    # LOAD DATA
    # ============================================================
    df = pd.read_csv(DATA_PATH)
    print("Dataset shape:", df.shape)

    X = df[FEATURE_COLS].values
    y = df["label"].values
    print("Feature matrix:", X.shape)
    print("Labels:", y.shape)

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nTrain size:", X_train.shape)
    print("Test size:", X_test.shape)

    # ============================================================
    # TRAIN + EVALUATE
    # ============================================================
    models = build_models()
    results = []

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        print("Metrics:", metrics)
        results.append({"model": name, **metrics})
        print("\nClassification report:")
        print(classification_report(y_test, model.predict(X_test), zero_division=0))

    results_df = pd.DataFrame(results)
    print("\n=== Model Comparison ===")
    print(results_df)

    # ============================================================
    # FEATURE IMPORTANCE (Random Forest)
    # ============================================================
    rf = models["random_forest"]
    importances = rf.feature_importances_
    feat_importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances
    }).sort_values("importance", ascending=False)
    print("\n=== Feature Importance (Random Forest, impurity-based) ===")
    print(feat_importance)

    # ============================================================
    # PLOT MODEL COMPARISON
    # ============================================================
    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)

    plt.figure()
    plt.bar(results_df["model"], results_df["roc_auc"])
    plt.ylabel("ROC-AUC")
    plt.title("Model Comparison (Simulation Stability Prediction)")
    plt.tight_layout()

    out_fig = results_dir / "model_comparison.png"
    plt.savefig(out_fig, dpi=300)
    print("\nSaved plot:", out_fig)

    return results_df


if __name__ == "__main__":
    main()
