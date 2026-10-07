"""SHAP feature importance for the Random Forest model."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split

try:
    from .train_models import build_models, FEATURE_COLS, DATA_PATH, BASE
except ImportError:
    from train_models import build_models, FEATURE_COLS, DATA_PATH, BASE


def compute_shap_importances(df: pd.DataFrame, random_state: int = 42):
    """Compute SHAP feature importance on a held-out test split."""
    X = df[FEATURE_COLS].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    rf = build_models()["random_forest"]
    rf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_test)

    # SHAP output format depends on the installed version.
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return importance_df, sv, X_test


def main():
    df = pd.read_csv(DATA_PATH)

    importance_df, sv, X_test = compute_shap_importances(df)

    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)

    importance_df.to_csv(
        results_dir / "shap_feature_importance.csv",
        index=False,
    )

    print(
        "=== SHAP feature importance "
        "(Random Forest, held-out test split) ==="
    )
    print(importance_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4.5))

    order = importance_df["feature"].values

    ax.barh(
        order[::-1],
        importance_df["mean_abs_shap"].values[::-1],
    )

    ax.set_xlabel("mean |SHAP value|")
    ax.set_title("SHAP feature importance (Random Forest)")

    fig.tight_layout()

    fig.savefig(
        results_dir / "shap_feature_importance.png",
        dpi=300,
    )

    print(
        "\nSaved:",
        results_dir / "shap_feature_importance.png",
    )


if __name__ == "__main__":
    main()
