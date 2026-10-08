"""Permutation test for cross-validated ROC-AUC of the logistic-regression model.

The null hypothesis is that the trajectory-derived features contain no
relationship with the binary stability label. Labels are permuted while the
feature matrix is kept fixed. Each permutation is evaluated with the same
stratified 5-fold CV protocol used for the observed score.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    from .train_models import build_models, DATA_PATH, FEATURE_COLS, BASE
except ImportError:
    from train_models import build_models, DATA_PATH, FEATURE_COLS, BASE


N_PERMUTATIONS = 1000
N_SPLITS = 5
RANDOM_STATE = 42


def cross_validated_roc_auc(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> float:
    """Return out-of-fold ROC-AUC for logistic regression."""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    model = build_models()["logreg"]
    probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    return roc_auc_score(y, probabilities)


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
):
    """Compare observed CV ROC-AUC with a label-permutation null distribution."""
    rng = np.random.default_rng(random_state)
    observed = cross_validated_roc_auc(X, y, n_splits, random_state)

    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        null_scores[i] = cross_validated_roc_auc(
            X,
            y_perm,
            n_splits,
            random_state,
        )

    # Add-one correction avoids a zero p-value and gives a conservative
    # finite-sample permutation p-value.
    p_value = (1 + np.sum(null_scores >= observed)) / (n_permutations + 1)

    return observed, null_scores, p_value


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].values
    y = df["label"].values

    observed, null_scores, p_value = permutation_test(X, y)

    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)

    pd.DataFrame({"null_roc_auc": null_scores}).to_csv(
        results_dir / "logreg_permutation_null.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [{
            "model": "logreg",
            "observed_cv_roc_auc": observed,
            "null_mean": null_scores.mean(),
            "null_std": null_scores.std(ddof=1),
            "null_95th_percentile": np.quantile(null_scores, 0.95),
            "permutation_p_value": p_value,
            "n_permutations": N_PERMUTATIONS,
            "n_splits": N_SPLITS,
        }]
    )
    summary.to_csv(
        results_dir / "logreg_permutation_test.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    ax.hist(null_scores, bins=30, alpha=0.8)
    ax.axvline(
        observed,
        linestyle="--",
        linewidth=2,
        label=f"Observed CV ROC-AUC = {observed:.3f}",
    )
    ax.set_xlabel("ROC-AUC under label permutation")
    ax.set_ylabel("Count")
    ax.set_title("Logistic Regression label-permutation test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        results_dir / "logreg_permutation_test.png",
        dpi=300,
    )

    print("=== Logistic Regression label-permutation test ===")
    print(f"Observed 5-fold CV ROC-AUC: {observed:.4f}")
    print(f"Null mean +/- std: {null_scores.mean():.4f} +/- {null_scores.std(ddof=1):.4f}")
    print(f"Null 95th percentile: {np.quantile(null_scores, 0.95):.4f}")
    print(f"Permutation p-value: {p_value:.4f}")
    print(f"Permutations: {N_PERMUTATIONS}")
    print(f"CV folds: {N_SPLITS}")
    print("\nSaved:", results_dir / "logreg_permutation_test.csv")
    print("Saved:", results_dir / "logreg_permutation_test.png")


if __name__ == "__main__":
    main()
