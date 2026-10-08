"""Compare repeated 80/20 holdout with repeated 5-fold CV on the same dataset.

This diagnostic tests whether the discrepancy between the original repeated
holdout ROC-AUC and the permutation-test CV ROC-AUC is mainly due to the
resampling protocol.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

try:
    from .train_models import build_models, DATA_PATH, FEATURE_COLS, BASE
except ImportError:
    from train_models import build_models, DATA_PATH, FEATURE_COLS, BASE


N_REPEATS = 30
TEST_SIZE = 0.2


def repeated_holdout_auc(df, n_repeats=N_REPEATS):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    scores = []

    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        model = build_models()["logreg"]
        model.fit(X_train, y_train)
        scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    return np.asarray(scores)


def repeated_5fold_auc(df, n_repeats=N_REPEATS, model_name="logreg"):
    X = df[FEATURE_COLS].values
    y = df["label"].values
    scores = []

    for seed in range(n_repeats):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        model = build_models()[model_name]
        probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        scores.append(roc_auc_score(y, probs))

    return np.asarray(scores)


def main():
    df = pd.read_csv(DATA_PATH)

    holdout = repeated_holdout_auc(df)
    cv5 = repeated_5fold_auc(df, model_name="logreg")
    cv5_by_model = {
        name: repeated_5fold_auc(df, model_name=name)
        for name in ("dummy", "logreg", "random_forest")
    }
    cv5_summary = pd.DataFrame([
        {
            "model": name,
            "mean_roc_auc": scores.mean(),
            "std_roc_auc": scores.std(ddof=1),
            "median_roc_auc": np.median(scores),
            "min_roc_auc": scores.min(),
            "max_roc_auc": scores.max(),
        }
        for name, scores in cv5_by_model.items()
    ])

    summary = pd.DataFrame([
        {
            "protocol": "30x stratified 80/20 holdout",
            "mean_roc_auc": holdout.mean(),
            "std_roc_auc": holdout.std(ddof=1),
            "median_roc_auc": np.median(holdout),
            "q25": np.quantile(holdout, 0.25),
            "q75": np.quantile(holdout, 0.75),
        },
        {
            "protocol": "30x repeated 5-fold CV",
            "mean_roc_auc": cv5.mean(),
            "std_roc_auc": cv5.std(ddof=1),
            "median_roc_auc": np.median(cv5),
            "q25": np.quantile(cv5, 0.25),
            "q75": np.quantile(cv5, 0.75),
        },
    ])

    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)
    summary.to_csv(results_dir / "resampling_protocol_comparison.csv", index=False)
    cv5_summary.to_csv(results_dir / "repeated_5fold_model_comparison.csv", index=False)
    print("\\n=== 30x repeated 5-fold CV: all models ===")
    print(cv5_summary.to_string(index=False))

    print("=== Resampling protocol comparison: Logistic Regression ===")
    print(summary.to_string(index=False))
    print("\nMean difference (holdout - 5-fold): "
          f"{holdout.mean() - cv5.mean():.4f}")
    print("Holdout range: "
          f"{holdout.min():.4f} - {holdout.max():.4f}")
    print("5-fold range: "
          f"{cv5.min():.4f} - {cv5.max():.4f}")
    print("\nSaved:", results_dir / "resampling_protocol_comparison.csv")
    print("Saved:", results_dir / "repeated_5fold_model_comparison.csv")


if __name__ == "__main__":
    main()
