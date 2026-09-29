"""
Tests for the model evaluation and statistical-testing helpers. Uses
small synthetic datasets, not the real project data, so these run in
under a second with no dependency on the (unpublished) raw trajectory
data.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.train_models import evaluate_model, build_models, FEATURE_COLS
from src.repeated_cv_evaluation import holm_bonferroni, run_repeated_evaluation, paired_significance


def test_build_models_returns_three_named_models():
    models = build_models()
    assert set(models.keys()) == {"dummy", "logreg", "random_forest"}


def test_evaluate_model_perfect_classifier_scores_near_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)  # perfectly separable on feature 0

    model = LogisticRegression().fit(X, y)
    metrics = evaluate_model(model, X, y)

    assert metrics["roc_auc"] > 0.99
    assert metrics["accuracy"] > 0.95


def test_evaluate_model_returns_expected_keys():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 2))
    y = rng.integers(0, 2, size=50)
    model = LogisticRegression().fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert set(metrics.keys()) == {"roc_auc", "avg_precision", "accuracy"}


def test_holm_bonferroni_never_decreases_pvalues():
    pvalues = np.array([0.01, 0.02, 0.03, 0.04])
    adjusted = holm_bonferroni(pvalues)
    assert all(adjusted >= pvalues - 1e-12)


def test_holm_bonferroni_caps_at_one():
    pvalues = np.array([0.9, 0.95, 0.99])
    assert all(holm_bonferroni(pvalues) <= 1.0)


def _make_synthetic_dataset(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_COLS)))
    # make label weakly but genuinely dependent on the first feature,
    # so logreg/rf should beat the dummy baseline more often than not
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df["label"] = y
    return df


def test_run_repeated_evaluation_produces_one_row_per_model_per_repeat():
    df = _make_synthetic_dataset()
    results = run_repeated_evaluation(df, n_repeats=5)
    assert len(results) == 5 * 3  # 5 repeats x 3 models
    assert set(results["model"].unique()) == {"dummy", "logreg", "random_forest"}


def test_paired_significance_detects_real_signal_on_synthetic_data():
    """With a genuinely predictive feature, logreg should score significantly
    higher than the dummy baseline (which always scores exactly 0.5)."""
    df = _make_synthetic_dataset(n=150)
    results = run_repeated_evaluation(df, n_repeats=20)
    sig_df = paired_significance(results, metric="roc_auc")

    row = sig_df[
        ((sig_df["model_a"] == "dummy") & (sig_df["model_b"] == "logreg")) |
        ((sig_df["model_a"] == "logreg") & (sig_df["model_b"] == "dummy"))
    ]
    assert len(row) == 1
    assert row.iloc[0]["wilcoxon_p_holm"] < 0.05
