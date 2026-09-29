# ============================================================
# REPEATED EVALUATION + PAIRED SIGNIFICANCE TESTING
# ============================================================
#
# The single train/test split in train_models.py has only 18 test
# samples (out of ~90 systems total). With a set that small, the gap
# between the dummy baseline (ROC-AUC 0.50) and Random Forest (0.55)
# could easily be noise from that one particular split rather than a
# real effect. 
# To identify this, the current script repeats the split many times, 
# evaluates all three models on the *same* split each repeat, and runs 
# a paired statistical test (mirroring the approach used in the tcr-peptide-
# ranking project) to check whether Random Forest's advantage over the
# dummy baseline holds up, or whether it's within noise.

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split

try:
    from .train_models import build_models, evaluate_model, FEATURE_COLS, DATA_PATH, BASE
except ImportError:  # running as a standalone script (python src/repeated_cv_evaluation.py)
    from train_models import build_models, evaluate_model, FEATURE_COLS, DATA_PATH, BASE

N_REPEATS = 30
TEST_SIZE = 0.2
METRIC = "roc_auc"


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    """Standard Holm step-down adjustment. Returns adjusted p-values, same order as input."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * pvalues[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def run_repeated_evaluation(df: pd.DataFrame, n_repeats: int = N_REPEATS, test_size: float = TEST_SIZE) -> pd.DataFrame:
    """
    Repeats the stratified train/test split n_repeats times (different
    random_state each time), evaluating all 3 models on the *same*
    split within each repeat -- so per-repeat scores are paired
    observations across models, same as the main train/test pipeline's
    single split, just repeated.
    """
    X = df[FEATURE_COLS].values
    y = df["label"].values

    rows = []
    for repeat in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=repeat, stratify=y
        )
        models = build_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
            rows.append({"repeat": repeat, "model": name, **metrics})

    return pd.DataFrame(rows)


def paired_significance(results_df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    wide = results_df.pivot(index="repeat", columns="model", values=metric)
    models = sorted(wide.columns)

    rows = []
    for model_a, model_b in combinations(models, 2):
        a, b = wide[model_a].values, wide[model_b].values
        diff = a - b
        wilcoxon_p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(a, b).pvalue
        ttest_p = stats.ttest_rel(a, b).pvalue
        rows.append({
            "model_a": model_a, "model_b": model_b,
            f"{metric}_mean_a": a.mean(), f"{metric}_mean_b": b.mean(),
            "mean_diff (a - b)": diff.mean(),
            "wilcoxon_p": wilcoxon_p, "paired_ttest_p": ttest_p,
        })

    result_df = pd.DataFrame(rows)
    result_df["wilcoxon_p_holm"] = holm_bonferroni(result_df["wilcoxon_p"].values)
    result_df["significant_after_correction (alpha=0.05)"] = result_df["wilcoxon_p_holm"] < 0.05
    return result_df.sort_values("wilcoxon_p_holm")


def main():
    df = pd.read_csv(DATA_PATH)
    results_df = run_repeated_evaluation(df)

    results_dir = BASE / "results"
    results_dir.mkdir(exist_ok=True)
    results_df.to_csv(results_dir / "repeated_eval_all_results.csv", index=False)

    summary = results_df.groupby("model")[["roc_auc", "avg_precision", "accuracy"]].agg(["mean", "std"])
    print(f"\n=== Mean +/- std across {N_REPEATS} repeats ===")
    print(summary)

    sig_df = paired_significance(results_df)
    sig_df.to_csv(results_dir / "paired_significance_roc_auc.csv", index=False)
    print(f"\n=== Paired significance testing (ROC-AUC, Holm-Bonferroni corrected) ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(sig_df.to_string(index=False))

    # boxplot of ROC-AUC across repeats, one box per model
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    order = ["dummy", "logreg", "random_forest"]
    data = [results_df[results_df["model"] == m]["roc_auc"].values for m in order]
    ax.boxplot(data, tick_labels=order)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (ROC-AUC=0.5)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"ROC-AUC across {N_REPEATS} repeated splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "repeated_eval_boxplot.png", dpi=300)
    print("\nSaved:", results_dir / "repeated_eval_boxplot.png")


if __name__ == "__main__":
    main()
