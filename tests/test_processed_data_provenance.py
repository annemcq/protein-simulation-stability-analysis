"""Regression checks for relations recoverable from committed processed tables.

These checks do not assert the unverified formulas for drift/contrast or
claim to reproduce features from unavailable raw trajectories.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def test_energy_range_and_fluctuation_relations():
    df = pd.read_csv(DATA / "project3_energy_features.csv")
    assert len(df) == 96
    np.testing.assert_allclose(
        df["range_deltaV"],
        df["max_deltaV"] - df["min_deltaV"],
        rtol=0, atol=1e-10,
    )
    np.testing.assert_allclose(
        df["fluct_deltaV"],
        df["std_deltaV"] / df["mean_deltaV"].abs(),
        rtol=0, atol=0.0016,
    )


def test_fitted_line_residual_and_perpendicular_distance():
    df = pd.read_csv(DATA / "rank_by_distance_to_fit_deltaG_fep_BOOTSTRAP.csv")
    assert len(df) == 95
    x = df["Ref_rank"].to_numpy()
    y = df["ΔG_fep"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fit = intercept + slope * x
    np.testing.assert_allclose(df["Fit_ŷ"], fit, rtol=0, atol=1e-10)
    residual = y - fit
    np.testing.assert_allclose(df["Residual"], residual, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        df["Perp_dist"], np.abs(residual) / np.sqrt(1 + slope**2),
        rtol=0, atol=1e-10,
    )


def test_dataset_intersection_trimming_and_class_balance():
    tables = [
        pd.read_csv(DATA / name)
        for name in (
            "project3_energy_features.csv",
            "project3_structure_features.csv",
            "rank_by_distance_to_fit_deltaG_fep_BOOTSTRAP.csv",
        )
    ]
    keys = [set(df["pdb"].astype(str).str.lower().str.strip()) for df in tables]
    assert [len(s) for s in keys] == [96, 126, 95]
    assert len(set.intersection(*keys)) == 90
    final = pd.read_csv(DATA / "project3_final_dataset.csv")
    assert len(final) == 88
    assert final["label"].value_counts().to_dict() == {0: 44, 1: 44}
    assert len(final.columns.intersection(["pdb", "Perp_dist", "label"])) == 3
