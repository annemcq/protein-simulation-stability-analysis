# ============================================================
# BUILD FINAL ML DATASET (ENERGY + STRUCTURE + LABELS)
# ============================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS (LOCAL REPO)
# ============================================================

BASE = Path(__file__).resolve().parents[1]

FEATURES_CSV = BASE / "data" / "project3_energy_features.csv"
STRUCT_CSV   = BASE / "data" / "project3_structure_features.csv"
PERP_CSV     = BASE / "data" / "rank_by_distance_to_fit_deltaG_fep_BOOTSTRAP.csv"

OUT_CSV = BASE / "data" / "project3_final_dataset.csv"

# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(FEATURES_CSV)
struct   = pd.read_csv(STRUCT_CSV)
labels   = pd.read_csv(PERP_CSV)

print("Energy features:", features.shape)
print("Structure features:", struct.shape)
print("Labels:", labels.shape)

# ============================================================
# CLEAN KEYS
# ============================================================

for df in [features, struct, labels]:
    df["pdb"] = df["pdb"].astype(str).str.lower().str.strip()

# Keep only relevant label column
labels = labels[["pdb", "Perp_dist"]].copy()

# ============================================================
# MERGE ALL
# ============================================================

df = pd.merge(features, struct, on="pdb", how="inner")
df = pd.merge(df, labels, on="pdb", how="inner")

print("Merged dataset:", df.shape)

# ============================================================
# CREATE LABEL (median split)
# ============================================================

threshold = df["Perp_dist"].median()

df["label"] = (df["Perp_dist"] > threshold).astype(int)

print("Threshold (median Perp_dist):", threshold)
print("\nLabel distribution:")
print(df["label"].value_counts())

# ============================================================
# OPTIONAL: REMOVE EXTREME OUTLIERS
# ============================================================

q_low = df["Perp_dist"].quantile(0.01)
q_high = df["Perp_dist"].quantile(0.99)

df = df[(df["Perp_dist"] >= q_low) & (df["Perp_dist"] <= q_high)].copy()

print("\nAfter outlier filtering:", df.shape)

# ============================================================
# SAVE
# ============================================================

df.to_csv(OUT_CSV, index=False)

print("\nSaved:", OUT_CSV)