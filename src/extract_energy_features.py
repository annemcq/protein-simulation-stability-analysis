# ============================================================
# EXTRACT STRUCTURAL FEATURES (RMSD)
# ============================================================

import os
import numpy as np
import pandas as pd
import mdtraj as md
from pathlib import Path

# Raw trajectory data is not included in this repository (see README's
# "Data Availability" section) -- this script documents the feature
# extraction pipeline as it was originally run, and needs these two
# environment variables (or edit the defaults below) pointing at your
# own raw trajectory data to actually execute.
BASE = os.environ.get("PROJECT3_RAW_DATA_DIR", "./raw_data")

TRAJ_BASE = os.environ.get("PROJECT3_TRAJ_DIR", os.path.join(BASE, "trajectories"))
PDB_BASE = os.path.join(BASE, "processed_data", "processed_data")

OUT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "project3_structure_features.csv")

EARLY_FRAC = 0.2

# ============================================================
# HELPERS
# ============================================================

def find_pdb(pdb):
    fname = f"MHC_{pdb}_cg_structure.pdb"
    path = os.path.join(PDB_BASE, fname)
    return path if os.path.exists(path) else None


def find_traj(pdb):
    folder = os.path.join(TRAJ_BASE, pdb)
    traj = os.path.join(folder, f"MHC_{pdb}_sim_trajectory.dcd")
    return traj if os.path.exists(traj) else None


# ============================================================
# FEATURE COMPUTATION
# ============================================================

def compute_rmsd_features(traj_file, pdb_file):
    traj = md.load(traj_file, top=pdb_file)

    rmsd = md.rmsd(traj, traj[0])

    n = len(rmsd)
    n_early = max(5, int(EARLY_FRAC * n))
    early = rmsd[:n_early]

    mean_rmsd = float(np.mean(early))
    std_rmsd = float(np.std(early))

    # slope
    x = np.arange(len(early))
    slope = float(np.polyfit(x, early, 1)[0])

    return {
        "mean_rmsd": mean_rmsd,
        "std_rmsd": std_rmsd,
        "slope_rmsd": slope
    }


# ============================================================
# MAIN
# ============================================================

def main():
    pdbs = sorted(os.listdir(TRAJ_BASE))

    rows = []

    for i, pdb in enumerate(pdbs, 1):
        try:
            pdb_file = find_pdb(pdb)
            traj_file = find_traj(pdb)

            if not pdb_file or not traj_file:
                raise ValueError("Missing files")

            feats = compute_rmsd_features(traj_file, pdb_file)
            feats["pdb"] = pdb

            rows.append(feats)

            print(f"[{i:03d}] {pdb} OK")

        except Exception as e:
            print(f"[SKIP] {pdb}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)
    print("Total systems:", len(df))


if __name__ == "__main__":
    main()