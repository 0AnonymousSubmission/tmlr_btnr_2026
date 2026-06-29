#!/usr/bin/env python3
"""One-off: compute std(y) (and a few related stats) for each dataset.
Run from the repo root (so utils/model imports resolve):
    python paper_script/compute_target_std.py
"""

import os
import sys
import csv

import torch

# Make repo root importable when run from anywhere.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.data import load_full_dataset_for_kfold  # noqa: E402

# The datasets that have a conf/dataset/<name>.yaml (excluding _base/csv).
DATASETS = [
    "abalone",
    "ai4i",
    "appliances",
    "bike",
    "concrete",
    "energy_efficiency",
    "obesity",
    "realstate",
    "seoulBike",
    "student_perf",
]

CAP = 50  # matches conf/dataset/_base.yaml
OUT_CSV = os.path.join(_SCRIPT_DIR, "target_std.csv")


def main():
    rows = []
    for name in DATASETS:
        try:
            _, y, info = load_full_dataset_for_kfold(name, cap=CAP)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {name}: {e}")
            continue

        y = y.reshape(-1).to(torch.float64)
        std = float(y.std(unbiased=True).item())
        mean = float(y.mean().item())
        ymin = float(y.min().item())
        ymax = float(y.max().item())
        n = int(y.numel())

        rows.append(
            {
                "dataset": name,
                "n_samples": n,
                "y_mean": mean,
                "y_std": std,
                "y_min": ymin,
                "y_max": ymax,
                "y_range": ymax - ymin,
            }
        )
        print(f"  {name:<18} n={n:<7} std(y)={std:.6g}  range={ymax - ymin:.6g}")

    if not rows:
        print("No datasets loaded; nothing written.")
        return

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "n_samples",
                "y_mean",
                "y_std",
                "y_min",
                "y_max",
                "y_range",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {os.path.relpath(OUT_CSV, _REPO_ROOT)} ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
