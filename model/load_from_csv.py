"""
Generic CSV dataset loader.
Loads any CSV from csvs/ directory (or absolute path) with the convention:
  - Last column = target
  - All other columns = features

Usage:
    from model.load_from_csv import get_csvdata
    X_train, y_train, X_val, y_val, X_test, y_test = get_csvdata("nic.csv")
"""

import os
import pandas as pd
from model._preproc import split_data, scale_dataframes, to_tensors

CSVS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csvs")


def resolve_csv_path(csv_path):
    """Resolve a CSV path: absolute path used as-is, relative resolved under csvs/."""
    if os.path.isabs(csv_path):
        return csv_path
    return os.path.join(CSVS_DIR, csv_path)


def get_csvdata(csv_path, task="regression", device="cpu", cap=None):
    """Load a CSV dataset and return train/val/test tensors.

    Convention: the last column is the target.
    All columns are treated as numeric and StandardScaled.

    Args:
        csv_path: Path to CSV file (absolute, or relative to csvs/)
        task: "regression" or "classification"
        device: Target device for tensors
        cap: Ignored (CSV data is all-numeric, no one-hot needed)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: torch tensors
    """
    full_path = resolve_csv_path(csv_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV not found: {full_path}. Looked in {CSVS_DIR}")

    df = pd.read_csv(full_path)
    target_col = df.columns[-1]
    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    X_train, X_val, X_test = scale_dataframes(X_train, X_val, X_test)

    return to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, task, device=device)
