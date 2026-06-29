"""
Shared data preprocessing utilities for dataset loaders.
Used by load_ucirepo.py and load_from_csv.py.
"""

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """Split into train/val/test with fixed seed for reproducibility."""
    test_val_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_val_size, random_state=random_state
    )
    relative_test_size = test_size / test_val_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_dataframes(X_train, X_val, X_test, num_cols=None):
    """Fit StandardScaler on training data and transform all splits."""
    if num_cols is None:
        num_cols = X_train.columns.tolist()

    if len(num_cols) == 0:
        return X_train, X_val, X_test

    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    X_train[num_cols] = X_train[num_cols].astype(float)
    X_val[num_cols] = X_val[num_cols].astype(float)
    X_test[num_cols] = X_test[num_cols].astype(float)

    X_train.loc[:, num_cols] = scaler.transform(X_train[num_cols])
    X_val.loc[:, num_cols] = scaler.transform(X_val[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test


def to_tensors(X_train, X_val, X_test, y_train, y_val, y_test, task, device="cpu"):
    """Convert pandas DataFrames/Series to torch tensors."""
    y_dtype = torch.float64 if task == "regression" else torch.long

    X_train_t = torch.tensor(X_train.values, dtype=torch.float64, device=device)
    y_train_t = torch.tensor(y_train.values, dtype=y_dtype, device=device)
    if task == "regression" and y_train_t.ndim == 1:
        y_train_t = y_train_t.unsqueeze(1)

    X_val_t = torch.tensor(X_val.values, dtype=torch.float64, device=device)
    y_val_t = torch.tensor(y_val.values, dtype=y_dtype, device=device)
    if task == "regression" and y_val_t.ndim == 1:
        y_val_t = y_val_t.unsqueeze(1)

    X_test_t = torch.tensor(X_test.values, dtype=torch.float64, device=device)
    y_test_t = torch.tensor(y_test.values, dtype=y_dtype, device=device)
    if task == "regression" and y_test_t.ndim == 1:
        y_test_t = y_test_t.unsqueeze(1)

    return X_train_t, y_train_t, X_val_t, y_val_t, X_test_t, y_test_t
