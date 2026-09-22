"""Shared data loading and Keras utilities for the proposed HDMFN model."""

from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

N_JOBS = 2
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

from ipf_training_window import filter_training_window, training_window_metadata

warnings.filterwarnings("ignore")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

SYNC_CSV = Path("nilm_sync_outputs_ipf/03_combined/all_devices_normalized_10s_long.csv")
RUNS_ROOT = Path("runs_ipf")
MODEL_NAME = "hmfn_common"
OUT_DIR = RUNS_ROOT / MODEL_NAME
SAVED_MODELS = Path("saved_models_ipf") / MODEL_NAME
CONFIG = {"test_ratio": 0.20, "n_splits": 3}

ROLL_PREFIXES = ("current_roll_", "pf_roll_")
TEMPORAL_COLS = [
    "hour", "dayofweek", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_daytime", "is_night", "is_weekend",
]
META_COLS = {
    "timestamp", "device", "sync_offset_ms", "long_gap_flag",
    "is_interpolated", "train_set",
}


def _aggregate_from_wide(wide: pd.DataFrame, devices: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Build aggregate single-timestamp input features without device identifiers."""
    X = pd.DataFrame(index=wide.index)

    def cols_for(measurement: str) -> List[str]:
        return [f"{device}__{measurement}" for device in devices if f"{device}__{measurement}" in wide.columns]

    current_cols = cols_for("current")
    abs_cols = cols_for("current_abs")
    active_cols = cols_for("active_current_proxy")
    reactive_cols = cols_for("reactive_current_proxy")
    pf_cols = cols_for("pf")
    pf_avail_cols = cols_for("pf_available")
    delta_i_cols = cols_for("delta_current")
    delta_pf_cols = cols_for("delta_pf")

    if current_cols:
        current = wide[current_cols].astype(float)
        X["agg_current_sum"] = current.sum(axis=1)
        X["agg_current_mean"] = current.mean(axis=1)
        X["agg_current_std"] = current.std(axis=1).fillna(0)
        X["agg_current_max"] = current.max(axis=1)
        X["agg_current_min"] = current.min(axis=1)
        X["agg_current_nonzero_count"] = (current.abs() > 1e-9).sum(axis=1)

    if abs_cols:
        X["agg_current_abs_sum"] = wide[abs_cols].astype(float).sum(axis=1)
    if active_cols:
        active = wide[active_cols].astype(float)
        X["agg_active_current_proxy_sum"] = active.sum(axis=1)
        X["agg_active_current_proxy_mean"] = active.mean(axis=1)
    if reactive_cols:
        reactive = wide[reactive_cols].astype(float)
        X["agg_reactive_current_proxy_sum"] = reactive.sum(axis=1)
        X["agg_reactive_current_proxy_mean"] = reactive.mean(axis=1)
    if pf_cols:
        pf = wide[pf_cols].astype(float).replace([np.inf, -np.inf], np.nan)
        X["agg_pf_mean"] = pf.mean(axis=1)
        X["agg_pf_std"] = pf.std(axis=1).fillna(0)
        X["agg_pf_min"] = pf.min(axis=1)
        X["agg_pf_max"] = pf.max(axis=1)
    if pf_avail_cols:
        available = wide[pf_avail_cols].astype(float)
        X["agg_pf_available_count"] = available.sum(axis=1)
        X["agg_pf_available_ratio"] = available.mean(axis=1)
    if delta_i_cols:
        delta_i = wide[delta_i_cols].astype(float)
        X["agg_delta_current_sum"] = delta_i.sum(axis=1)
        X["agg_delta_current_abs_sum"] = delta_i.abs().sum(axis=1)
    if delta_pf_cols:
        delta_pf = wide[delta_pf_cols].astype(float)
        X["agg_delta_pf_mean"] = delta_pf.mean(axis=1)
        X["agg_delta_pf_abs_mean"] = delta_pf.abs().mean(axis=1)

    for prefix in ROLL_PREFIXES:
        measurements = sorted({
            column.split("__", 1)[1]
            for column in wide.columns
            if "__" in column and column.split("__", 1)[1].startswith(prefix)
        })
        for measurement in measurements:
            columns = [f"{device}__{measurement}" for device in devices if f"{device}__{measurement}" in wide.columns]
            if columns:
                values = wide[columns].astype(float)
                X[f"agg_{measurement}_mean"] = values.mean(axis=1)
                X[f"agg_{measurement}_sum"] = values.sum(axis=1)

    for column in TEMPORAL_COLS:
        if column in wide.columns:
            X[column] = wide[column]

    timestamps = pd.to_datetime(wide["timestamp"], errors="coerce")
    hour = timestamps.dt.hour.fillna(0).astype(float)
    weekday = timestamps.dt.dayofweek.fillna(0).astype(float)
    month = timestamps.dt.month.fillna(1).astype(float)
    if "hour_sin" not in X.columns:
        X["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        X["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    if "dow_sin" not in X.columns:
        X["dow_sin"] = np.sin(2 * np.pi * weekday / 7.0)
        X["dow_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    if "month_sin" not in X.columns:
        X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
        X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    if "is_daytime" not in X.columns:
        X["is_daytime"] = ((hour >= 6) & (hour < 18)).astype(float)
    if "is_night" not in X.columns:
        X["is_night"] = 1 - X["is_daytime"]
    if "is_weekend" not in X.columns:
        X["is_weekend"] = (weekday >= 5).astype(float)

    if "agg_current_sum" in X.columns:
        X["agg_current_sum_diff"] = X["agg_current_sum"].diff().fillna(0)
        X["agg_current_sum_roll_1min_mean"] = X["agg_current_sum"].rolling(6, min_periods=1).mean()
        X["agg_current_sum_roll_10min_mean"] = X["agg_current_sum"].rolling(60, min_periods=1).mean()
        X["agg_current_sum_roll_10min_std"] = X["agg_current_sum"].rolling(60, min_periods=2).std().fillna(0)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, X.columns.tolist()


def load_data() -> Tuple:
    """Load the synchronized CSV and create the chronological 80/20 split."""
    if not SYNC_CSV.exists():
        raise FileNotFoundError(f"Missing {SYNC_CSV}; run nilm_preparation_ipf_pipeline.py first")

    df_long = pd.read_csv(SYNC_CSV, low_memory=False)
    if "timestamp" not in df_long.columns or "device" not in df_long.columns:
        raise ValueError("Input CSV must contain timestamp and device columns")
    df_long["timestamp"] = pd.to_datetime(df_long["timestamp"], errors="coerce")
    df_long = df_long.dropna(subset=["timestamp", "device"]).sort_values(["timestamp", "device"]).reset_index(drop=True)
    df_long = filter_training_window(df_long)
    devices = sorted(df_long["device"].dropna().unique().tolist())

    useful = [column for column in ["current", "pf", "active_current_proxy", "reactive_current_proxy", "status"] if column in df_long.columns]
    if useful:
        df_long = df_long[~df_long[useful].isna().all(axis=1)].reset_index(drop=True)

    frames = []
    meta_frames = []
    for device, group in df_long.groupby("device"):
        value_columns = [
            column for column in group.columns
            if column not in META_COLS and not column.endswith(("_z", "_minmax"))
        ]
        meta_columns = [column for column in ["is_interpolated", "long_gap_flag", "train_set"] if column in group.columns]
        values = group.set_index("timestamp")[value_columns].copy()
        values.columns = [f"{device}__{column}" for column in value_columns]
        frames.append(values)
        if meta_columns:
            meta_frames.append(group.set_index("timestamp")[meta_columns].copy())

    wide = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "timestamp"})
    if meta_frames:
        meta_wide = pd.concat(meta_frames, axis=1)
        for meta_column in ["is_interpolated", "long_gap_flag", "train_set"]:
            columns = [column for column in meta_wide.columns if column == meta_column]
            if columns:
                wide[meta_column] = meta_wide[columns].max(axis=1).values
    if wide.columns.duplicated().any():
        wide = wide.loc[:, ~wide.columns.duplicated(keep="first")]
    if "long_gap_flag" in wide.columns:
        wide = wide.loc[wide["long_gap_flag"].fillna(0).values == 0].reset_index(drop=True)

    for device in devices:
        status_source = f"{device}__status"
        current_source = f"{device}__current"
        if status_source in wide.columns:
            wide[f"y_status__{device}"] = wide[status_source].fillna(0).astype(np.float32)
        elif current_source in wide.columns:
            current = wide[current_source].fillna(0).astype(float)
            threshold = max(0.02, float(np.nanpercentile(current, 95)) * 0.10)
            wide[f"y_status__{device}"] = (current >= threshold).astype(np.float32)
        if current_source in wide.columns:
            wide[f"y_current__{device}"] = wide[current_source].fillna(0).astype(np.float32)

    X_df, input_columns = _aggregate_from_wide(wide, devices)
    target_columns = [column for column in wide.columns if column.startswith(("y_status__", "y_current__"))]
    status_columns = [column for column in target_columns if column.startswith("y_status__")]
    current_columns = [column for column in target_columns if column.startswith("y_current__")]
    wide[target_columns] = wide[target_columns].fillna(0)

    X = X_df.values.astype(np.float32)
    Y = wide[target_columns].copy()
    split = int(len(X) * (1 - CONFIG["test_ratio"]))
    X_train, X_test = X[:split], X[split:]
    Y_train = Y.iloc[:split].reset_index(drop=True)
    Y_test = Y.iloc[split:].reset_index(drop=True)

    input_scaler = RobustScaler().fit(X_train)
    X_train = input_scaler.transform(X_train)
    X_test = input_scaler.transform(X_test)

    target_scalers: Dict[str, StandardScaler] = {}
    for column in current_columns:
        values = Y_train[column].values.reshape(-1, 1).astype(np.float32)
        nonzero = np.abs(values[:, 0]) > 1e-9
        fitting_values = values[nonzero] if nonzero.any() else values
        if len(fitting_values) < 2:
            continue
        scaler = StandardScaler().fit(fitting_values)
        target_scalers[column] = scaler
        Y_train[column] = scaler.transform(values).ravel()
        Y_test[column] = scaler.transform(Y_test[column].values.reshape(-1, 1)).ravel()

    SAVED_MODELS.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": input_scaler, "Y": target_scalers}, SAVED_MODELS / "scalers.pkl")
    metadata = {
        "input_csv": str(SYNC_CSV),
        **training_window_metadata(),
        "input_cols": input_columns,
        "target_cols": target_columns,
        "status_cols": status_columns,
        "current_cols": current_columns,
        "note": "Aggregate I/PF features with a chronological 80/20 split and train-only model scalers.",
    }
    (OUT_DIR / "reports" / "data_manifest_ipf.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return X_train, X_test, Y_train, Y_test, input_columns, target_columns, target_scalers, devices


def inverse_df(frame: pd.DataFrame, target_scalers: Dict[str, StandardScaler]) -> pd.DataFrame:
    result = frame.copy()
    for column, scaler in target_scalers.items():
        if column in result.columns:
            result[column] = scaler.inverse_transform(result[[column]].values).ravel()
    return result
