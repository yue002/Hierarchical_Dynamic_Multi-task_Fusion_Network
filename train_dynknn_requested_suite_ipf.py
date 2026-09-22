#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Requested DynKNN suite:
#   1) DynKNN(7->3 random) regression only
#   2) DynKNN(7->3 random) + DNN regression only
#   3) Conv1,2,3 + DynKNN(7->3 random) + DNN regression only
#   4) Parallel:
#        top  = Conv1,2,3 + DynKNN(7->3 random) + DNN classification
#        bot  = Conv1,2,3 + DynKNN(7->3 random) + Dense1024 regression
#        4a) final current = P(on) * I_reg
#        4b) final current = hard status gate * I_reg
#   5) Fixed KNN k=7 regression-driven baseline
#
# Important assumption for "regression only" models:
#   There is no explicit classification branch, so F1 is derived by converting
#   predicted current back to ON/OFF with per-device thresholds tuned on the
#   chronological train split only.

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

with contextlib.redirect_stdout(io.StringIO()):
    import hmfn_common as base


SUITE_NAME = "dynknn_requested_suite_v1"
RUNS_ROOT = Path("runs_ipf")
SAVED_ROOT = Path("saved_models_ipf")
OUT_ROOT = RUNS_ROOT / SUITE_NAME
SAVED_SUITE_ROOT = SAVED_ROOT / SUITE_NAME
SHARED_OUT_DIR = RUNS_ROOT / f"{SUITE_NAME}_shared"
SHARED_SAVED_DIR = SAVED_ROOT / f"{SUITE_NAME}_shared"

for d in [OUT_ROOT, SAVED_SUITE_ROOT, SHARED_OUT_DIR / "reports", SHARED_SAVED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


CONFIG = dict(base.CONFIG)
CONFIG.update({
    "n_splits": 3,
    "test_ratio": 0.20,
    "dyn_knn_k_min": 3.0,
    "dyn_knn_k_max": 7.0,
    "dyn_knn_max_neighbors": 7,
    "dyn_knn_metric": "euclidean",
    "dyn_knn_seed": 42,
    "fixed_knn_k": 7,
    "cnn_epochs": 20,
    "cnn_batch": 1024,
    "cnn_patience": 5,
    "cnn_lr": 1e-3,
    "embedding_dim": 64,
    "reg_dnn_epochs": 80,
    "reg_dnn_batch": 1024,
    "reg_dnn_lr": 1e-3,
    "reg_dnn_patience": 8,
    "reg_transformer_epochs": 80,
    "reg_transformer_batch": 1024,
    "reg_transformer_lr": 1e-3,
    "reg_transformer_patience": 8,
    "reg_transformer_heads": 8,
    "reg_transformer_key_dim": 32,
    "cls_dnn_epochs": 80,
    "cls_dnn_batch": 1024,
    "cls_dnn_lr": 1e-3,
    "cls_dnn_patience": 8,
    "dense1024_epochs": 80,
    "dense1024_batch": 1024,
    "dense1024_lr": 1e-3,
    "dense1024_patience": 8,
    "bp_hidden1": 64,
    "bp_hidden2": 32,
    "bp_epochs": 30,
    "bp_batch": 1024,
    "bp_lr": 1e-3,
    "bp_patience": 8,
    "classification_threshold": 0.5,
})

base.MODEL_NAME = SUITE_NAME
base.OUT_DIR = SHARED_OUT_DIR
base.SAVED_MODELS = SHARED_SAVED_DIR
base.CONFIG = CONFIG


MODEL_SPECS = [
    {
        "key": "dynknn_reg_only",
        "label": "DynKNN 7->3 Random (Reg only)",
        "variant": "dynknn_reg_only",
        "training_param": "Regression-only DynKNN on aggregate features; K = round(uniform(3,7)) per query; inverse-distance weighted neighbor current average",
    },
    {
        "key": "dynknn_dnn_reg_only",
        "label": "DynKNN 7->3 Random + DNN (Reg only)",
        "variant": "dynknn_dnn_reg_only",
        "training_param": "Regression-only DynKNN on aggregate features -> DNN calibrator Dense128 -> Dense64 -> Dense50 -> Linear",
    },
    {
        "key": "dynknn_transformer_reg_only",
        "label": "DynKNN 7->3 Random + Transformer (Reg only)",
        "variant": "dynknn_transformer_reg_only",
        "training_param": "Regression-only DynKNN on aggregate features -> Transformer calibrator token Dense64 -> MHA8 -> residual LayerNorm -> Dense128 -> Dense64 -> Dense32 -> Linear",
    },
    {
        "key": "conv_dynknn_dnn_reg_only",
        "label": "Conv1,2,3 + DynKNN 7->3 Random + DNN (Reg only)",
        "variant": "conv_dynknn_dnn_reg_only",
        "training_param": "Regression-only Conv1D filters[30,40,32], kernels[10,5,1] -> embedding -> DynKNN -> DNN calibrator Dense128 -> Dense64 -> Dense50 -> Linear",
    },
    {
        "key": "parallel_p_on",
        "label": "Parallel Conv DynKNN + DNN / Conv DynKNN + Dense1024 (use P(on))",
        "variant": "parallel_p_on",
        "training_param": "Parallel top branch: Conv1D[30,40,32] -> DynKNN -> DNN classifier Dense128 -> Dense64 -> Sigmoid; bottom branch: Conv1D[30,40,32] -> DynKNN regression -> Dense1024 -> Linear; final current = P(on) * I_reg",
    },
    {
        "key": "parallel_hard_gate",
        "label": "Parallel Conv DynKNN + DNN / Conv DynKNN + Dense1024 (hard gate)",
        "variant": "parallel_hard_gate",
        "training_param": "Parallel top branch: Conv1D[30,40,32] -> DynKNN -> DNN classifier Dense128 -> Dense64 -> Sigmoid; bottom branch: Conv1D[30,40,32] -> DynKNN regression -> Dense1024 -> Linear; final current = status_gate * I_reg",
    },
    {
        "key": "parallel_hard_gate_bp_both",
        "label": "Parallel Conv DynKNN + BP + DNN / Conv DynKNN + BP + Dense1024 (hard gate)",
        "variant": "parallel_hard_gate_bp_both",
        "training_param": "Parallel top branch: Conv1D[30,40,32] -> DynKNN -> BP adapter Dense64 -> Dense32 -> DNN classifier Dense128 -> Dense64 -> Sigmoid; bottom branch: Conv1D[30,40,32] -> DynKNN regression -> BP adapter Dense64 -> Dense32 -> Dense1024 -> Linear; final current = status_gate * I_reg",
    },
    {
        "key": "knn_k7_reg_driven",
        "label": "KNN Fixed k=7 (Reg driven)",
        "variant": "knn_k7_reg_driven",
        "training_param": "Regression-driven KNN baseline on aggregate features; K = 7 fixed; inverse-distance regression, ON/OFF derived from current thresholds",
    },
]

MODEL_SPEC_BY_KEY = {spec["key"]: spec for spec in MODEL_SPECS}


def model_dirs(model_key: str) -> Tuple[Path, Path]:
    out_dir = OUT_ROOT / model_key
    saved_dir = SAVED_SUITE_ROOT / model_key
    return out_dir, saved_dir


def reset_model_dirs(model_key: str) -> Tuple[Path, Path]:
    out_dir, saved_dir = model_dirs(model_key)
    for folder in [out_dir / "reports", out_dir / "preds", out_dir / "plots", saved_dir]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
    return out_dir, saved_dir


def device_from_target(col: str) -> str:
    return col.split("__", 1)[1] if "__" in col else col


def compute_nde(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.square(y_true)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(np.square(y_true - y_pred)) / denom)


def finite_mean(values: List[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


class LocalEpochTimer(base.tf.keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Dict[str, float]] = []
        self._t0 = 0.0

    def on_epoch_begin(self, epoch, logs=None) -> None:
        self._t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.rows.append({
            "epoch": int(epoch) + 1,
            "time_sec": float(time.perf_counter() - self._t0),
        })


@dataclass
class ComponentResult:
    oof: Optional[np.ndarray]
    test: np.ndarray
    fit_time_sec: float
    test_predict_time_sec: float
    mean_epoch_sec: float
    notes: Dict[str, object]


def save_classification_outputs(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    status_cols: List[str],
) -> Dict[str, float]:
    pred_rows = {"row_index": np.arange(len(y_true))}
    metric_rows = []
    for i, col in enumerate(status_cols):
        yt = y_true[:, i].astype(int)
        yp = y_pred[:, i].astype(int)
        pp = y_prob[:, i].astype(float)
        pred_rows[f"true__{col}"] = yt
        pred_rows[f"pred__{col}"] = yp
        pred_rows[f"prob__{col}"] = pp
        metric_rows.append({
            "device": device_from_target(col),
            "target": col,
            "F1": float(f1_score(yt, yp, zero_division=0)),
            "MCC": float(matthews_corrcoef(yt, yp)) if (len(np.unique(yt)) > 1 or len(np.unique(yp)) > 1) else 0.0,
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "preds" / "fold0_TEST_status_predictions.csv", index=False)

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out_dir / "reports" / "fold0_TEST_per_device_f1_mcc.csv", index=False)
    metric_df.to_csv(out_dir / "reports" / "fold0_TEST_per_device.csv", index=False)

    summary = {
        "accuracy": float((y_true.astype(int) == y_pred.astype(int)).mean()),
        "f1_macro": float(f1_score(y_true.astype(int), y_pred.astype(int), average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true.astype(int), y_pred.astype(int), average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true.astype(int), y_pred.astype(int), average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true.astype(int), y_pred.astype(int))),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "reports" / "fold0_TEST_cls_metrics.csv", index=False)
    return summary


def regression_metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)
    if len(yt) < 2:
        return {
            "R2": float("nan"),
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "SAE": float("nan"),
            "NDE": float("nan"),
        }
    return {
        "R2": float(r2_score(yt, yp)),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
        "SAE": float(np.sum(np.abs(yt - yp))),
        "NDE": compute_nde(yt, yp),
    }


def save_regression_outputs(
    out_dir: Path,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    current_cols: List[str],
) -> Dict[str, float]:
    pred_rows = {"row_index": np.arange(len(y_true_raw))}
    metric_rows = []
    for j, col in enumerate(current_cols):
        yt = y_true_raw[:, j].astype(float)
        yp = y_pred_raw[:, j].astype(float)
        pred_rows[f"true__{col}"] = yt
        pred_rows[f"pred__{col}"] = yp
        pred_rows[f"abs_error__{col}"] = np.abs(yt - yp)
        metric_rows.append({
            "device": device_from_target(col),
            "target": col,
            **regression_metric_dict(yt, yp),
        })

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "preds" / "fold0_TEST_current_predictions.csv", index=False)

    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(out_dir / "reports" / "fold0_TEST_current_per_device_reg_metrics.csv", index=False)

    total_metrics = regression_metric_dict(y_true_raw.ravel(), y_pred_raw.ravel())
    pd.DataFrame([total_metrics]).to_csv(out_dir / "reports" / "fold0_TEST_current_reg_metrics.csv", index=False)
    return {
        "current_mae": total_metrics["MAE"],
        "current_rmse": total_metrics["RMSE"],
        "current_sae": total_metrics["SAE"],
        "current_r2": total_metrics["R2"],
        "current_nde": total_metrics["NDE"],
    }


def write_final_summary(
    out_dir: Path,
    model_key: str,
    label: str,
    variant: str,
    cls_summary: Dict[str, float],
    reg_summary: Dict[str, float],
    extras: Optional[Dict[str, object]] = None,
) -> None:
    row = {
        "model": model_key,
        "variant": variant,
        "label": label,
        "test_rows": int(extras.get("test_rows", 0)) if extras else 0,
        "test_accuracy": cls_summary["accuracy"],
        "test_f1_macro": cls_summary["f1_macro"],
        "test_precision_macro": cls_summary["precision_macro"],
        "test_recall_macro": cls_summary["recall_macro"],
        "test_hamming_loss": cls_summary["hamming_loss"],
        "test_current_mae": reg_summary["current_mae"],
        "test_current_rmse": reg_summary["current_rmse"],
        "test_current_sae": reg_summary["current_sae"],
        "test_current_r2": reg_summary["current_r2"],
        "test_current_nde": reg_summary["current_nde"],
    }
    if extras:
        row.update(extras)
    pd.DataFrame([row]).to_csv(out_dir / "reports" / f"_FINAL_TEST_{model_key}.csv", index=False)


def write_runtime_summary(
    out_dir: Path,
    algorithm: str,
    training_param: str,
    time_per_epoch_sec: float,
    decomposition_time_sec: float,
) -> None:
    payload = {
        "algorithm": algorithm,
        "training_param": training_param,
        "time_per_epoch_sec": None if not np.isfinite(time_per_epoch_sec) else float(time_per_epoch_sec),
        "time_per_epoch_label": "nan" if not np.isfinite(time_per_epoch_sec) else f"{time_per_epoch_sec:.4f}",
        "decomposition_time_sec": float(decomposition_time_sec),
        "decomposition_time_label": f"{decomposition_time_sec:.4f}",
    }
    (out_dir / "reports" / "paper_runtime_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_dynamic_k(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k_float = rng.uniform(float(CONFIG["dyn_knn_k_min"]), float(CONFIG["dyn_knn_k_max"]), size=n_samples)
    k = np.rint(k_float).astype(int)
    return np.clip(k, int(CONFIG["dyn_knn_k_min"]), int(CONFIG["dyn_knn_k_max"]))


def fit_dynamic_knn_regression(X_train: np.ndarray, y_train_raw: np.ndarray) -> Dict[str, object]:
    nn_index = NearestNeighbors(
        n_neighbors=int(CONFIG["dyn_knn_max_neighbors"]) + 1,
        metric=CONFIG["dyn_knn_metric"],
        n_jobs=1,
    )
    nn_index.fit(X_train.astype(np.float32))
    return {
        "nn": nn_index,
        "y_train": y_train_raw.astype(np.float32),
        "eps": 1e-8,
    }


def predict_dynamic_knn_regression(
    bundle: Dict[str, object],
    X_query: np.ndarray,
    *,
    seed: int,
    exclude_self: bool = False,
) -> np.ndarray:
    nn_index: NearestNeighbors = bundle["nn"]  # type: ignore[assignment]
    y_train = np.asarray(bundle["y_train"], dtype=np.float32)
    eps = float(bundle["eps"])
    max_k = int(CONFIG["dyn_knn_max_neighbors"])

    n_request = max_k + 1 if exclude_self else max_k
    dists, idx = nn_index.kneighbors(X_query.astype(np.float32), n_neighbors=n_request, return_distance=True)
    if exclude_self:
        dists = dists[:, 1:]
        idx = idx[:, 1:]

    k_dyn = sample_dynamic_k(len(X_query), seed=seed)
    mask = (np.arange(max_k)[None, :] < k_dyn[:, None]).astype(np.float32)
    inv_dist = 1.0 / np.maximum(dists, eps)
    weights = inv_dist * mask
    denom = np.maximum(np.sum(weights, axis=1, keepdims=True), eps)
    pred = np.sum(weights[:, :, None] * y_train[idx], axis=1) / denom
    return pred.astype(np.float32)


def fit_dynamic_knn_classification(X_train: np.ndarray, y_train_status: np.ndarray) -> Dict[str, object]:
    nn_index = NearestNeighbors(
        n_neighbors=int(CONFIG["dyn_knn_max_neighbors"]) + 1,
        metric=CONFIG["dyn_knn_metric"],
        n_jobs=1,
    )
    nn_index.fit(X_train.astype(np.float32))
    return {
        "nn": nn_index,
        "y_train": y_train_status.astype(np.int8),
        "eps": 1e-8,
    }


def predict_dynamic_knn_classification(
    bundle: Dict[str, object],
    X_query: np.ndarray,
    *,
    seed: int,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    nn_index: NearestNeighbors = bundle["nn"]  # type: ignore[assignment]
    y_train = np.asarray(bundle["y_train"], dtype=np.int8)
    eps = float(bundle["eps"])
    max_k = int(CONFIG["dyn_knn_max_neighbors"])

    n_request = max_k + 1 if exclude_self else max_k
    dists, idx = nn_index.kneighbors(X_query.astype(np.float32), n_neighbors=n_request, return_distance=True)
    if exclude_self:
        dists = dists[:, 1:]
        idx = idx[:, 1:]

    k_dyn = sample_dynamic_k(len(X_query), seed=seed)
    labels = y_train[idx]
    mask = (np.arange(max_k)[None, :] < k_dyn[:, None]).astype(np.float32)
    inv_dist = 1.0 / np.maximum(dists, eps)
    weights = inv_dist[:, :, None] * mask[:, :, None]
    w_on = np.sum(weights * labels, axis=1)
    w_off = np.sum(weights * (1 - labels), axis=1)
    denom = w_on + w_off
    p_on = np.divide(
        w_on,
        denom,
        out=np.full_like(w_on, 0.5, dtype=np.float32),
        where=denom > eps,
    ).astype(np.float32)
    pred = (p_on >= float(CONFIG["classification_threshold"])).astype(int)
    return p_on, pred


def fit_fixed_knn_regression(X_train: np.ndarray, y_train_raw: np.ndarray) -> KNeighborsRegressor:
    reg = KNeighborsRegressor(
        n_neighbors=int(CONFIG["fixed_knn_k"]),
        weights="distance",
        metric="euclidean",
        n_jobs=1,
    )
    reg.fit(X_train.astype(np.float32), y_train_raw.astype(np.float32))
    return reg


def build_classification_cnn_backbone(n_features: int, embedding_dim: int) -> base.Model:
    inp = base.Input(shape=(n_features,), name="cls_input_features")
    x = base.Reshape((n_features, 1), name="cls_reshape")(inp)
    x = base.Conv1D(filters=30, kernel_size=10, strides=1, padding="same", activation="relu", name="cls_conv1")(x)
    x = base.Conv1D(filters=40, kernel_size=5, strides=1, padding="same", activation="relu", name="cls_conv2")(x)
    x = base.Conv1D(filters=32, kernel_size=1, strides=1, padding="same", activation="relu", name="cls_conv3")(x)
    x = base.Flatten(name="cls_flatten")(x)
    x = base.Dense(128, activation="relu", name="cls_mlp1")(x)
    x = base.Dropout(0.3, name="cls_dropout1")(x)
    x = base.Dense(embedding_dim, activation="relu", name="cls_embedding")(x)
    return base.Model(inputs=inp, outputs=x, name="requested_cls_backbone")


def build_classification_cnn_pretrain(n_features: int, n_status: int, embedding_dim: int) -> base.Model:
    backbone = build_classification_cnn_backbone(n_features, embedding_dim)
    out = base.Dense(n_status, activation="sigmoid", name="cls_pretrain_output")(backbone.output)
    return base.Model(backbone.input, out, name="requested_cls_pretrain")


def build_regression_cnn_backbone(n_features: int, embedding_dim: int) -> base.Model:
    inp = base.Input(shape=(n_features,), name="reg_input_features")
    x = base.Reshape((n_features, 1), name="reg_reshape")(inp)
    x = base.Conv1D(filters=30, kernel_size=10, strides=1, padding="same", activation="relu", name="reg_conv1")(x)
    x = base.Conv1D(filters=40, kernel_size=5, strides=1, padding="same", activation="relu", name="reg_conv2")(x)
    x = base.Conv1D(filters=32, kernel_size=1, strides=1, padding="same", activation="relu", name="reg_conv3")(x)
    x = base.Flatten(name="reg_flatten")(x)
    x = base.Dense(128, activation="relu", name="reg_mlp1")(x)
    x = base.Dropout(0.3, name="reg_dropout1")(x)
    x = base.Dense(embedding_dim, activation="relu", name="reg_embedding")(x)
    return base.Model(inputs=inp, outputs=x, name="requested_reg_backbone")


def build_regression_cnn_pretrain(n_features: int, n_reg: int, embedding_dim: int) -> base.Model:
    backbone = build_regression_cnn_backbone(n_features, embedding_dim)
    hidden = base.Dense(64, activation="relu", name="reg_hidden")(backbone.output)
    out = base.Dense(n_reg, activation="linear", name="reg_pretrain_output")(hidden)
    return base.Model(backbone.input, out, name="requested_reg_pretrain")


def build_reg_dnn_calibrator(input_dim: int, n_reg: int) -> base.Model:
    inp = base.Input(shape=(input_dim,), name="reg_dnn_input")
    x = base.Dense(128, activation="relu", name="reg_dnn_dense128")(inp)
    x = base.Dense(64, activation="relu", name="reg_dnn_dense64")(x)
    x = base.Dense(50, activation="relu", name="reg_dnn_dense50")(x)
    out = base.Dense(n_reg, activation="linear", name="reg_dnn_output")(x)
    model = base.Model(inp, out, name="reg_dnn_calibrator")
    model.compile(optimizer=base.Adam(CONFIG["reg_dnn_lr"]), loss=base.tf.keras.losses.Huber())
    return model


def build_reg_transformer_calibrator(input_dim: int, n_reg: int) -> base.Model:
    inp = base.Input(shape=(input_dim,), name="reg_transformer_input")
    x = base.Reshape((input_dim, 1), name="reg_transformer_reshape")(inp)
    x = base.Dense(64, activation="relu", name="reg_transformer_token_projection")(x)
    attn = base.tf.keras.layers.MultiHeadAttention(
        num_heads=int(CONFIG["reg_transformer_heads"]),
        key_dim=int(CONFIG["reg_transformer_key_dim"]),
        name="reg_transformer_attention",
    )(x, x)
    x = base.tf.keras.layers.LayerNormalization(epsilon=1e-6, name="reg_transformer_norm_1")(
        base.tf.keras.layers.Add(name="reg_transformer_residual_1")([x, attn])
    )
    ff = base.Dense(128, activation="relu", name="reg_transformer_ff_1")(x)
    ff = base.Dense(64, activation="linear", name="reg_transformer_ff_2")(ff)
    x = base.tf.keras.layers.LayerNormalization(epsilon=1e-6, name="reg_transformer_norm_2")(
        base.tf.keras.layers.Add(name="reg_transformer_residual_2")([x, ff])
    )
    x = base.Dense(32, activation="relu", name="reg_transformer_dense_1")(x)
    x = base.Dense(1, activation="linear", name="reg_transformer_dense_2")(x)
    out = base.Reshape((n_reg,), name="reg_transformer_output")(x)
    model = base.Model(inp, out, name="reg_transformer_calibrator")
    model.compile(optimizer=base.Adam(CONFIG["reg_transformer_lr"]), loss=base.tf.keras.losses.Huber())
    return model


def build_cls_dnn_adapter(input_dim: int, n_status: int) -> base.Model:
    inp = base.Input(shape=(input_dim,), name="cls_dnn_input")
    x = base.Dense(128, activation="relu", name="cls_dnn_dense128")(inp)
    x = base.Dense(64, activation="relu", name="cls_dnn_dense64")(x)
    out = base.Dense(n_status, activation="sigmoid", name="cls_dnn_output")(x)
    model = base.Model(inp, out, name="cls_dnn_adapter")
    model.compile(optimizer=base.Adam(CONFIG["cls_dnn_lr"]), loss="binary_crossentropy")
    return model


def build_cls_bp_adapter(input_dim: int, n_status: int) -> base.Model:
    inp = base.Input(shape=(input_dim,), name="cls_bp_input")
    x = base.Dense(int(CONFIG["bp_hidden1"]), activation="relu", name="cls_bp_dense64")(inp)
    x = base.Dense(int(CONFIG["bp_hidden2"]), activation="relu", name="cls_bp_dense32")(x)
    out = base.Dense(n_status, activation="sigmoid", name="cls_bp_output")(x)
    model = base.Model(inp, out, name="cls_bp_adapter")
    model.compile(optimizer=base.Adam(CONFIG["bp_lr"]), loss="binary_crossentropy")
    return model


def build_reg_bp_adapter(input_dim: int, n_reg: int) -> base.Model:
    inp = base.Input(shape=(input_dim,), name="reg_bp_input")
    x = base.Dense(int(CONFIG["bp_hidden1"]), activation="relu", name="reg_bp_dense64")(inp)
    x = base.Dense(int(CONFIG["bp_hidden2"]), activation="relu", name="reg_bp_dense32")(x)
    out = base.Dense(n_reg, activation="linear", name="reg_bp_output")(x)
    model = base.Model(inp, out, name="reg_bp_adapter")
    model.compile(optimizer=base.Adam(CONFIG["bp_lr"]), loss=base.tf.keras.losses.Huber())
    return model


def build_reg_dense1024_adapter(input_dim: int, n_reg: int) -> base.Model:
    inp = base.Input(shape=(input_dim,), name="reg_dense1024_input")
    x = base.Dense(1024, activation="relu", name="reg_dense1024")(inp)
    out = base.Dense(n_reg, activation="linear", name="reg_dense1024_output")(x)
    model = base.Model(inp, out, name="reg_dense1024_adapter")
    model.compile(optimizer=base.Adam(CONFIG["dense1024_lr"]), loss=base.tf.keras.losses.Huber())
    return model


def fit_keras_model(
    model: base.Model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    verbose: int = 0,
) -> Tuple[base.Model, float, float]:
    timer = LocalEpochTimer()
    callbacks = [timer]
    if len(X) >= 20:
        callbacks.append(
            base.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=0,
            )
        )
    t0 = time.perf_counter()
    model.fit(
        X.astype(np.float32),
        y.astype(np.float32),
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2 if len(X) >= 20 else 0.0,
        shuffle=False,
        callbacks=callbacks,
        verbose=verbose,
    )
    fit_time = float(time.perf_counter() - t0)
    mean_epoch = finite_mean([row["time_sec"] for row in timer.rows])
    return model, fit_time, mean_epoch


def tune_classification_thresholds(
    y_true: np.ndarray,
    prob: np.ndarray,
    status_cols: List[str],
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    grid = np.arange(0.05, 0.901, 0.01)
    for i, col in enumerate(status_cols):
        yt = y_true[:, i].astype(int)
        pp = prob[:, i].astype(float)
        best_score = -1.0
        best_threshold = float(CONFIG["classification_threshold"])
        for threshold in grid:
            pred = (pp >= threshold).astype(int)
            score = float(f1_score(yt, pred, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[col] = best_threshold
    return thresholds


def threshold_predictions(prob: np.ndarray, status_cols: List[str], thresholds: Dict[str, float]) -> np.ndarray:
    arr = np.array([float(thresholds.get(col, CONFIG["classification_threshold"])) for col in status_cols], dtype=np.float32)
    return (prob >= arr[None, :]).astype(int)


def finite_row_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_row_mask requires at least one array")
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr).all(axis=1)
    return mask


def tune_status_from_current(
    y_status_true: np.ndarray,
    current_pred_raw: np.ndarray,
    status_cols: List[str],
    current_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    valid_mask = finite_row_mask(y_status_true.astype(np.float32), current_pred_raw)
    y_status_true = y_status_true[valid_mask]
    current_pred_raw = current_pred_raw[valid_mask]
    status_index = {device_from_target(col): i for i, col in enumerate(status_cols)}
    thresholds: Dict[str, Dict[str, float]] = {}
    for j, col in enumerate(current_cols):
        dev = device_from_target(col)
        status_idx = status_index.get(dev)
        if status_idx is None:
            continue
        yt = y_status_true[:, status_idx].astype(int)
        yp_current = current_pred_raw[:, j].astype(float)
        if len(yp_current) == 0:
            thresholds[dev] = {"threshold": 0.0, "scale": 1.0}
            continue
        quantiles = np.unique(np.quantile(yp_current, np.linspace(0.00, 0.99, 80)))
        quantiles = np.concatenate([[0.0], quantiles])
        best_score = -1.0
        best_threshold = float(np.median(quantiles))
        for threshold in quantiles:
            pred = (yp_current >= threshold).astype(int)
            score = float(f1_score(yt, pred, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        scale = float(np.std(yp_current))
        thresholds[dev] = {"threshold": best_threshold, "scale": max(scale, 1e-6)}
    return thresholds


def status_from_current_predictions(
    current_pred_raw: np.ndarray,
    status_cols: List[str],
    current_cols: List[str],
    threshold_info: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    current_index = {device_from_target(col): i for i, col in enumerate(current_cols)}
    prob = np.full((len(current_pred_raw), len(status_cols)), 0.5, dtype=np.float32)
    pred = np.zeros((len(current_pred_raw), len(status_cols)), dtype=int)
    for i, status_col in enumerate(status_cols):
        dev = device_from_target(status_col)
        current_idx = current_index.get(dev)
        if current_idx is None:
            continue
        info = threshold_info.get(dev, {"threshold": 0.0, "scale": 1.0})
        thr = float(info["threshold"])
        scale = float(info["scale"])
        raw = current_pred_raw[:, current_idx].astype(np.float32)
        p = sigmoid((raw - thr) / max(scale, 1e-6)).astype(np.float32)
        prob[:, i] = p
        pred[:, i] = (raw >= thr).astype(int)
    return prob, pred


def align_prob_on_to_current(
    p_on_status: np.ndarray,
    status_cols: List[str],
    current_cols: List[str],
) -> np.ndarray:
    status_index = {device_from_target(col): i for i, col in enumerate(status_cols)}
    out = np.full((len(p_on_status), len(current_cols)), 0.5, dtype=np.float32)
    for j, col in enumerate(current_cols):
        dev = device_from_target(col)
        idx = status_index.get(dev)
        if idx is not None:
            out[:, j] = p_on_status[:, idx].astype(np.float32)
    return out


def status_gate_for_current(
    status_pred: np.ndarray,
    status_cols: List[str],
    current_cols: List[str],
) -> np.ndarray:
    status_index = {device_from_target(col): i for i, col in enumerate(status_cols)}
    gate = np.ones((len(status_pred), len(current_cols)), dtype=np.float32)
    for j, col in enumerate(current_cols):
        dev = device_from_target(col)
        idx = status_index.get(dev)
        if idx is not None:
            gate[:, j] = status_pred[:, idx].astype(np.float32)
    return gate


def build_top_level_summary(
    model_key: str,
    label: str,
    cls_summary: Dict[str, float],
    reg_summary: Dict[str, float],
) -> Dict[str, object]:
    return {
        "model": model_key,
        "label": label,
        "test_accuracy": cls_summary["accuracy"],
        "test_f1_macro": cls_summary["f1_macro"],
        "test_current_r2": reg_summary["current_r2"],
        "test_current_mae": reg_summary["current_mae"],
        "test_current_sae": reg_summary["current_sae"],
        "test_current_nde": reg_summary["current_nde"],
    }


def generate_dynknn_regression_oof_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_raw: np.ndarray,
    *,
    seed_offset: int,
) -> ComponentResult:
    tscv = TimeSeriesSplit(n_splits=int(CONFIG["n_splits"]))
    oof = np.full_like(y_train_raw, np.nan, dtype=np.float32)
    fit_time = 0.0
    for fold_num, (tr_idx, vl_idx) in enumerate(tscv.split(X_train), start=1):
        t0 = time.perf_counter()
        bundle = fit_dynamic_knn_regression(X_train[tr_idx], y_train_raw[tr_idx])
        fit_time += time.perf_counter() - t0
        oof[vl_idx] = predict_dynamic_knn_regression(
            bundle,
            X_train[vl_idx],
            seed=seed_offset + fold_num,
            exclude_self=False,
        )

    t0 = time.perf_counter()
    final_bundle = fit_dynamic_knn_regression(X_train, y_train_raw)
    fit_time += time.perf_counter() - t0
    t0 = time.perf_counter()
    test_pred = predict_dynamic_knn_regression(final_bundle, X_test, seed=seed_offset + 1000, exclude_self=False)
    test_predict_time = time.perf_counter() - t0
    return ComponentResult(oof=oof, test=test_pred, fit_time_sec=fit_time, test_predict_time_sec=test_predict_time, mean_epoch_sec=float("nan"), notes={})


def generate_knn_k7_regression_oof_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_raw: np.ndarray,
) -> ComponentResult:
    tscv = TimeSeriesSplit(n_splits=int(CONFIG["n_splits"]))
    oof = np.full_like(y_train_raw, np.nan, dtype=np.float32)
    fit_time = 0.0
    for tr_idx, vl_idx in tscv.split(X_train):
        t0 = time.perf_counter()
        reg = fit_fixed_knn_regression(X_train[tr_idx], y_train_raw[tr_idx])
        fit_time += time.perf_counter() - t0
        oof[vl_idx] = reg.predict(X_train[vl_idx]).astype(np.float32)

    t0 = time.perf_counter()
    final_reg = fit_fixed_knn_regression(X_train, y_train_raw)
    fit_time += time.perf_counter() - t0
    t0 = time.perf_counter()
    test_pred = final_reg.predict(X_test).astype(np.float32)
    test_predict_time = time.perf_counter() - t0
    return ComponentResult(oof=oof, test=test_pred, fit_time_sec=fit_time, test_predict_time_sec=test_predict_time, mean_epoch_sec=float("nan"), notes={})


def generate_conv_dynknn_regression_oof_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_scaled: np.ndarray,
    y_train_raw: np.ndarray,
    *,
    seed_offset: int,
) -> ComponentResult:
    tscv = TimeSeriesSplit(n_splits=int(CONFIG["n_splits"]))
    n_features = X_train.shape[1]
    n_reg = y_train_scaled.shape[1]
    emb_dim = int(CONFIG["embedding_dim"])
    oof = np.full_like(y_train_raw, np.nan, dtype=np.float32)
    fit_time = 0.0
    epoch_means: List[float] = []

    for fold_num, (tr_idx, vl_idx) in enumerate(tscv.split(X_train), start=1):
        base.tf.keras.backend.clear_session()
        pretrain = build_regression_cnn_pretrain(n_features, n_reg, emb_dim)
        pretrain.compile(optimizer=base.Adam(CONFIG["cnn_lr"]), loss=base.tf.keras.losses.Huber())
        timer = LocalEpochTimer()
        callbacks = [
            timer,
            base.EarlyStopping(
                monitor="val_loss",
                patience=int(CONFIG["cnn_patience"]),
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=0,
            ),
        ]
        t0 = time.perf_counter()
        pretrain.fit(
            X_train[tr_idx],
            y_train_scaled[tr_idx],
            validation_data=(X_train[vl_idx], y_train_scaled[vl_idx]),
            epochs=int(CONFIG["cnn_epochs"]),
            batch_size=int(CONFIG["cnn_batch"]),
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )
        fit_time += time.perf_counter() - t0
        epoch_means.append(finite_mean([row["time_sec"] for row in timer.rows]))

        backbone = base.Model(pretrain.input, pretrain.get_layer("reg_embedding").output)
        emb_tr = backbone.predict(X_train[tr_idx], batch_size=2048, verbose=0)
        emb_vl = backbone.predict(X_train[vl_idx], batch_size=2048, verbose=0)
        bundle = fit_dynamic_knn_regression(emb_tr, y_train_raw[tr_idx])
        oof[vl_idx] = predict_dynamic_knn_regression(bundle, emb_vl, seed=seed_offset + fold_num, exclude_self=False)

    base.tf.keras.backend.clear_session()
    pretrain = build_regression_cnn_pretrain(n_features, n_reg, emb_dim)
    pretrain.compile(optimizer=base.Adam(CONFIG["cnn_lr"]), loss=base.tf.keras.losses.Huber())
    timer = LocalEpochTimer()
    callbacks = [
        timer,
        base.EarlyStopping(
            monitor="val_loss",
            patience=int(CONFIG["cnn_patience"]),
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=0,
        ),
    ]
    val_cut = int(len(X_train) * 0.90)
    t0 = time.perf_counter()
    pretrain.fit(
        X_train[:val_cut],
        y_train_scaled[:val_cut],
        validation_data=(X_train[val_cut:], y_train_scaled[val_cut:]),
        epochs=int(CONFIG["cnn_epochs"]),
        batch_size=int(CONFIG["cnn_batch"]),
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )
    fit_time += time.perf_counter() - t0
    epoch_means.append(finite_mean([row["time_sec"] for row in timer.rows]))
    backbone = base.Model(pretrain.input, pretrain.get_layer("reg_embedding").output)
    emb_train = backbone.predict(X_train, batch_size=2048, verbose=0)
    emb_test = backbone.predict(X_test, batch_size=2048, verbose=0)
    final_bundle = fit_dynamic_knn_regression(emb_train, y_train_raw)
    t0 = time.perf_counter()
    test_pred = predict_dynamic_knn_regression(final_bundle, emb_test, seed=seed_offset + 1000, exclude_self=False)
    test_predict_time = time.perf_counter() - t0
    return ComponentResult(oof=oof, test=test_pred, fit_time_sec=fit_time, test_predict_time_sec=test_predict_time, mean_epoch_sec=finite_mean(epoch_means), notes={})


def generate_conv_dynknn_classification_oof_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train_status: np.ndarray,
    *,
    seed_offset: int,
) -> ComponentResult:
    tscv = TimeSeriesSplit(n_splits=int(CONFIG["n_splits"]))
    n_features = X_train.shape[1]
    n_status = y_train_status.shape[1]
    emb_dim = int(CONFIG["embedding_dim"])
    oof = np.full((len(X_train), n_status), np.nan, dtype=np.float32)
    fit_time = 0.0
    epoch_means: List[float] = []

    for fold_num, (tr_idx, vl_idx) in enumerate(tscv.split(X_train), start=1):
        base.tf.keras.backend.clear_session()
        pretrain = build_classification_cnn_pretrain(n_features, n_status, emb_dim)
        pretrain.compile(optimizer=base.Adam(CONFIG["cnn_lr"]), loss="binary_crossentropy")
        timer = LocalEpochTimer()
        callbacks = [
            timer,
            base.EarlyStopping(
                monitor="val_loss",
                patience=int(CONFIG["cnn_patience"]),
                min_delta=1e-5,
                restore_best_weights=True,
                verbose=0,
            ),
        ]
        t0 = time.perf_counter()
        pretrain.fit(
            X_train[tr_idx],
            y_train_status[tr_idx].astype(np.float32),
            validation_data=(X_train[vl_idx], y_train_status[vl_idx].astype(np.float32)),
            epochs=int(CONFIG["cnn_epochs"]),
            batch_size=int(CONFIG["cnn_batch"]),
            shuffle=False,
            callbacks=callbacks,
            verbose=0,
        )
        fit_time += time.perf_counter() - t0
        epoch_means.append(finite_mean([row["time_sec"] for row in timer.rows]))

        backbone = base.Model(pretrain.input, pretrain.get_layer("cls_embedding").output)
        emb_tr = backbone.predict(X_train[tr_idx], batch_size=2048, verbose=0)
        emb_vl = backbone.predict(X_train[vl_idx], batch_size=2048, verbose=0)
        bundle = fit_dynamic_knn_classification(emb_tr, y_train_status[tr_idx])
        p_on_vl, _pred_vl = predict_dynamic_knn_classification(bundle, emb_vl, seed=seed_offset + fold_num, exclude_self=False)
        oof[vl_idx] = p_on_vl

    base.tf.keras.backend.clear_session()
    pretrain = build_classification_cnn_pretrain(n_features, n_status, emb_dim)
    pretrain.compile(optimizer=base.Adam(CONFIG["cnn_lr"]), loss="binary_crossentropy")
    timer = LocalEpochTimer()
    callbacks = [
        timer,
        base.EarlyStopping(
            monitor="val_loss",
            patience=int(CONFIG["cnn_patience"]),
            min_delta=1e-5,
            restore_best_weights=True,
            verbose=0,
        ),
    ]
    val_cut = int(len(X_train) * 0.90)
    t0 = time.perf_counter()
    pretrain.fit(
        X_train[:val_cut],
        y_train_status[:val_cut].astype(np.float32),
        validation_data=(X_train[val_cut:], y_train_status[val_cut:].astype(np.float32)),
        epochs=int(CONFIG["cnn_epochs"]),
        batch_size=int(CONFIG["cnn_batch"]),
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )
    fit_time += time.perf_counter() - t0
    epoch_means.append(finite_mean([row["time_sec"] for row in timer.rows]))
    backbone = base.Model(pretrain.input, pretrain.get_layer("cls_embedding").output)
    emb_train = backbone.predict(X_train, batch_size=2048, verbose=0)
    emb_test = backbone.predict(X_test, batch_size=2048, verbose=0)
    final_bundle = fit_dynamic_knn_classification(emb_train, y_train_status)
    t0 = time.perf_counter()
    p_on_test, _pred_test = predict_dynamic_knn_classification(final_bundle, emb_test, seed=seed_offset + 1000, exclude_self=False)
    test_predict_time = time.perf_counter() - t0
    return ComponentResult(oof=oof, test=p_on_test, fit_time_sec=fit_time, test_predict_time_sec=test_predict_time, mean_epoch_sec=finite_mean(epoch_means), notes={})


def train_regression_calibrator(
    builder_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[base.Model, float, float]:
    valid_mask = finite_row_mask(X_train, y_train)
    base.tf.keras.backend.clear_session()
    model = builder_fn(X_train.shape[1], y_train.shape[1])
    return fit_keras_model(
        model,
        X_train[valid_mask],
        y_train[valid_mask],
        epochs=int(CONFIG["reg_dnn_epochs"]),
        batch_size=int(CONFIG["reg_dnn_batch"]),
        patience=int(CONFIG["reg_dnn_patience"]),
    )


def train_cls_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[base.Model, float, float]:
    valid_mask = finite_row_mask(X_train, y_train.astype(np.float32))
    base.tf.keras.backend.clear_session()
    model = build_cls_dnn_adapter(X_train.shape[1], y_train.shape[1])
    return fit_keras_model(
        model,
        X_train[valid_mask],
        y_train[valid_mask],
        epochs=int(CONFIG["cls_dnn_epochs"]),
        batch_size=int(CONFIG["cls_dnn_batch"]),
        patience=int(CONFIG["cls_dnn_patience"]),
    )


def train_reg_transformer_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[base.Model, float, float]:
    valid_mask = finite_row_mask(X_train, y_train)
    base.tf.keras.backend.clear_session()
    model = build_reg_transformer_calibrator(X_train.shape[1], y_train.shape[1])
    return fit_keras_model(
        model,
        X_train[valid_mask],
        y_train[valid_mask],
        epochs=int(CONFIG["reg_transformer_epochs"]),
        batch_size=int(CONFIG["reg_transformer_batch"]),
        patience=int(CONFIG["reg_transformer_patience"]),
    )


def train_dense1024_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[base.Model, float, float]:
    valid_mask = finite_row_mask(X_train, y_train)
    base.tf.keras.backend.clear_session()
    model = build_reg_dense1024_adapter(X_train.shape[1], y_train.shape[1])
    return fit_keras_model(
        model,
        X_train[valid_mask],
        y_train[valid_mask],
        epochs=int(CONFIG["dense1024_epochs"]),
        batch_size=int(CONFIG["dense1024_batch"]),
        patience=int(CONFIG["dense1024_patience"]),
    )


def train_cls_bp_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[base.Model, float, float]:
    valid_mask = finite_row_mask(X_train, y_train.astype(np.float32))
    base.tf.keras.backend.clear_session()
    model = build_cls_bp_adapter(X_train.shape[1], y_train.shape[1])
    return fit_keras_model(
        model,
        X_train[valid_mask],
        y_train[valid_mask],
        epochs=int(CONFIG["bp_epochs"]),
        batch_size=int(CONFIG["bp_batch"]),
        patience=int(CONFIG["bp_patience"]),
    )


def train_reg_bp_adapter(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[base.Model, float, float]:
    valid_mask = finite_row_mask(X_train, y_train)
    base.tf.keras.backend.clear_session()
    model = build_reg_bp_adapter(X_train.shape[1], y_train.shape[1])
    return fit_keras_model(
        model,
        X_train[valid_mask],
        y_train[valid_mask],
        epochs=int(CONFIG["bp_epochs"]),
        batch_size=int(CONFIG["bp_batch"]),
        patience=int(CONFIG["bp_patience"]),
    )


def save_model_spec(
    out_dir: Path,
    label: str,
    training_param: str,
    notes: Optional[Dict[str, object]] = None,
) -> None:
    payload = {
        "model": label,
        "training_param": training_param,
        "requested_suite": SUITE_NAME,
    }
    if notes:
        payload.update(notes)
    (out_dir / "reports" / "model_spec.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def finalize_model(
    model_key: str,
    label: str,
    variant: str,
    training_param: str,
    y_test_status: np.ndarray,
    status_pred: np.ndarray,
    status_prob: np.ndarray,
    y_test_current_raw: np.ndarray,
    current_pred_raw: np.ndarray,
    status_cols: List[str],
    current_cols: List[str],
    *,
    fit_time_sec: float,
    predict_time_sec: float,
    mean_epoch_sec: float,
    extra_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    out_dir, _saved_dir = reset_model_dirs(model_key)
    save_model_spec(out_dir, label, training_param, notes=extra_summary)
    cls_summary = save_classification_outputs(out_dir, y_test_status, status_pred, status_prob, status_cols)
    reg_summary = save_regression_outputs(out_dir, y_test_current_raw, current_pred_raw, current_cols)
    extras = {"test_rows": int(len(y_test_status))}
    if extra_summary:
        extras.update(extra_summary)
    write_final_summary(out_dir, model_key, label, variant, cls_summary, reg_summary, extras=extras)
    write_runtime_summary(out_dir, label, training_param, mean_epoch_sec, predict_time_sec)
    return build_top_level_summary(model_key, label, cls_summary, reg_summary)


def predict_with_timing(model: base.Model, X: np.ndarray) -> Tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    pred = model.predict(X, batch_size=2048, verbose=0).astype(np.float32)
    return pred, float(time.perf_counter() - t0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run requested DynKNN suite models.")
    parser.add_argument(
        "models",
        nargs="*",
        help="Optional model keys to run. Default: run every model in the suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_keys = list(MODEL_SPEC_BY_KEY)
    requested = set(args.models) if args.models else set(all_keys)
    invalid = sorted(requested - set(all_keys))
    if invalid:
        raise SystemExit(f"Unknown model key(s): {invalid}. Available: {all_keys}")

    print(f"[suite] Running {SUITE_NAME}")
    print(f"[suite] Requested models: {sorted(requested)}")
    X_train, X_test, Y_train, Y_test, input_cols, target_cols, scalers_Y, devices = base.load_data()
    status_cols = [c for c in target_cols if c.startswith("y_status__")]
    current_cols = [c for c in target_cols if c.startswith("y_current__")]
    y_train_status = Y_train[status_cols].values.astype(int)
    y_test_status = Y_test[status_cols].values.astype(int)
    y_train_current_scaled = Y_train[current_cols].values.astype(np.float32)
    y_test_current_scaled = Y_test[current_cols].values.astype(np.float32)
    Y_train_raw = base.inverse_df(Y_train.copy(), scalers_Y)
    Y_test_raw = base.inverse_df(Y_test.copy(), scalers_Y)
    y_train_current_raw = Y_train_raw[current_cols].values.astype(np.float32)
    y_test_current_raw = Y_test_raw[current_cols].values.astype(np.float32)

    need_agg_dyn = bool({"dynknn_reg_only", "dynknn_dnn_reg_only", "dynknn_transformer_reg_only"} & requested)
    need_knn_k7 = "knn_k7_reg_driven" in requested
    need_conv_dyn_reg = bool({"conv_dynknn_dnn_reg_only", "parallel_p_on", "parallel_hard_gate", "parallel_hard_gate_bp_both"} & requested)
    need_conv_dyn_cls = bool({"parallel_p_on", "parallel_hard_gate", "parallel_hard_gate_bp_both"} & requested)

    agg_dyn: Optional[ComponentResult] = None
    knn_k7: Optional[ComponentResult] = None
    conv_dyn_reg: Optional[ComponentResult] = None
    conv_dyn_cls: Optional[ComponentResult] = None

    if need_agg_dyn:
        print("[suite] Shared base: aggregate DynKNN regression")
        agg_dyn = generate_dynknn_regression_oof_test(X_train, X_test, y_train_current_raw, seed_offset=100)

    if need_knn_k7:
        print("[suite] Shared base: fixed KNN k=7 regression")
        knn_k7 = generate_knn_k7_regression_oof_test(X_train, X_test, y_train_current_raw)

    if need_conv_dyn_reg:
        print("[suite] Shared base: Conv DynKNN regression")
        conv_dyn_reg = generate_conv_dynknn_regression_oof_test(
            X_train,
            X_test,
            y_train_current_scaled,
            y_train_current_raw,
            seed_offset=300,
        )

    if need_conv_dyn_cls:
        print("[suite] Shared base: Conv DynKNN classification")
        conv_dyn_cls = generate_conv_dynknn_classification_oof_test(
            X_train,
            X_test,
            y_train_status,
            seed_offset=500,
        )

    summary_rows: List[Dict[str, object]] = []

    # 1) DynKNN regression only
    if "dynknn_reg_only" in requested:
        assert agg_dyn is not None
        spec = MODEL_SPEC_BY_KEY["dynknn_reg_only"]
        dyn_thresholds = tune_status_from_current(y_train_status, agg_dyn.oof, status_cols, current_cols)  # type: ignore[arg-type]
        dyn_prob, dyn_pred = status_from_current_predictions(agg_dyn.test, status_cols, current_cols, dyn_thresholds)
        summary_rows.append(
            finalize_model(
                "dynknn_reg_only",
                spec["label"],
                spec["variant"],
                spec["training_param"],
                y_test_status,
                dyn_pred,
                dyn_prob,
                y_test_current_raw,
                agg_dyn.test,
                status_cols,
                current_cols,
                fit_time_sec=agg_dyn.fit_time_sec,
                predict_time_sec=agg_dyn.test_predict_time_sec,
                mean_epoch_sec=agg_dyn.mean_epoch_sec,
                extra_summary={"fusion": "regression_only_current_threshold_status"},
            )
        )

    # 2) DynKNN + DNN regression only
    if "dynknn_dnn_reg_only" in requested:
        assert agg_dyn is not None
        spec = MODEL_SPEC_BY_KEY["dynknn_dnn_reg_only"]
        print("[suite] Calibrator: DynKNN + DNN regression only")
        reg_dnn_model, reg_dnn_fit, reg_dnn_epoch = train_regression_calibrator(build_reg_dnn_calibrator, agg_dyn.oof, y_train_current_raw)  # type: ignore[arg-type]
        oof_dyn_dnn = reg_dnn_model.predict(agg_dyn.oof, batch_size=2048, verbose=0).astype(np.float32)  # type: ignore[arg-type]
        test_dyn_dnn, dyn_dnn_predict_time = predict_with_timing(reg_dnn_model, agg_dyn.test)
        dyn_dnn_thresholds = tune_status_from_current(y_train_status, oof_dyn_dnn, status_cols, current_cols)
        dyn_dnn_prob, dyn_dnn_pred = status_from_current_predictions(test_dyn_dnn, status_cols, current_cols, dyn_dnn_thresholds)
        summary_rows.append(
            finalize_model(
                "dynknn_dnn_reg_only",
                spec["label"],
                spec["variant"],
                spec["training_param"],
                y_test_status,
                dyn_dnn_pred,
                dyn_dnn_prob,
                y_test_current_raw,
                test_dyn_dnn,
                status_cols,
                current_cols,
                fit_time_sec=agg_dyn.fit_time_sec + reg_dnn_fit,
                predict_time_sec=agg_dyn.test_predict_time_sec + dyn_dnn_predict_time,
                mean_epoch_sec=finite_mean([agg_dyn.mean_epoch_sec, reg_dnn_epoch]),
                extra_summary={"fusion": "dynknn_then_dnn_regression"},
            )
        )

    # 3) DynKNN + Transformer regression only
    if "dynknn_transformer_reg_only" in requested:
        assert agg_dyn is not None
        spec = MODEL_SPEC_BY_KEY["dynknn_transformer_reg_only"]
        print("[suite] Calibrator: DynKNN + Transformer regression only")
        reg_transformer_model, reg_transformer_fit, reg_transformer_epoch = train_reg_transformer_adapter(
            agg_dyn.oof,
            y_train_current_raw,
        )  # type: ignore[arg-type]
        oof_dyn_transformer = reg_transformer_model.predict(agg_dyn.oof, batch_size=2048, verbose=0).astype(np.float32)  # type: ignore[arg-type]
        test_dyn_transformer, dyn_transformer_predict_time = predict_with_timing(reg_transformer_model, agg_dyn.test)
        dyn_transformer_thresholds = tune_status_from_current(y_train_status, oof_dyn_transformer, status_cols, current_cols)
        dyn_transformer_prob, dyn_transformer_pred = status_from_current_predictions(
            test_dyn_transformer,
            status_cols,
            current_cols,
            dyn_transformer_thresholds,
        )
        summary_rows.append(
            finalize_model(
                "dynknn_transformer_reg_only",
                spec["label"],
                spec["variant"],
                spec["training_param"],
                y_test_status,
                dyn_transformer_pred,
                dyn_transformer_prob,
                y_test_current_raw,
                test_dyn_transformer,
                status_cols,
                current_cols,
                fit_time_sec=agg_dyn.fit_time_sec + reg_transformer_fit,
                predict_time_sec=agg_dyn.test_predict_time_sec + dyn_transformer_predict_time,
                mean_epoch_sec=finite_mean([agg_dyn.mean_epoch_sec, reg_transformer_epoch]),
                extra_summary={"fusion": "dynknn_then_transformer_regression"},
            )
        )

    # 4) Conv + DynKNN + DNN regression only
    if "conv_dynknn_dnn_reg_only" in requested:
        assert conv_dyn_reg is not None
        spec = MODEL_SPEC_BY_KEY["conv_dynknn_dnn_reg_only"]
        print("[suite] Calibrator: Conv DynKNN + DNN regression only")
        conv_reg_dnn_model, conv_reg_dnn_fit, conv_reg_dnn_epoch = train_regression_calibrator(build_reg_dnn_calibrator, conv_dyn_reg.oof, y_train_current_raw)  # type: ignore[arg-type]
        oof_conv_dyn_dnn = conv_reg_dnn_model.predict(conv_dyn_reg.oof, batch_size=2048, verbose=0).astype(np.float32)  # type: ignore[arg-type]
        test_conv_dyn_dnn, conv_dyn_dnn_predict_time = predict_with_timing(conv_reg_dnn_model, conv_dyn_reg.test)
        conv_dyn_dnn_thresholds = tune_status_from_current(y_train_status, oof_conv_dyn_dnn, status_cols, current_cols)
        conv_dyn_dnn_prob, conv_dyn_dnn_pred = status_from_current_predictions(test_conv_dyn_dnn, status_cols, current_cols, conv_dyn_dnn_thresholds)
        summary_rows.append(
            finalize_model(
                "conv_dynknn_dnn_reg_only",
                spec["label"],
                spec["variant"],
                spec["training_param"],
                y_test_status,
                conv_dyn_dnn_pred,
                conv_dyn_dnn_prob,
                y_test_current_raw,
                test_conv_dyn_dnn,
                status_cols,
                current_cols,
                fit_time_sec=conv_dyn_reg.fit_time_sec + conv_reg_dnn_fit,
                predict_time_sec=conv_dyn_reg.test_predict_time_sec + conv_dyn_dnn_predict_time,
                mean_epoch_sec=finite_mean([conv_dyn_reg.mean_epoch_sec, conv_reg_dnn_epoch]),
                extra_summary={"fusion": "conv_dynknn_then_dnn_regression"},
            )
        )

    # Shared parallel branch adapters
    if {"parallel_p_on", "parallel_hard_gate"} & requested:
        assert conv_dyn_cls is not None and conv_dyn_reg is not None
        print("[suite] Parallel top branch classification DNN adapter")
        cls_adapter, cls_fit_time, cls_epoch_time = train_cls_adapter(conv_dyn_cls.oof, y_train_status)  # type: ignore[arg-type]
        oof_cls_prob = cls_adapter.predict(conv_dyn_cls.oof, batch_size=2048, verbose=0).astype(np.float32)  # type: ignore[arg-type]
        test_cls_prob, cls_predict_time = predict_with_timing(cls_adapter, conv_dyn_cls.test)
        cls_valid_mask = finite_row_mask(oof_cls_prob, y_train_status.astype(np.float32))
        cls_thresholds = tune_classification_thresholds(y_train_status[cls_valid_mask], oof_cls_prob[cls_valid_mask], status_cols)
        test_cls_pred = threshold_predictions(test_cls_prob, status_cols, cls_thresholds)

        print("[suite] Parallel bottom branch Dense1024 regression adapter")
        reg_dense1024, reg1024_fit_time, reg1024_epoch_time = train_dense1024_adapter(conv_dyn_reg.oof, y_train_current_raw)  # type: ignore[arg-type]
        test_reg_dense1024, reg1024_predict_time = predict_with_timing(reg_dense1024, conv_dyn_reg.test)

        p_on_current = align_prob_on_to_current(test_cls_prob, status_cols, current_cols)
        hard_gate = status_gate_for_current(test_cls_pred, status_cols, current_cols)

        # 4a) Parallel use p_on
        if "parallel_p_on" in requested:
            spec = MODEL_SPEC_BY_KEY["parallel_p_on"]
            fused_p_on = (p_on_current * test_reg_dense1024).astype(np.float32)
            summary_rows.append(
                finalize_model(
                    "parallel_p_on",
                    spec["label"],
                    spec["variant"],
                    spec["training_param"],
                    y_test_status,
                    test_cls_pred,
                    test_cls_prob,
                    y_test_current_raw,
                    fused_p_on,
                    status_cols,
                    current_cols,
                    fit_time_sec=conv_dyn_cls.fit_time_sec + cls_fit_time + conv_dyn_reg.fit_time_sec + reg1024_fit_time,
                    predict_time_sec=conv_dyn_cls.test_predict_time_sec + cls_predict_time + conv_dyn_reg.test_predict_time_sec + reg1024_predict_time,
                    mean_epoch_sec=finite_mean([conv_dyn_cls.mean_epoch_sec, cls_epoch_time, conv_dyn_reg.mean_epoch_sec, reg1024_epoch_time]),
                    extra_summary={"fusion": "p_on_times_regression_current"},
                )
            )

        # 4b) Parallel no p_on => hard gate
        if "parallel_hard_gate" in requested:
            spec = MODEL_SPEC_BY_KEY["parallel_hard_gate"]
            fused_hard = (hard_gate * test_reg_dense1024).astype(np.float32)
            summary_rows.append(
                finalize_model(
                    "parallel_hard_gate",
                    spec["label"],
                    spec["variant"],
                    spec["training_param"],
                    y_test_status,
                    test_cls_pred,
                    test_cls_prob,
                    y_test_current_raw,
                    fused_hard,
                    status_cols,
                    current_cols,
                    fit_time_sec=conv_dyn_cls.fit_time_sec + cls_fit_time + conv_dyn_reg.fit_time_sec + reg1024_fit_time,
                    predict_time_sec=conv_dyn_cls.test_predict_time_sec + cls_predict_time + conv_dyn_reg.test_predict_time_sec + reg1024_predict_time,
                    mean_epoch_sec=finite_mean([conv_dyn_cls.mean_epoch_sec, cls_epoch_time, conv_dyn_reg.mean_epoch_sec, reg1024_epoch_time]),
                    extra_summary={"fusion": "hard_status_gate_times_regression_current"},
                )
            )

    # 4c) Parallel hard gate with BP on both branches
    if "parallel_hard_gate_bp_both" in requested:
        assert conv_dyn_cls is not None and conv_dyn_reg is not None
        spec = MODEL_SPEC_BY_KEY["parallel_hard_gate_bp_both"]
        print("[suite] Parallel top branch classification BP adapter")
        cls_bp_adapter, cls_bp_fit, cls_bp_epoch = train_cls_bp_adapter(conv_dyn_cls.oof, y_train_status)  # type: ignore[arg-type]
        oof_cls_bp = cls_bp_adapter.predict(conv_dyn_cls.oof, batch_size=2048, verbose=0).astype(np.float32)  # type: ignore[arg-type]
        test_cls_bp, cls_bp_predict_time = predict_with_timing(cls_bp_adapter, conv_dyn_cls.test)

        print("[suite] Parallel top branch classification DNN adapter after BP")
        cls_bp_dnn_adapter, cls_bp_dnn_fit, cls_bp_dnn_epoch = train_cls_adapter(oof_cls_bp, y_train_status)
        oof_cls_prob_bp = cls_bp_dnn_adapter.predict(oof_cls_bp, batch_size=2048, verbose=0).astype(np.float32)
        test_cls_prob_bp, cls_bp_dnn_predict_time = predict_with_timing(cls_bp_dnn_adapter, test_cls_bp)
        cls_bp_valid_mask = finite_row_mask(oof_cls_prob_bp, y_train_status.astype(np.float32))
        cls_bp_thresholds = tune_classification_thresholds(
            y_train_status[cls_bp_valid_mask],
            oof_cls_prob_bp[cls_bp_valid_mask],
            status_cols,
        )
        test_cls_pred_bp = threshold_predictions(test_cls_prob_bp, status_cols, cls_bp_thresholds)

        print("[suite] Parallel bottom branch regression BP adapter")
        reg_bp_adapter, reg_bp_fit, reg_bp_epoch = train_reg_bp_adapter(conv_dyn_reg.oof, y_train_current_raw)  # type: ignore[arg-type]
        oof_reg_bp = reg_bp_adapter.predict(conv_dyn_reg.oof, batch_size=2048, verbose=0).astype(np.float32)  # type: ignore[arg-type]
        test_reg_bp, reg_bp_predict_time = predict_with_timing(reg_bp_adapter, conv_dyn_reg.test)

        print("[suite] Parallel bottom branch Dense1024 regression adapter after BP")
        reg_dense1024_bp, reg1024_bp_fit, reg1024_bp_epoch = train_dense1024_adapter(oof_reg_bp, y_train_current_raw)
        test_reg_dense1024_bp, reg1024_bp_predict_time = predict_with_timing(reg_dense1024_bp, test_reg_bp)

        hard_gate_bp = status_gate_for_current(test_cls_pred_bp, status_cols, current_cols)
        fused_hard_bp = (hard_gate_bp * test_reg_dense1024_bp).astype(np.float32)

        summary_rows.append(
            finalize_model(
                "parallel_hard_gate_bp_both",
                spec["label"],
                spec["variant"],
                spec["training_param"],
                y_test_status,
                test_cls_pred_bp,
                test_cls_prob_bp,
                y_test_current_raw,
                fused_hard_bp,
                status_cols,
                current_cols,
                fit_time_sec=(
                    conv_dyn_cls.fit_time_sec
                    + cls_bp_fit
                    + cls_bp_dnn_fit
                    + conv_dyn_reg.fit_time_sec
                    + reg_bp_fit
                    + reg1024_bp_fit
                ),
                predict_time_sec=(
                    conv_dyn_cls.test_predict_time_sec
                    + cls_bp_predict_time
                    + cls_bp_dnn_predict_time
                    + conv_dyn_reg.test_predict_time_sec
                    + reg_bp_predict_time
                    + reg1024_bp_predict_time
                ),
                mean_epoch_sec=finite_mean([
                    conv_dyn_cls.mean_epoch_sec,
                    cls_bp_epoch,
                    cls_bp_dnn_epoch,
                    conv_dyn_reg.mean_epoch_sec,
                    reg_bp_epoch,
                    reg1024_bp_epoch,
                ]),
                extra_summary={
                    "fusion": "hard_status_gate_times_regression_current",
                    "classification_bp": True,
                    "regression_bp": True,
                },
            )
        )

    # 5) Fixed KNN k=7 reg-driven
    if "knn_k7_reg_driven" in requested:
        assert knn_k7 is not None
        spec = MODEL_SPEC_BY_KEY["knn_k7_reg_driven"]
        knn7_thresholds = tune_status_from_current(y_train_status, knn_k7.oof, status_cols, current_cols)  # type: ignore[arg-type]
        knn7_prob, knn7_pred = status_from_current_predictions(knn_k7.test, status_cols, current_cols, knn7_thresholds)
        summary_rows.append(
            finalize_model(
                "knn_k7_reg_driven",
                spec["label"],
                spec["variant"],
                spec["training_param"],
                y_test_status,
                knn7_pred,
                knn7_prob,
                y_test_current_raw,
                knn_k7.test,
                status_cols,
                current_cols,
                fit_time_sec=knn_k7.fit_time_sec,
                predict_time_sec=knn_k7.test_predict_time_sec,
                mean_epoch_sec=knn_k7.mean_epoch_sec,
                extra_summary={"fusion": "regression_only_current_threshold_status", "fixed_k": int(CONFIG["fixed_knn_k"])},
            )
        )

    summary_path = OUT_ROOT / "requested_suite_summary.json"
    existing_rows: List[Dict[str, object]] = []
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                existing_rows = [row for row in payload if isinstance(row, dict)]
        except Exception:
            existing_rows = []
    merged = {str(row.get("model")): row for row in existing_rows if row.get("model")}
    for row in summary_rows:
        merged[str(row["model"])] = row
    ordered = [merged[key] for key in all_keys if key in merged]
    summary_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")
    print(f"[ok] Summary -> {summary_path}")


if __name__ == "__main__":
    main()
