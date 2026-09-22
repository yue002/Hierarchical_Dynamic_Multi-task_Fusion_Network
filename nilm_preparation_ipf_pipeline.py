from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from openpyxl import load_workbook
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG — I/PF preparation pipeline
# ─────────────────────────────────────────
INPUT_XLSX     = Path(__file__).parent / "_all_31.3-6.4.xlsx"
OUTPUT_DIR     = Path(__file__).parent / "nilm_sync_outputs_ipf"
DIR_CLEANED    = OUTPUT_DIR / "01_cleaned"
DIR_NORMALIZED = OUTPUT_DIR / "02_normalized"
DIR_COMBINED   = OUTPUT_DIR / "03_combined"
DIR_PLOTS      = OUTPUT_DIR / "04_diagnostic_plots"

MASTER_FREQ = "10s"
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# Dataset จริงในไฟล์ _all_31.3-6.4.xlsx
FAN_SHEETS = ["cooling_fan1", "cooling_fan2", "cooling_fan3"]
IPV_SHEETS = ["coolingpad_pump", "fert_pump", "solar"]  # อ่าน I เป็นหลัก, V ไม่ส่งออกเป็น feature
LED_SHEET = "led"
ALL_DEVICES = FAN_SHEETS + IPV_SHEETS + [LED_SHEET]

# ทุก device จะ sync เป็น 10s grid แล้ว interpolate เฉพาะ gap สั้น
INTERP_LIMIT_PER_DEVICE: Dict[str, int] = {
    "cooling_fan1": 6,       # 60 s
    "cooling_fan2": 6,
    "cooling_fan3": 6,
    "coolingpad_pump": 6,    # 60 s
    "fert_pump": 6,
    "solar": 12,             # 120 s, เพราะ native ~60s
    "led": 12,
}
LONG_GAP_STEPS = 60          # 10 min @ 10s
ZSCORE_THRESHOLD = 3.5

# PF ไม่มีใน fan/pump/solar ของไฟล์นี้ → impute แบบ neutral เพื่อไม่ให้ train เจอ NaN
# pf_available จะบอกว่า PF เป็นค่าที่วัดจริงหรือไม่
PF_IMPUTE_VALUE = 1.0

# Current-based preliminary ON/OFF label. ใช้เป็น label เบื้องต้น ปรับ threshold ได้ภายหลัง
# ถ้า device ไม่อยู่ใน map จะ auto estimate จาก p95 ของ current
STATUS_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # i_on / i_off ตั้ง conservative จาก distribution จริงในไฟล์นี้
    "cooling_fan1":    {"i_on": 0.30, "i_off": 0.10},
    "cooling_fan2":    {"i_on": 0.30, "i_off": 0.10},
    "cooling_fan3":    {"i_on": 0.30, "i_off": 0.10},
    "coolingpad_pump": {"i_on": 0.50, "i_off": 0.15},
    "fert_pump":       {"i_on": 0.50, "i_off": 0.15},
    "solar":           {"i_on": 1.00, "i_off": 0.30},
    "led":             {"i_on": 0.05, "i_off": 0.02},
}

# Rolling feature windows: samples on 10s grid
ROLLING_WINDOWS = {
    "1min": 6,
    "10min": 60,
}

# คอลัมน์สุดท้ายที่อนุญาตให้ไปถึง combined/train-ready CSV
# NOTE: ไม่ใส่ voltage / power เพื่อบังคับให้ไฟล์ train ใช้แกน I/PF เท่านั้น
BASE_OUTPUT_COLUMNS = [
    "timestamp", "device",
    "current", "pf", "pf_available", "status",
    "active_current_proxy", "reactive_current_proxy", "current_abs",
    "delta_current", "delta_pf",
    "hour", "dayofweek", "month",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_daytime", "is_night", "is_weekend",
    "sync_offset_ms", "long_gap_flag", "is_interpolated", "train_set",
]


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def _to_naive_datetime(s: pd.Series) -> pd.Series:
    """Parse timestamp and remove timezone if present."""
    ts = pd.to_datetime(s, errors="coerce", utc=True)
    return ts.dt.tz_convert(None)


def _parse_date_time(date_s: pd.Series, time_s: pd.Series) -> pd.Series:
    """Parse fan Date + Time columns such as 31-Mar-26 / 12:13:55."""
    dt = date_s.astype(str).str.strip() + " " + time_s.astype(str).str.strip()
    return pd.to_datetime(dt, format="%d-%b-%y %H:%M:%S", errors="coerce")


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clip_pf(pf: pd.Series) -> pd.Series:
    # PF บาง meter อาจเป็น signed PF ได้ แต่ไฟล์นี้ส่วนใหญ่ positive; clip เพื่อความเสถียรของ sqrt
    return pd.to_numeric(pf, errors="coerce").clip(-1.0, 1.0)


# ─────────────────────────────────────────
# 1) SHEET PARSERS → standardized raw format
#    output: timestamp, current, pf_raw, pf_available, device
# ─────────────────────────────────────────
def parse_fan_ws(ws, sheet_name: str) -> pd.DataFrame:
    """Parse fan sheets using openpyxl read-only rows for speed."""
    rows = []

    # Data starts at row 5: Date | Time | Uxx RMS | Axx RMS | ...
    for row in ws.iter_rows(min_row=5, values_only=True):
        if row is None or len(row) < 4:
            continue

        date_v = row[0]
        time_v = row[1]
        current_v = row[3]   # Axx RMS

        if date_v is None or time_v is None:
            continue

        rows.append((date_v, time_v, current_v))

    if not rows:
        return pd.DataFrame(columns=["timestamp", "current", "pf_raw", "pf_available", "device"])

    raw = pd.DataFrame(rows, columns=["date", "time", "current"])
    raw["timestamp"] = _parse_date_time(raw["date"], raw["time"])
    raw["current"] = _safe_numeric(raw["current"])

    df = raw[["timestamp", "current"]].dropna(subset=["timestamp"]).copy()
    df["pf_raw"] = np.nan
    df["pf_available"] = 0
    df["device"] = sheet_name
    return df.sort_values("timestamp").reset_index(drop=True)


def parse_i_ws(ws, sheet_name: str) -> pd.DataFrame:
    """Parse pump/solar sheets: last_changed | I | last_changed | V. Use only I timestamp + current."""
    rows = []
    for ts_v, current_v, *_ in ws.iter_rows(min_row=2, values_only=True):
        if ts_v is None:
            continue
        rows.append((ts_v, current_v))
    if not rows:
        return pd.DataFrame(columns=["timestamp", "current", "pf_raw", "pf_available", "device"])
    raw = pd.DataFrame(rows, columns=["timestamp", "current"])
    raw["timestamp"] = _to_naive_datetime(raw["timestamp"])
    raw["current"] = _safe_numeric(raw["current"])
    df = raw[["timestamp", "current"]].dropna(subset=["timestamp"]).copy()
    df["pf_raw"] = np.nan
    df["pf_available"] = 0
    df["device"] = sheet_name
    return df.sort_values("timestamp").reset_index(drop=True)


def parse_led_ws(ws, sheet_name: str = LED_SHEET) -> pd.DataFrame:
    """Parse LED sheet: timestamp/current/pf columns."""
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    col_map = {str(name).strip().lower(): idx for idx, name in enumerate(header) if name is not None}
    ts_idx = col_map.get("timestamp")
    current_idx = col_map.get("current")
    pf_idx = col_map.get("pf")
    if ts_idx is None or current_idx is None:
        raise ValueError(f"{sheet_name}: timestamp/current columns not found")
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        ts_v = row[ts_idx]
        if ts_v is None:
            continue
        current_v = row[current_idx]
        pf_v = row[pf_idx] if pf_idx is not None else np.nan
        rows.append((ts_v, current_v, pf_v))
    raw = pd.DataFrame(rows, columns=["timestamp", "current", "pf_raw"])
    if pd.api.types.is_numeric_dtype(raw["timestamp"]):
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="D", origin="1899-12-30", errors="coerce")
    else:
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw["current"] = _safe_numeric(raw["current"])
    raw["pf_raw"] = _clip_pf(raw["pf_raw"])
    raw["pf_available"] = raw["pf_raw"].notna().astype(int)
    df = raw[["timestamp", "current", "pf_raw", "pf_available"]].dropna(subset=["timestamp"]).copy()
    df["device"] = sheet_name
    return df.sort_values("timestamp").reset_index(drop=True)


def load_all_sheets(file_path: Path) -> Dict[str, pd.DataFrame]:
    wb = load_workbook(file_path, read_only=True, data_only=True)
    parsed: Dict[str, pd.DataFrame] = {}
    try:
        for sheet in FAN_SHEETS:
            parsed[sheet] = parse_fan_ws(wb[sheet], sheet)
        for sheet in IPV_SHEETS:
            parsed[sheet] = parse_i_ws(wb[sheet], sheet)
        parsed[LED_SHEET] = parse_led_ws(wb[LED_SHEET], LED_SHEET)
    finally:
        wb.close()
    return parsed


# ─────────────────────────────────────────
# 2) SYNC / CLEAN
# ─────────────────────────────────────────
def initial_domain_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["timestamp", "device", "current", "pf_raw"]).sort_values("timestamp")

    # current should not be negative for this use-case; tiny negative meter noise → 0
    if "current" in df.columns:
        df.loc[df["current"] < 0, "current"] = 0.0
        df.loc[df["current"] > 5000, "current"] = np.nan

    if "pf_raw" in df.columns:
        df.loc[(df["pf_raw"] < -1.2) | (df["pf_raw"] > 1.2), "pf_raw"] = np.nan
        df["pf_raw"] = _clip_pf(df["pf_raw"])
        df["pf_available"] = df["pf_raw"].notna().astype(int)

    return df


def build_master_grid(device_frames: Dict[str, pd.DataFrame], freq: str = MASTER_FREQ) -> pd.DatetimeIndex:
    min_ts = min(df["timestamp"].min() for df in device_frames.values())
    max_ts = max(df["timestamp"].max() for df in device_frames.values())
    return pd.date_range(min_ts.floor(freq), max_ts.ceil(freq), freq=freq)


def sync_to_master_grid(df: pd.DataFrame, master_grid: pd.DatetimeIndex, device_name: str) -> pd.DataFrame:
    """
    Snap raw timestamps to nearest 10s grid before reindex.
    This avoids losing rows whose timestamps are e.g. xx:xx:55 while the master grid starts at xx:xx:50/00.
    """
    df = df.copy().dropna(subset=["timestamp"])
    freq = pd.Timedelta(MASTER_FREQ)

    # snap/round to nearest grid slot
    df["grid_ts"] = df["timestamp"].dt.round(MASTER_FREQ)
    df["sync_offset_ms"] = ((df["timestamp"] - df["grid_ts"]).dt.total_seconds() * 1000.0).round(1)

    # Aggregate duplicates in each grid slot
    agg = {
        "current": "mean",
        "pf_raw": "mean",
        "pf_available": "max",
        "sync_offset_ms": lambda x: float(np.nanmean(np.abs(x))) if len(x) else np.nan,
    }
    raw_agg = (
        df.groupby("grid_ts", as_index=False)
          .agg(agg)
          .rename(columns={"grid_ts": "timestamp"})
          .sort_values("timestamp")
    )

    synced = raw_agg.set_index("timestamp").reindex(master_grid)
    synced.index.name = "timestamp"
    synced["device"] = device_name
    return synced.reset_index()


def interpolate_short_gaps(df: pd.DataFrame, device_name: str) -> pd.DataFrame:
    df = df.copy().set_index("timestamp")
    limit = INTERP_LIMIT_PER_DEVICE.get(device_name, 6)

    # interpolate current and measured pf_raw only; do not interpolate sync offset
    for col in ["current", "pf_raw"]:
        if col in df.columns:
            df[col] = df[col].interpolate(method="time", limit=limit, limit_direction="forward")

    # pf_available: after interpolation, still indicates whether the device has real PF channel at that row
    if "pf_available" in df.columns:
        df["pf_available"] = df["pf_available"].fillna(0).astype(int)
    return df.reset_index()


def mark_long_gaps(df: pd.DataFrame, long_gap_steps: int = LONG_GAP_STEPS) -> pd.DataFrame:
    df = df.copy()
    is_nan = df["current"].isna()
    run_id = (~is_nan).cumsum()
    run_id[~is_nan] = np.nan
    run_lengths = is_nan.groupby(run_id).transform("sum").where(is_nan, 0)
    df["long_gap_flag"] = (run_lengths >= long_gap_steps).astype(int)
    return df


def zscore_outlier_clean(df: pd.DataFrame, device_name: str, threshold: float = ZSCORE_THRESHOLD) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Rolling z-score for current and pf_raw. Conservative; skips current outlier removal for devices that are mostly constant/high."""
    df = df.copy()
    counts: Dict[str, int] = {}

    # Current: use rolling z-score for high-frequency devices, global for others
    for col in ["current", "pf_raw"]:
        if col not in df.columns:
            continue
        series = df[col]
        valid = series.dropna()
        if len(valid) < 20 or valid.std(ddof=0) == 0:
            counts[col] = 0
            continue

        # 1-hour rolling window on 10s grid
        roll_window = 360
        if len(valid) > roll_window:
            roll_mean = series.rolling(window=roll_window, center=True, min_periods=20).mean()
            roll_std = series.rolling(window=roll_window, center=True, min_periods=20).std(ddof=0).replace(0, np.nan)
            z = (series - roll_mean) / roll_std
        else:
            z = (series - valid.mean()) / valid.std(ddof=0)

        mask = z.abs() > threshold
        counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    return df, counts


def add_interpolation_flags(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    out = after.copy()
    flags = pd.Series(False, index=out.index)
    for col in ["current", "pf_raw"]:
        if col in before.columns and col in after.columns:
            flags = flags | (before[col].isna() & after[col].notna())
    out["is_interpolated"] = flags.astype(int)
    return out


# ─────────────────────────────────────────
# 3) I/PF FEATURE ENGINEERING
# ─────────────────────────────────────────
def add_pf_imputation_and_features(df: pd.DataFrame, device_name: str) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    # PF จริงถ้ามี; ถ้าไม่มีให้เติม neutral value และเก็บ flag ไว้
    df["pf_available"] = df.get("pf_available", pd.Series(0, index=df.index)).fillna(0).astype(int)
    df["pf"] = _clip_pf(df["pf_raw"]).fillna(PF_IMPUTE_VALUE)

    # Keep current numeric and fill remaining short/edge NaN with 0 only after long-gap flag exists
    df["current"] = pd.to_numeric(df["current"], errors="coerce").fillna(0.0)
    df.loc[df["current"] < 0, "current"] = 0.0

    pf_abs = df["pf"].abs().clip(0, 1)
    df["current_abs"] = df["current"].abs()
    df["active_current_proxy"] = df["current"] * df["pf"]
    df["reactive_current_proxy"] = df["current"] * np.sqrt(np.maximum(0.0, 1.0 - np.square(pf_abs)))

    df["delta_current"] = df["current"].diff().fillna(0.0)
    df["delta_pf"] = df["pf"].diff().fillna(0.0)

    for name, win in ROLLING_WINDOWS.items():
        df[f"current_roll_mean_{name}"] = df["current"].rolling(win, min_periods=1).mean()
        df[f"current_roll_std_{name}"] = df["current"].rolling(win, min_periods=2).std(ddof=0).fillna(0.0)
        df[f"pf_roll_mean_{name}"] = df["pf"].rolling(win, min_periods=1).mean()
        df[f"pf_roll_std_{name}"] = df["pf"].rolling(win, min_periods=2).std(ddof=0).fillna(0.0)

    # Temporal features
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = ts.dt.hour.fillna(0).astype(int)
    df["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    df["month"] = ts.dt.month.fillna(1).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)
    df["is_night"] = 1 - df["is_daytime"]
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df


def _auto_status_threshold(current: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(current, errors="coerce").dropna()
    s = s[s >= 0]
    if len(s) == 0:
        return {"i_on": 0.05, "i_off": 0.02}
    p95 = float(np.percentile(s, 95))
    i_on = max(0.02, 0.10 * p95)
    i_off = max(0.01, 0.40 * i_on)
    return {"i_on": i_on, "i_off": i_off}


def add_status_hysteresis(df: pd.DataFrame, device_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    th = STATUS_THRESHOLDS.get(device_name) or _auto_status_threshold(df["current"])
    i_on = float(th["i_on"])
    i_off = float(th["i_off"])

    state = 0
    status = []
    for val in df["current"].fillna(0.0).to_numpy(dtype=float):
        if state == 0 and val >= i_on:
            state = 1
        elif state == 1 and val <= i_off:
            state = 0
        status.append(state)
    df["status"] = np.asarray(status, dtype=np.int8)
    return df, {"i_on": i_on, "i_off": i_off}


# ─────────────────────────────────────────
# 4) NORMALIZATION
# ─────────────────────────────────────────
def add_normalized_columns(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> pd.DataFrame:
    """Fit scaler statistics on chronological train portion only, then apply to all rows."""
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    n_train = max(2, int(len(df) * train_ratio))

    skip_cols = {"status", "pf_available", "is_daytime", "is_night", "is_weekend", "is_interpolated", "long_gap_flag", "train_set"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in skip_cols and not c.endswith(("_z", "_minmax"))
    ]

    train_df = df.iloc[:n_train]
    for col in numeric_cols:
        train_valid = train_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(train_valid) < 2:
            continue
        mu = train_valid.mean()
        sigma = train_valid.std(ddof=0)
        cmin = train_valid.min()
        cmax = train_valid.max()
        s = df[col].replace([np.inf, -np.inf], np.nan)
        if sigma and not np.isnan(sigma):
            df[f"{col}_z"] = (s - mu) / sigma
        if cmax > cmin:
            df[f"{col}_minmax"] = (s - cmin) / (cmax - cmin)

    df["train_set"] = (df.index < n_train).astype(int)
    return df


# ─────────────────────────────────────────
# 5) DIAGNOSTIC PLOTS
# ─────────────────────────────────────────
def save_diagnostic_plots(df: pd.DataFrame, device: str) -> List[str]:
    saved: List[str] = []
    DIR_PLOTS.mkdir(parents=True, exist_ok=True)

    # sample for large plot
    work = df[["timestamp", "current", "pf", "status"]].copy()
    if len(work) == 0:
        return saved
    stride = max(1, math.ceil(len(work) / 5000))
    w = work.iloc[::stride]

    fig, ax1 = plt.subplots(figsize=(16, 5))
    ax1.plot(w["timestamp"], w["current"], linewidth=0.8, label="current")
    ax1.set_ylabel("Current (A)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.step(w["timestamp"], w["status"], where="post", linewidth=0.8, alpha=0.7, label="status")
    ax2.set_ylabel("Status")
    ax2.set_ylim(-0.1, 1.1)
    fig.suptitle(f"Current + preliminary status — {device}")
    fig.tight_layout()
    out = DIR_PLOTS / f"{device}_current_status.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    saved.append(str(out))

    if df["pf_available"].sum() > 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        sample = df[df["pf_available"] == 1][["current", "pf"]].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=RANDOM_STATE)
        ax.scatter(sample["current"], sample["pf"], s=4, alpha=0.4)
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("PF")
        ax.set_title(f"I-PF scatter — {device}")
        ax.grid(True, alpha=0.3)
        out = DIR_PLOTS / f"{device}_ipf_scatter.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(out))

    return saved


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def run_pipeline(file_path: Path = INPUT_XLSX, output_dir: Path = OUTPUT_DIR) -> None:
    for d in [OUTPUT_DIR, DIR_CLEANED, DIR_NORMALIZED, DIR_COMBINED, DIR_PLOTS]:
        d.mkdir(parents=True, exist_ok=True)

    print("Loading sheets …")
    device_frames = load_all_sheets(file_path)
    master_grid = build_master_grid(device_frames, freq=MASTER_FREQ)
    print(f"Master grid: {master_grid[0]} → {master_grid[-1]} ({len(master_grid):,} steps @ {MASTER_FREQ})")

    normalized_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []
    diagnostic_paths: List[str] = []
    thresholds_used: Dict[str, Dict[str, float]] = {}

    for device, raw_df in device_frames.items():
        print(f"\nProcessing: {device} (raw rows: {len(raw_df):,})")
        native_diffs = raw_df["timestamp"].sort_values().drop_duplicates().diff().dt.total_seconds().dropna()
        median_native_dt = float(native_diffs.median()) if len(native_diffs) else np.nan

        step1 = initial_domain_clean(raw_df)
        step2 = sync_to_master_grid(step1, master_grid, device)
        step2["device"] = device

        step3a = interpolate_short_gaps(step2, device)
        step3a = mark_long_gaps(step3a, long_gap_steps=LONG_GAP_STEPS)
        step3b, outlier_counts = zscore_outlier_clean(step3a, device, threshold=ZSCORE_THRESHOLD)
        step3c = interpolate_short_gaps(step3b, device)
        step3d = add_interpolation_flags(step2, step3c)
        step3d["device"] = device
        if "long_gap_flag" in step3a.columns:
            step3d["long_gap_flag"] = step3a["long_gap_flag"].values

        # I/PF feature engineering + preliminary status
        step4 = add_pf_imputation_and_features(step3d, device)
        step4, th_used = add_status_hysteresis(step4, device)
        thresholds_used[device] = th_used

        # cleaned output: still not normalized, but already I/PF train-ready
        cleaned_csv = DIR_CLEANED / f"{device}_cleaned_ipf_10s.csv"
        # keep only I/PF columns and engineered features; do not output voltage/power
        cleaned_cols = [c for c in step4.columns if c in BASE_OUTPUT_COLUMNS or c.startswith(("current_roll_", "pf_roll_"))]
        step4[cleaned_cols].to_csv(cleaned_csv, index=False)
        print(f"  [cleaned]    {cleaned_csv.name} ({len(step4):,} rows)")

        step5 = add_normalized_columns(step4)
        step5["device"] = device
        norm_csv = DIR_NORMALIZED / f"{device}_normalized_ipf_10s.csv"
        # include normalized feature columns too
        allowed_cols = [c for c in step5.columns if c in BASE_OUTPUT_COLUMNS or c.startswith(("current_roll_", "pf_roll_")) or c.endswith(("_z", "_minmax"))]
        step5[allowed_cols].to_csv(norm_csv, index=False)
        print(f"  [normalized] {norm_csv.name} ({len(step5):,} rows)")

        normalized_frames.append(step5[allowed_cols].copy())
        diagnostic_paths.extend(save_diagnostic_plots(step5, device))

        summary_rows.append({
            "device": device,
            "raw_rows": int(len(raw_df)),
            "synced_rows": int(len(step5)),
            "start_time": raw_df["timestamp"].min(),
            "end_time": raw_df["timestamp"].max(),
            "median_native_dt_sec": median_native_dt,
            "pf_rows_measured_after_sync": int(step5["pf_available"].sum()),
            "pf_available_ratio": float(step5["pf_available"].mean()),
            "interpolated_rows": int(step5["is_interpolated"].sum()),
            "long_gap_rows": int(step5["long_gap_flag"].sum()),
            "status_on_ratio": float(step5["status"].mean()),
            "i_on": th_used["i_on"],
            "i_off": th_used["i_off"],
            "train_rows": int(step5["train_set"].sum()),
            "test_rows": int((step5["train_set"] == 0).sum()),
            "outliers_current": outlier_counts.get("current", 0),
            "outliers_pf_raw": outlier_counts.get("pf_raw", 0),
        })

    all_data = pd.concat(normalized_frames, ignore_index=True)
    combined_csv = DIR_COMBINED / "all_devices_normalized_10s_long.csv"
    all_data.to_csv(combined_csv, index=False)
    print(f"\n  [combined] {combined_csv.name} ({len(all_data):,} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "pipeline_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n── Pipeline summary ──")
    show_cols = ["device", "raw_rows", "synced_rows", "median_native_dt_sec", "pf_available_ratio", "status_on_ratio", "interpolated_rows", "long_gap_rows"]
    print(summary_df[show_cols].to_string(index=False))

    metadata = {
        "pipeline": "I/PF preparation pipeline",
        "input_file": str(file_path),
        "output_dir": str(output_dir),
        "master_frequency": MASTER_FREQ,
        "train_ratio": TRAIN_RATIO,
        "devices": ALL_DEVICES,
        "core_input_assumption": "timestamp + current(I) + power factor(PF). Voltage/power are not exported as training features.",
        "pf_policy": {
            "pf_available": "1 if PF is measured in the raw file row after sync; 0 otherwise",
            "pf_impute_value": PF_IMPUTE_VALUE,
            "note": "Fan/pump/solar sheets in this workbook do not provide PF directly, so PF is imputed and pf_available=0 for those rows. LED has measured PF.",
        },
        "feature_engineering": {
            "active_current_proxy": "current * pf",
            "reactive_current_proxy": "current * sqrt(max(0, 1 - abs(pf)^2))",
            "rolling_windows_samples": ROLLING_WINDOWS,
            "temporal_features": ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"],
            "removed": "V-I trajectory disabled/removed because this preparation version uses I/PF only.",
        },
        "status_label": {
            "method": "current hysteresis",
            "thresholds": thresholds_used,
            "note": "Preliminary status labels. Tune thresholds before final research reporting if needed.",
        },
        "cleaning": {
            "interpolation_limit_per_device_steps": INTERP_LIMIT_PER_DEVICE,
            "long_gap_steps": LONG_GAP_STEPS,
            "zscore_threshold": ZSCORE_THRESHOLD,
            "normalization": "fit statistics on chronological train portion only, then apply to train+test",
        },
        "output_files": {
            "combined_train_ready_long_csv": str(combined_csv),
            "summary": str(summary_path),
            "diagnostic_plots": diagnostic_paths,
        },
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print(f"""
Done. Output folders:
  01_cleaned    → {DIR_CLEANED}
  02_normalized → {DIR_NORMALIZED}
  03_combined   → {DIR_COMBINED}
  04_plots      → {DIR_PLOTS}
  summary       → {summary_path}
""")


if __name__ == "__main__":
    run_pipeline()
