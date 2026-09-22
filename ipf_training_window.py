"""Shared date-window filtering for IPF training and evaluation scripts."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd


ENV_START = "IPF_TRAIN_START"
ENV_END = "IPF_TRAIN_END"

DEFAULT_START = "2026-04-01 00:00:00"
DEFAULT_END = "2026-04-05 23:59:59"


def _parse_window_value(value: str) -> Optional[pd.Timestamp]:
    value = (value or "").strip()
    if not value or value.lower() in {"all", "none", "null"}:
        return None
    return pd.Timestamp(value)


def get_training_window() -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    start = _parse_window_value(os.getenv(ENV_START, DEFAULT_START))
    end = _parse_window_value(os.getenv(ENV_END, DEFAULT_END))
    if start is not None and end is not None and start > end:
        raise ValueError(f"{ENV_START} must be before or equal to {ENV_END}")
    return start, end


def format_training_window() -> str:
    start, end = get_training_window()
    start_s = "all" if start is None else str(start)
    end_s = "all" if end is None else str(end)
    return f"{start_s} to {end_s}"


def training_window_metadata() -> dict:
    start, end = get_training_window()
    return {
        "date_window_start": None if start is None else str(start),
        "date_window_end": None if end is None else str(end),
        "date_window_env_start": ENV_START,
        "date_window_env_end": ENV_END,
    }


def filter_training_window(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    label: str = "training/evaluation",
) -> pd.DataFrame:
    start, end = get_training_window()
    if start is None and end is None:
        print(f"[load] Date window ({label}): all rows")
        return df.reset_index(drop=True)

    before = len(df)
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")
    mask = ts.notna()
    if start is not None:
        mask &= ts >= start
    if end is not None:
        mask &= ts <= end

    out = df.loc[mask].copy().reset_index(drop=True)
    print(
        f"[load] Date window ({label}): {format_training_window()} | "
        f"kept {len(out):,}/{before:,} rows"
    )
    if out.empty:
        raise ValueError(f"No rows remain after applying date window: {format_training_window()}")
    return out
