"""
Utilities for aggregating weekly earthquake predictions into monthly risk metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


class MonthlyAggregationError(ValueError):
    """Raised when the weekly dataset cannot be aggregated into months."""


DATE_COLUMN_CANDIDATES: tuple[str, ...] = (
    "prediction_date",
    "prediction_timestamp",
    "timestamp",
    "week_start",
    "week_date",
    "date",
)

@dataclass(frozen=True)
class MonthlyAggregationResult:
    month: str
    probability: float
    risk_classification: str


def classify_monthly_risk(probability: float | None) -> str:
    """Map a probability into a discrete Low/Medium/High bucket."""
    if probability is None or not np.isfinite(probability):
        return "Unknown"
    if probability > 0.6:
        return "High"
    if probability >= 0.3:
        return "Medium"
    if probability >= 0.0:
        return "Low"
    return "Unknown"


def _combine_weekly_probabilities(values: Sequence[float]) -> float:
    """Compute 1 - Π (1 - p_week) while avoiding floating point overflow."""
    if not len(values):
        return 0.0
    arr = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    complement = 1.0 - arr
    combined = 1.0 - float(np.prod(complement))
    # numerical drift can push slightly outside [0, 1]
    return float(np.clip(combined, 0.0, 1.0))


def _aggregate_probability_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    probability = _combine_weekly_probabilities(numeric.to_list())
    return pd.Series({"probability": probability, "weekly_count": len(numeric)})


def _year_week_to_timestamp(years: pd.Series, weeks: pd.Series) -> pd.Series:
    timestamps = []
    for year, week in zip(years, weeks):
        try:
            year_int = int(year)
            week_int = int(week)
            timestamps.append(pd.Timestamp.fromisocalendar(year_int, week_int, 1))
        except (TypeError, ValueError):
            timestamps.append(pd.NaT)
    return pd.Series(timestamps, index=years.index)


def _extract_month_series(df: pd.DataFrame) -> pd.Series:
    for column in DATE_COLUMN_CANDIDATES:
        if column in df.columns:
            dt_series = pd.to_datetime(df[column], errors="coerce", utc=True)
            if dt_series.notna().any():
                return dt_series.dt.tz_localize(None)

    if {"year", "month"}.issubset(df.columns):
        dt_series = pd.to_datetime(
            {
                "year": pd.to_numeric(df["year"], errors="coerce"),
                "month": pd.to_numeric(df["month"], errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )
        return dt_series

    if {"year", "week"}.issubset(df.columns):
        return _year_week_to_timestamp(df["year"], df["week"])

    raise MonthlyAggregationError(
        "Weekly predictions must contain either a timestamp/date column "
        "or (year, week)/(year, month) columns for monthly aggregation."
    )


def aggregate_weekly_predictions(
    df: pd.DataFrame,
    probability_column: str,
    group_by: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate weekly predictions into monthly probabilities per group.

    Returns a DataFrame containing at least:
        - month_label (str, YYYY-MM)
        - probability (float)
        - risk_classification (str)
        - weekly_count (int)

    Additional grouping columns (e.g., grid_id) are preserved.
    """
    if probability_column not in df.columns:
        raise MonthlyAggregationError(f"Kolom probabilitas '{probability_column}' tidak ditemukan.")

    working = df.copy()
    month_series = _extract_month_series(working)
    month_periods = month_series.dt.to_period("M")
    working["agg_month_ts"] = month_periods.dt.to_timestamp()
    working["agg_month_label"] = month_periods.astype(str)
    working = working[month_periods.notna()].copy()

    base_group_cols = list(dict.fromkeys((group_by or ["grid_id"])))
    if not base_group_cols:
        raise MonthlyAggregationError("Setidaknya satu kolom pengelompokan diperlukan (mis. grid_id).")
    if any(col not in working.columns for col in base_group_cols):
        missing = [col for col in base_group_cols if col not in working.columns]
        raise MonthlyAggregationError(f"Kolom pengelompokan hilang: {', '.join(missing)}")

    if working.empty:
        empty_cols = base_group_cols + ["month_label", "probability", "risk_classification", "weekly_count"]
        return pd.DataFrame(columns=empty_cols)

    group_cols = base_group_cols + ["agg_month_label"]
    prob_stats = (
        working.groupby(group_cols)[probability_column]
        .apply(_aggregate_probability_series)
        .unstack()
        .reset_index()
    )

    meta_cols = [col for col in ("grid_lat", "grid_lon", "centroid_lat", "centroid_lon") if col in working.columns]
    named_aggs: dict[str, tuple[str, str]] = {col: (col, "first") for col in meta_cols}
    named_aggs["month_start"] = ("agg_month_ts", "first")

    meta = working.groupby(group_cols).agg(**named_aggs).reset_index()

    result = prob_stats.merge(meta, on=group_cols, how="left")
    result = result.rename(columns={"agg_month_label": "month_label"})

    if "month_start" in result.columns:
        month_ts = pd.to_datetime(result["month_start"], errors="coerce")
        result["year"] = month_ts.dt.year.astype("Int64")
        result["month_number"] = month_ts.dt.month.astype("Int64")
    else:
        result["year"] = pd.NA
        result["month_number"] = pd.NA

    result["risk_classification"] = result["probability"].apply(classify_monthly_risk)
    sort_cols = base_group_cols + ["month_label"]
    return result.sort_values(sort_cols).reset_index(drop=True)
