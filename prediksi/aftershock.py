"""
Rule-based aftershock probability estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
GRID_DATA_PATH = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"
DECISION_SUPPORT_LABEL = "Decision Support – Non Official Early Warning"


@lru_cache(maxsize=1)
def _load_latest_magnitudes() -> dict[str, float]:
    if not GRID_DATA_PATH.exists():
        return {}
    df = pd.read_csv(GRID_DATA_PATH)
    if "year" not in df.columns or "grid_id" not in df.columns:
        return {}
    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year].copy()
    magnitude_cols = [col for col in ("max_mag", "mean_mag") if col in latest.columns]
    if not magnitude_cols:
        return {}
    lookup: dict[str, float] = {}
    for _, row in latest.iterrows():
        grid_id = str(row["grid_id"])
        magnitude = None
        for col in magnitude_cols:
            value = row.get(col)
            if pd.notna(value):
                magnitude = float(value)
                break
        if magnitude is not None:
            lookup[grid_id] = magnitude
    return lookup


def classify_aftershock_risk(magnitude: Optional[float]) -> str:
    if magnitude is None or not math.isfinite(magnitude):
        return "Low"
    if magnitude >= 6.0:
        return "High"
    if magnitude >= 5.5:
        return "Medium"
    return "Low"


@dataclass(frozen=True)
class AftershockEstimate:
    aftershock_risk: str
    mainshock_magnitude: Optional[float]
    decision_support_label: str
    explanation: str


class AftershockEstimator:
    """Derives aftershock risk for each grid-id using rule-based thresholds."""

    def __init__(self) -> None:
        self._lookup = _load_latest_magnitudes()

    def describe(self, grid_id: str | None, magnitude: Optional[float] = None) -> AftershockEstimate:
        magnitude_value = None
        if magnitude is not None and math.isfinite(magnitude):
            magnitude_value = float(magnitude)
        elif grid_id:
            magnitude_value = self._lookup.get(str(grid_id))

        risk = classify_aftershock_risk(magnitude_value)
        if magnitude_value is None:
            explanation = (
                "Data magnitudo tidak tersedia; risiko aftershock diasumsikan rendah untuk dukungan keputusan."
            )
        else:
            explanation = (
                f"Magnitudo utama {magnitude_value:.1f}; aturan {risk} "
                "diterapkan untuk Decision Support – Non Official Early Warning."
            )
        return AftershockEstimate(
            aftershock_risk=risk,
            mainshock_magnitude=magnitude_value,
            decision_support_label=DECISION_SUPPORT_LABEL,
            explanation=explanation,
        )

    def describe_properties(self, grid_id: str | None, magnitude: Optional[float] = None) -> dict:
        estimate = self.describe(grid_id, magnitude)
        magnitude_value = estimate.mainshock_magnitude
        return {
            "aftershock_risk": estimate.aftershock_risk,
            "aftershock_decision_label": estimate.decision_support_label,
            "aftershock_explanation": estimate.explanation,
            "mainshock_magnitude": None if magnitude_value is None else round(magnitude_value, 2),
        }


AFTERSHOCK_ESTIMATOR = AftershockEstimator()
