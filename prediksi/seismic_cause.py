"""
Spatial rule-based analysis that explains the likely seismic cause for each grid.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_PATH = BASE_DIR / "data" / "processed" / "seismic_reference_features.json"

_DEFAULT_REFERENCE = {
    "subduction_zones": [
        {
            "name": "Sunda Megathrust",
            "coordinates": [
                [-10.0, 103.5],
                [-9.8, 104.5],
                [-9.6, 105.5],
                [-9.4, 106.5],
                [-9.2, 107.5],
                [-9.1, 108.5],
                [-9.0, 109.5],
                [-8.9, 110.5],
            ],
        }
    ],
    "active_faults": [
        {
            "name": "Cimandiri Fault",
            "coordinates": [
                [-7.1, 106.5],
                [-6.98, 106.8],
                [-6.85, 107.1],
                [-6.7, 107.4],
            ],
        },
        {
            "name": "Lembang Fault",
            "coordinates": [
                [-6.9, 107.4],
                [-6.85, 107.6],
                [-6.8, 107.8],
            ],
        },
        {
            "name": "Baribis Fault",
            "coordinates": [
                [-6.6, 107.0],
                [-6.5, 107.5],
                [-6.4, 108.0],
                [-6.3, 108.5],
            ],
        },
    ],
    "volcanoes": [
        {"name": "Tangkuban Perahu", "coordinates": [-6.77, 107.62]},
        {"name": "Gede Pangrango", "coordinates": [-6.78, 106.98]},
        {"name": "Papandayan", "coordinates": [-7.32, 107.72]},
        {"name": "Ciremai", "coordinates": [-6.89, 108.4]},
        {"name": "Galunggung", "coordinates": [-7.25, 108.05]},
    ],
}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon pairs in kilometers."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return float(r * c)


def _iter_coords(obj) -> Iterable[tuple[float, float]]:
    if obj is None:
        return
    if isinstance(obj, (int, float)):
        return
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if len(obj) >= 2 and all(isinstance(val, (int, float)) for val in obj[:2]):
            try:
                yield (float(obj[0]), float(obj[1]))
            except (TypeError, ValueError):
                return
        else:
            for item in obj:
                yield from _iter_coords(item)


def _prepare_features(raw_entries: Iterable[dict]) -> list[dict]:
    prepared: list[dict] = []
    for entry in raw_entries or []:
        points = [coord for coord in _iter_coords(entry.get("coordinates"))]
        if not points:
            continue
        prepared.append(
            {
                "name": entry.get("name") or "Unknown",
                "points": points,
            }
        )
    return prepared


@lru_cache(maxsize=1)
def _load_reference_data() -> dict:
    if REFERENCE_PATH.exists():
        try:
            with REFERENCE_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (json.JSONDecodeError, OSError):
            loaded = _DEFAULT_REFERENCE
    else:
        loaded = _DEFAULT_REFERENCE
    return {
        "subduction": _prepare_features(loaded.get("subduction_zones")),
        "faults": _prepare_features(loaded.get("active_faults")),
        "volcanoes": _prepare_features(loaded.get("volcanoes")),
    }


class SeismicCauseAnalyzer:
    """Computes the dominant seismic cause following the provided rule set."""

    def __init__(self) -> None:
        data = _load_reference_data()
        self.subduction = data["subduction"]
        self.faults = data["faults"]
        self.volcanoes = data["volcanoes"]

    def _nearest(self, lat: float, lon: float, catalog: list[dict]) -> tuple[float, str | None]:
        if not catalog:
            return math.inf, None
        best_dist = math.inf
        best_name = None
        for entry in catalog:
            for target_lat, target_lon in entry["points"]:
                dist = _haversine_km(lat, lon, target_lat, target_lon)
                if dist < best_dist:
                    best_dist = dist
                    best_name = entry["name"]
        return best_dist, best_name

    def describe(self, lat: float | None, lon: float | None) -> dict:
        if lat is None or lon is None:
            return {
                "cause": "Regional Tectonic Activity",
                "explanation": "Koordinat tidak valid, gunakan penyebab regional sebagai default.",
                "distances": {},
                "nearest": {},
            }
        sub_dist, sub_name = self._nearest(lat, lon, self.subduction)
        fault_dist, fault_name = self._nearest(lat, lon, self.faults)
        volcano_dist, volcano_name = self._nearest(lat, lon, self.volcanoes)

        if sub_dist < 150:
            cause = "Subduction of Indo-Australian Plate"
            detail_name = sub_name or "zona subduksi terdekat"
            explanation = (
                f"Grid berada {sub_dist:.1f} km dari {detail_name}, lebih dekat dari ambang 150 km."
            )
        elif fault_dist < 50:
            cause = "Active Fault Movement"
            detail_name = fault_name or "sesar aktif terdekat"
            explanation = (
                f"Grid berada {fault_dist:.1f} km dari {detail_name}, lebih dekat dari ambang 50 km."
            )
        elif volcano_dist < 30:
            cause = "Volcanic Activity"
            detail_name = volcano_name or "gunung api aktif terdekat"
            explanation = (
                f"Grid berada {volcano_dist:.1f} km dari {detail_name}, lebih dekat dari ambang 30 km."
            )
        else:
            cause = "Regional Tectonic Activity"
            explanation = "Tidak ada struktur utama dalam jarak ambang, penyebab diasumsikan tektonik regional."

        return {
            "cause": cause,
            "explanation": explanation,
            "distances": {
                "subduction_km": sub_dist,
                "fault_km": fault_dist,
                "volcano_km": volcano_dist,
            },
            "nearest": {
                "subduction": sub_name,
                "fault": fault_name,
                "volcano": volcano_name,
            },
        }

    def describe_properties(self, lat: float | None, lon: float | None) -> dict:
        info = self.describe(lat, lon)

        def _clean(value: float) -> float | None:
            return None if value is None or not math.isfinite(value) else round(float(value), 2)

        distances = info.get("distances", {})
        nearest = info.get("nearest", {})

        return {
            "seismic_cause": info.get("cause"),
            "cause_explanation": info.get("explanation"),
            "distance_to_subduction_km": _clean(distances.get("subduction_km")),
            "distance_to_fault_km": _clean(distances.get("fault_km")),
            "distance_to_volcano_km": _clean(distances.get("volcano_km")),
            "nearest_subduction_name": nearest.get("subduction"),
            "nearest_fault_name": nearest.get("fault"),
            "nearest_volcano_name": nearest.get("volcano"),
        }


CAUSE_ANALYZER = SeismicCauseAnalyzer()
