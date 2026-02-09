import json
import math
import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from .aftershock import AFTERSHOCK_ESTIMATOR
from .monthly_aggregation import MonthlyAggregationError, aggregate_weekly_predictions
from .seismic_cause import CAUSE_ANALYZER

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "data" / "processed" / "grid_predictions.csv"
WEEKLY_PRED_PATH = BASE_DIR / "data" / "processed" / "grid_weekly_predictions.csv"
GRID_SIZE = 1.0
BMKG_CLEAN_PATH = BASE_DIR / "data" / "processed" / "bmkg_clean.csv"

WEST_JAVA_LAT_MIN = -7.8
WEST_JAVA_LAT_MAX = -5.9
WEST_JAVA_LON_MIN = 106.3
WEST_JAVA_LON_MAX = 108.9
WEST_JAVA_BBOX = (WEST_JAVA_LON_MIN, WEST_JAVA_LAT_MIN, WEST_JAVA_LON_MAX, WEST_JAVA_LAT_MAX)
WEST_JAVA_PROVINCE_NORM = "jawabarat"

GRID_PATH = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = BASE_DIR / "ml_models"
OUT_PATH = PRED_PATH
MODEL_SCORE_PATH = BASE_DIR / "data" / "processed" / "model_scores.csv"
ADMIN_GEOJSON_PATH = BASE_DIR / "data" / "raw" / "geo" / "indonesia_kabkota.geojson"
BPS_MASTER_PATH = BASE_DIR / "data" / "processed" / "bps_kecamatan_master_clean.csv"

MODEL_FIELDS = {
    "rf": "rf_prob",
    "svm": "svm_prob",
    "poisson": "poisson_prob",
}

MONTH_NAMES = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
]

# kolom fitur yang dipakai (sama seperti di train_models.py)
FEATURE_COLS = ["event_count", "mean_mag", "max_mag", "mean_depth", "event_occur"]

RISK_SCALE = [
    (1_000_000, "Extreme"),
    (300_000, "High"),
    (100_000, "Medium"),
    (0, "Low"),
]

DECISION_SUPPORT_NOTE = "Decision Support System, not official early warning"

MITIGATION_GUIDANCE = {
    "Extreme": [
        "Prepare emergency kit",
        "Evacuation planning",
        "Structural safety checks",
        "Public alert dissemination",
    ],
    "High": [
        "Community preparedness",
        "Emergency logistics readiness",
    ],
    "Medium": [
        "Awareness campaigns",
    ],
    "Low": [
        "Monitoring and information update",
    ],
}

_GRID_LOOKUP_CACHE: dict[str, dict[str, str]] | None = None
_POP_LOOKUP_CACHE: dict[tuple[str, str], float] | None = None
_ADMIN_FEATURES_CACHE: list[dict] | None = None
_MONTHLY_HIST_CACHE: list[float] | None = None
_BMKG_WEST_CACHE: pd.DataFrame | None = None


def _get_mitigation_actions(level: str) -> list[str]:
    return MITIGATION_GUIDANCE.get(level, MITIGATION_GUIDANCE["Low"])


def _get_cause_properties(lat: float | None, lon: float | None) -> dict:
    try:
        lat_val = float(lat) if lat is not None else None
        lon_val = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lat_val = None
        lon_val = None
    return CAUSE_ANALYZER.describe_properties(lat_val, lon_val)


def _get_aftershock_properties(grid_id: str | None, magnitude: float | None = None) -> dict:
    try:
        mag_value = float(magnitude) if magnitude is not None else None
    except (TypeError, ValueError):
        mag_value = None
    return AFTERSHOCK_ESTIMATOR.describe_properties(grid_id, mag_value)


def _as_int(value):
    return int(value) if pd.notna(value) else None


def _load_model_scores():
    if not MODEL_SCORE_PATH.exists():
        return {"columns": [], "rows": []}

    df = pd.read_csv(MODEL_SCORE_PATH)
    columns = df.columns.tolist()
    rows = []
    for _, row in df.iterrows():
        display_row = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                display_row.append("-")
            elif isinstance(value, str):
                display_row.append(value)
            elif isinstance(value, (int, np.integer)):
                display_row.append(f"{int(value)}")
            elif isinstance(value, (float, np.floating)):
                display_row.append(f"{float(value):.3f}")
            else:
                display_row.append(str(value))
        rows.append(display_row)

    return {"columns": columns, "rows": rows}


def home(request):
    score_data = _load_model_scores()
    return render(request, "prediksi/home.html", {"score_data": score_data})


def _normalize_text(value: object, drop_words: tuple[str, ...]) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    replacements = {
        "kabupatan": "kabupaten",
        "kab.": "kabupaten",
        "kota adm": "kota administrasi",
        "kotamadya": "kota",
        "kotamadaya": "kota",
        "propinsi": "provinsi",
        "&": " dan ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    for word in drop_words:
        text = text.replace(word, " ")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def _normalize_province(name: object) -> str:
    drop_words = (
        "provinsi",
        "prov",
        "province",
        "propinsi",
        "daerahkhususibukota",
        "daerahistimewa",
    )
    normalized = _normalize_text(name, drop_words)
    special = {
        "jakarta": "jakarta",
        "dki": "jakarta",
        "dki jakarta": "jakarta",
        "diajogjakarta": "yogyakarta",
        "diyogyakarta": "yogyakarta",
    }
    return special.get(normalized, normalized)


def _normalize_kab_kota(name: object) -> str:
    drop_words = (
        "kabupaten",
        "kabupat",
        "kab",
        "kota",
        "kotamadya",
        "kotamady",
        "administrasi",
        "adm",
        "regency",
        "city",
        "kabupatenadministrasi",
        "kotamadmadya",
    )
    return _normalize_text(name, drop_words)


def _format_province_label(label: object) -> str:
    if label is None:
        return ""
    value = str(label).strip()
    if not value:
        return ""
    title = value.title()
    title = title.replace("Dki ", "DKI ").replace("Di ", "DI ")
    return title


def _format_kab_kota_label(label: object) -> str:
    if label is None:
        return ""
    value = str(label).strip()
    if not value:
        return ""
    title = value.title()
    title = title.replace("Kab ", "Kab ")
    title = title.replace("Dki ", "DKI ")
    title = title.replace("Di ", "DI ")
    title = title.replace("Adm.", "Administrasi")
    title = title.replace(" Adm ", " Administrasi ")
    return title


def _bbox_overlaps(
    bbox: tuple[float, float, float, float] | None, region: tuple[float, float, float, float] = WEST_JAVA_BBOX
) -> bool:
    if bbox is None:
        return False
    min_lon, min_lat, max_lon, max_lat = bbox
    region_min_lon, region_min_lat, region_max_lon, region_max_lat = region
    return not (
        max_lon < region_min_lon
        or min_lon > region_max_lon
        or max_lat < region_min_lat
        or min_lat > region_max_lat
    )


def _parse_bbox_param(value: str) -> tuple[float, float, float, float]:
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except (TypeError, ValueError) as exc:
        raise ValueError("Parameter bbox harus berupa 'min_lon,min_lat,max_lon,max_lat'.") from exc
    if len(parts) != 4:
        raise ValueError("Parameter bbox harus memuat empat angka: min_lon,min_lat,max_lon,max_lat.")
    min_lon, min_lat, max_lon, max_lat = parts
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Nilai bbox tidak valid: pastikan min < max.")
    return (min_lon, min_lat, max_lon, max_lat)


def _bbox_within_region(
    bbox: tuple[float, float, float, float], region: tuple[float, float, float, float] = WEST_JAVA_BBOX
) -> bool:
    min_lon, min_lat, max_lon, max_lat = bbox
    region_min_lon, region_min_lat, region_max_lon, region_max_lat = region
    return (
        region_min_lon <= min_lon < max_lon <= region_max_lon
        and region_min_lat <= min_lat < max_lat <= region_max_lat
    )


def _get_requested_bbox(request) -> tuple[float, float, float, float]:
    bbox_param = request.GET.get("bbox")
    if not bbox_param:
        return WEST_JAVA_BBOX
    bbox = _parse_bbox_param(bbox_param)
    if not _bbox_within_region(bbox, WEST_JAVA_BBOX):
        raise ValueError(
            "Permintaan berada di luar batas Jawa Barat "
            "(lat -7.8 s/d -5.9 dan lon 106.3 s/d 108.9)."
        )
    return bbox


def _filter_predictions_by_bbox(
    df: pd.DataFrame, bbox: tuple[float, float, float, float]
) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    lat_series = pd.to_numeric(df["grid_lat"], errors="coerce") + GRID_SIZE / 2.0
    lon_series = pd.to_numeric(df["grid_lon"], errors="coerce") + GRID_SIZE / 2.0
    mask = (
        lat_series.notna()
        & lon_series.notna()
        & (lat_series >= min_lat)
        & (lat_series <= max_lat)
        & (lon_series >= min_lon)
        & (lon_series <= max_lon)
    )
    return df.loc[mask].copy()


def _geometry_bbox(geometry: dict) -> tuple[float, float, float, float] | None:
    coords = geometry.get("coordinates")
    if not coords:
        return None
    xs = []
    ys = []
    if geometry["type"] == "Polygon":
        for ring in coords:
            for lon, lat in ring:
                xs.append(lon)
                ys.append(lat)
    elif geometry["type"] == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for lon, lat in ring:
                    xs.append(lon)
                    ys.append(lat)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_contains(bbox: tuple[float, float, float, float] | None, lon: float, lat: float) -> bool:
    if bbox is None:
        return True
    x_min, y_min, x_max, y_max = bbox
    return x_min <= lon <= x_max and y_min <= lat <= y_max


def _point_in_ring(lon: float, lat: float, ring: list) -> bool:
    intersects = False
    ring_len = len(ring)
    if ring_len < 3:
        return False
    x = lon
    y = lat
    for i in range(ring_len):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % ring_len]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-9) + x1):
            intersects = not intersects
    return intersects


def _point_in_polygon(lon: float, lat: float, polygon: list) -> bool:
    if not polygon:
        return False
    if not _point_in_ring(lon, lat, polygon[0]):
        return False
    for hole in polygon[1:]:
        if _point_in_ring(lon, lat, hole):
            return False
    return True


def _point_in_geometry(lon: float, lat: float, geometry: dict) -> bool:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return False
    if gtype == "Polygon":
        return _point_in_polygon(lon, lat, coords)
    if gtype == "MultiPolygon":
        for polygon in coords:
            if _point_in_polygon(lon, lat, polygon):
                return True
    return False


def _load_admin_features() -> list[dict]:
    global _ADMIN_FEATURES_CACHE
    if _ADMIN_FEATURES_CACHE is not None:
        return _ADMIN_FEATURES_CACHE
    if not ADMIN_GEOJSON_PATH.exists():
        _ADMIN_FEATURES_CACHE = []
        return _ADMIN_FEATURES_CACHE
    data = json.loads(ADMIN_GEOJSON_PATH.read_text(encoding="utf-8"))
    features = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry")
        props = feature.get("properties", {})
        if not geometry:
            continue
        prov_raw = props.get("prov_name") or props.get("PROVINSI") or props.get("province")
        kab_raw = props.get("alt_name") or props.get("ALT_NAME") or props.get("name")
        prov_label = _format_province_label(prov_raw)
        kab_label = _format_kab_kota_label(kab_raw)
        prov_norm = _normalize_province(prov_label)
        bbox = _geometry_bbox(geometry)
        if prov_norm != WEST_JAVA_PROVINCE_NORM:
            continue
        if not _bbox_overlaps(bbox, WEST_JAVA_BBOX):
            continue
        features.append(
            {
                "geometry": geometry,
                "bbox": bbox,
                "provinsi": prov_label,
                "kab_kota": kab_label,
                "prov_norm": prov_norm,
                "kab_norm": _normalize_kab_kota(kab_label),
            }
        )
    _ADMIN_FEATURES_CACHE = features
    return _ADMIN_FEATURES_CACHE


def _get_population_lookup() -> dict[tuple[str, str], float]:
    global _POP_LOOKUP_CACHE
    if _POP_LOOKUP_CACHE is not None:
        return _POP_LOOKUP_CACHE
    if not BPS_MASTER_PATH.exists():
        _POP_LOOKUP_CACHE = {}
        return _POP_LOOKUP_CACHE
    df = pd.read_csv(BPS_MASTER_PATH)
    if "jumlah_penduduk" not in df.columns:
        _POP_LOOKUP_CACHE = {}
        return _POP_LOOKUP_CACHE
    df["jumlah_penduduk"] = pd.to_numeric(df["jumlah_penduduk"], errors="coerce")
    df["prov_norm"] = df["provinsi"].apply(_normalize_province)
    df = df[df["prov_norm"] == WEST_JAVA_PROVINCE_NORM]
    df["kab_norm"] = df["kabupaten"].apply(_normalize_kab_kota)
    grouped = (
        df.dropna(subset=["prov_norm", "kab_norm"])
        .groupby(["prov_norm", "kab_norm"])["jumlah_penduduk"]
        .sum(min_count=1)
        .reset_index()
    )
    _POP_LOOKUP_CACHE = {
        (row["prov_norm"], row["kab_norm"]): float(row["jumlah_penduduk"] or 0.0)
        for _, row in grouped.iterrows()
    }
    return _POP_LOOKUP_CACHE


def _load_grid_lookup() -> dict[str, dict[str, str]]:
    global _GRID_LOOKUP_CACHE
    if _GRID_LOOKUP_CACHE is not None:
        return _GRID_LOOKUP_CACHE
    if not PRED_PATH.exists():
        _GRID_LOOKUP_CACHE = {}
        return _GRID_LOOKUP_CACHE
    admin_features = _load_admin_features()
    if not admin_features:
        _GRID_LOOKUP_CACHE = {}
        return _GRID_LOOKUP_CACHE
    df = pd.read_csv(PRED_PATH)
    required_cols = {"grid_id", "grid_lat", "grid_lon"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        _GRID_LOOKUP_CACHE = {}
        return _GRID_LOOKUP_CACHE
    df = _filter_predictions_by_bbox(df, WEST_JAVA_BBOX)
    if df.empty:
        _GRID_LOOKUP_CACHE = {}
        return _GRID_LOOKUP_CACHE
    unique_grids = df[["grid_id", "grid_lat", "grid_lon"]].drop_duplicates()
    lookup: dict[str, dict[str, str]] = {}
    for _, row in unique_grids.iterrows():
        grid_id = str(row["grid_id"])
        center_lon = float(row["grid_lon"]) + GRID_SIZE / 2.0
        center_lat = float(row["grid_lat"]) + GRID_SIZE / 2.0
        match = None
        for feature in admin_features:
            if not _bbox_contains(feature.get("bbox"), center_lon, center_lat):
                continue
            if _point_in_geometry(center_lon, center_lat, feature["geometry"]):
                match = feature
                break
        if match:
            lookup[grid_id] = {
                "provinsi": match["provinsi"],
                "kab_kota": match["kab_kota"],
                "prov_norm": match["prov_norm"],
                "kab_norm": match["kab_norm"],
            }
        else:
            lookup[grid_id] = {
                "provinsi": "Perairan Terbuka",
                "kab_kota": "Grid di Laut / Perbatasan",
                "prov_norm": "",
                "kab_norm": "",
            }
    _GRID_LOOKUP_CACHE = lookup
    return _GRID_LOOKUP_CACHE


def _clamp_probability(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _classify_risk(value: float) -> str:
    for threshold, label in RISK_SCALE:
        if value >= threshold:
            return label
    return "Tidak Diketahui"


def _monthly_probability_from_yearly(prob_year: float) -> float:
    try:
        py = float(prob_year)
    except (TypeError, ValueError):
        return 0.0
    py = max(0.0, min(1.0, py))
    return 1 - (1 - py) ** (1 / 12)


def _classify_probability(prob: float) -> str:
    pct = prob * 100
    if pct < 30:
        return "Rendah"
    if pct <= 60:
        return "Sedang"
    return "Tinggi"


def _load_bmkg_west_java() -> pd.DataFrame:
    global _BMKG_WEST_CACHE
    if _BMKG_WEST_CACHE is not None:
        return _BMKG_WEST_CACHE
    if not BMKG_CLEAN_PATH.exists():
        _BMKG_WEST_CACHE = pd.DataFrame(columns=["lat", "lon", "date", "month"])
        return _BMKG_WEST_CACHE
    try:
        df = pd.read_csv(BMKG_CLEAN_PATH)
    except Exception:
        _BMKG_WEST_CACHE = pd.DataFrame(columns=["lat", "lon", "date", "month"])
        return _BMKG_WEST_CACHE
    # ensure date parsing and month extraction
    date_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.NaT
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
    # Filter West Java bounds (extended per requirement)
    mask = (
        pd.to_numeric(df["lat"], errors="coerce").between(-8.5, -5.5)
        & pd.to_numeric(df["lon"], errors="coerce").between(106.0, 109.0)
    )
    df = df.loc[mask, ["lat", "lon", "month"]].dropna(subset=["month"])
    _BMKG_WEST_CACHE = df
    return _BMKG_WEST_CACHE


def _get_monthly_hist_norm() -> list[float]:
    global _MONTHLY_HIST_CACHE
    if _MONTHLY_HIST_CACHE is not None:
        return _MONTHLY_HIST_CACHE
    df = _load_bmkg_west_java()
    if df.empty:
        _MONTHLY_HIST_CACHE = [1 / 12] * 12
        return _MONTHLY_HIST_CACHE
    counts = df["month"].value_counts().reindex(range(1, 13), fill_value=0).tolist()
    total = sum(counts)
    _MONTHLY_HIST_CACHE = [c / total for c in counts] if total else [1 / 12] * 12
    return _MONTHLY_HIST_CACHE


@require_GET
def prediksi_geojson(request):
    model_key = request.GET.get("model", "rf").lower()
    if model_key not in MODEL_FIELDS:
        model_key = "rf"

    if not PRED_PATH.exists():
        return JsonResponse({"error": "grid_predictions.csv tidak ditemukan."}, status=404)

    df = pd.read_csv(PRED_PATH)
    required_cols = [
        "grid_id",
        "grid_lat",
        "grid_lon",
        "year",
        "target_year",
        "rf_prob",
        "svm_prob",
        "poisson_prob",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return JsonResponse(
            {"error": f"Kolom hilang di grid_predictions.csv: {', '.join(missing)}"},
            status=500,
        )

    try:
        requested_bbox = _get_requested_bbox(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    df = _filter_predictions_by_bbox(df, requested_bbox)
    if df.empty:
        return JsonResponse(
            {"error": "Tidak ada data prediksi di dalam batas koordinat yang diminta."},
            status=404,
        )

    grid_lookup = _load_grid_lookup()
    pop_lookup = _get_population_lookup()
    stats_template = {
        "probability": {"min": 1.0, "max": 0.0},
        "risk": {"min": math.inf, "max": 0.0},
        "pop_exposed": {"min": math.inf, "max": 0.0},
    }
    stats = {key: value.copy() for key, value in stats_template.items()}
    features = []
    for _, row in df.iterrows():
        lat = float(row["grid_lat"])
        lon = float(row["grid_lon"])
        polygon = [
            [lon, lat],
            [lon + GRID_SIZE, lat],
            [lon + GRID_SIZE, lat + GRID_SIZE],
            [lon, lat + GRID_SIZE],
            [lon, lat],
        ]
        centroid_lon = lon + GRID_SIZE / 2.0
        centroid_lat = lat + GRID_SIZE / 2.0
        grid_id = str(row["grid_id"])
        location = grid_lookup.get(grid_id, {})
        provinsi = location.get("provinsi") or "Tidak diketahui"
        kab_kota = location.get("kab_kota") or "Tidak diketahui"
        prov_norm = location.get("prov_norm") or ""
        kab_norm = location.get("kab_norm") or ""
        pop_value = 0.0
        if prov_norm and kab_norm:
            pop_value = pop_lookup.get((prov_norm, kab_norm), 0.0)
        probability = _clamp_probability(row[MODEL_FIELDS[model_key]])
        exposure = probability * pop_value
        risk_label = _classify_risk(exposure)
        mitigation = _get_mitigation_actions(risk_label)
        stats["probability"]["min"] = min(stats["probability"]["min"], probability)
        stats["probability"]["max"] = max(stats["probability"]["max"], probability)
        stats["risk"]["min"] = min(stats["risk"]["min"], exposure)
        stats["risk"]["max"] = max(stats["risk"]["max"], exposure)
        stats["pop_exposed"]["min"] = min(stats["pop_exposed"]["min"], pop_value)
        stats["pop_exposed"]["max"] = max(stats["pop_exposed"]["max"], pop_value)
        properties = {
            "grid_id": grid_id,
            "grid_lat": lat,
            "grid_lon": lon,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "year": _as_int(row["year"]),
            "target_year": _as_int(row["target_year"]),
            "provinsi": provinsi,
            "kab_kota": kab_kota,
            "probability": probability,
            "pop_exposed": int(round(pop_value)),
            "exposure": exposure,
            "risk": exposure,
            "risk_label": risk_label,
            "rf_prob": _clamp_probability(row["rf_prob"]),
            "svm_prob": _clamp_probability(row["svm_prob"]),
            "poisson_prob": _clamp_probability(row["poisson_prob"]),
            "probability_field": MODEL_FIELDS[model_key],
            "mitigation_recommendations": mitigation,
            "decision_support_note": DECISION_SUPPORT_NOTE,
        }
        properties.update(_get_cause_properties(centroid_lat, centroid_lon))
        properties.update(_get_aftershock_properties(grid_id))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": properties,
            }
        )

    for key in stats:
        if stats[key]["min"] is math.inf or stats[key]["min"] > stats[key]["max"]:
            stats[key]["min"] = 0.0
        if stats[key]["max"] is math.inf:
            stats[key]["max"] = 0.0

    return JsonResponse(
        {
            "type": "FeatureCollection",
            "features": features,
            "meta": {
                "model": model_key,
                "property": MODEL_FIELDS[model_key],
                "stats": stats,
                "color_modes": ["risk", "probability", "pop_exposed"],
                "bounds": {
                    "min_lon": requested_bbox[0],
                    "min_lat": requested_bbox[1],
                    "max_lon": requested_bbox[2],
                    "max_lat": requested_bbox[3],
                },
            },
        }
    )


def prediksi_points(request):
    if not PRED_PATH.exists():
        return JsonResponse({"error": "grid_predictions.csv tidak ditemukan."}, status=404)

    df = pd.read_csv(PRED_PATH)
    required_cols = [
        "grid_id",
        "grid_lat",
        "grid_lon",
        "year",
        "target_year",
        "rf_prob",
        "svm_prob",
        "poisson_prob",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return JsonResponse(
            {"error": f"Kolom hilang di grid_predictions.csv: {', '.join(missing)}"},
            status=500,
        )

    try:
        requested_bbox = _get_requested_bbox(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    df = _filter_predictions_by_bbox(df, requested_bbox)
    if df.empty:
        return JsonResponse(
            {"error": "Tidak ada data prediksi di dalam batas koordinat yang diminta."},
            status=404,
        )

    grid_lookup = _load_grid_lookup()
    pop_lookup = _get_population_lookup()

    features = []
    for _, row in df.iterrows():
        grid_lat = float(row["grid_lat"])
        grid_lon = float(row["grid_lon"])
        lat = grid_lat + 0.5
        lon = grid_lon + 0.5
        grid_id = str(row["grid_id"])
        location = grid_lookup.get(grid_id, {})
        prov_norm = location.get("prov_norm") or ""
        kab_norm = location.get("kab_norm") or ""
        pop_value = 0.0
        if prov_norm and kab_norm:
            pop_value = pop_lookup.get((prov_norm, kab_norm), 0.0)
        probability = _clamp_probability(row["rf_prob"])
        exposure = probability * pop_value
        risk_label = _classify_risk(exposure)
        mitigation = _get_mitigation_actions(risk_label)

        props = {
            "grid_id": grid_id,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "year": int(row["year"]),
            "target_year": int(row["target_year"]),
            "rf_prob": float(row["rf_prob"]),
            "svm_prob": float(row["svm_prob"]),
            "poisson_prob": float(row["poisson_prob"]),
            "probability": probability,
            "pop_exposed": int(round(pop_value)),
            "exposure": exposure,
            "risk": exposure,
            "risk_label": risk_label,
            "mitigation_recommendations": mitigation,
            "decision_support_note": DECISION_SUPPORT_NOTE,
        }
        if location:
            props["provinsi"] = location.get("provinsi")
            props["kab_kota"] = location.get("kab_kota")
        props.update(_get_cause_properties(lat, lon))
        props.update(_get_aftershock_properties(props["grid_id"]))

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )

    return JsonResponse(
        {
            "type": "FeatureCollection",
            "features": features,
            "bounds": {
                "min_lon": requested_bbox[0],
                "min_lat": requested_bbox[1],
                "max_lon": requested_bbox[2],
                "max_lat": requested_bbox[3],
            },
        }
    )


@require_GET
def predict_monthly(request):
    model_key = request.GET.get("model", "rf").lower()
    if model_key not in MODEL_FIELDS:
        model_key = "rf"

    probability_field = MODEL_FIELDS[model_key]
    source_path = WEEKLY_PRED_PATH if WEEKLY_PRED_PATH.exists() else PRED_PATH
    if not source_path.exists():
        return JsonResponse({"error": "Dataset prediksi mingguan tidak ditemukan."}, status=404)

    df = pd.read_csv(source_path)
    required_cols = {"grid_id", "grid_lat", "grid_lon", probability_field}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return JsonResponse(
            {"error": f"Kolom hilang di {source_path.name}: {', '.join(missing)}"},
            status=500,
        )

    try:
        requested_bbox = _get_requested_bbox(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    df = _filter_predictions_by_bbox(df, requested_bbox)
    if df.empty:
        return JsonResponse(
            {"error": "Tidak ada data prediksi di dalam batas koordinat yang diminta."},
            status=404,
        )

    try:
        monthly_df = aggregate_weekly_predictions(df, probability_field, group_by=["grid_id"])
    except MonthlyAggregationError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    if monthly_df.empty:
        return JsonResponse(
            {"error": "Data mingguan tidak memiliki informasi bulan yang valid."},
            status=404,
        )

    grid_lookup = _load_grid_lookup()
    if grid_lookup:
        monthly_df["provinsi"] = monthly_df["grid_id"].map(lambda gid: grid_lookup.get(str(gid), {}).get("provinsi"))
        monthly_df["kab_kota"] = monthly_df["grid_id"].map(lambda gid: grid_lookup.get(str(gid), {}).get("kab_kota"))
        monthly_df["prov_norm"] = monthly_df["grid_id"].map(lambda gid: grid_lookup.get(str(gid), {}).get("prov_norm"))
        monthly_df["kab_norm"] = monthly_df["grid_id"].map(lambda gid: grid_lookup.get(str(gid), {}).get("kab_norm"))
    else:
        monthly_df["prov_norm"] = pd.NA
        monthly_df["kab_norm"] = pd.NA

    pop_lookup = _get_population_lookup()

    if "month_start" in monthly_df.columns:
        monthly_df["month_start"] = pd.to_datetime(monthly_df["month_start"], errors="coerce")

    results: list[dict] = []
    for _, row in monthly_df.iterrows():
        centroid_lat = row.get("centroid_lat")
        centroid_lon = row.get("centroid_lon")
        if pd.isna(centroid_lat):
            base_lat = row.get("grid_lat")
            centroid_lat = float(base_lat) + GRID_SIZE / 2.0 if pd.notna(base_lat) else None
        if pd.isna(centroid_lon):
            base_lon = row.get("grid_lon")
            centroid_lon = float(base_lon) + GRID_SIZE / 2.0 if pd.notna(base_lon) else None
        cause_props = _get_cause_properties(centroid_lat, centroid_lon)
        prov_norm_val = row.get("prov_norm")
        kab_norm_val = row.get("kab_norm")
        pop_value = 0.0
        if isinstance(prov_norm_val, str) and prov_norm_val and isinstance(kab_norm_val, str) and kab_norm_val:
            pop_value = pop_lookup.get((prov_norm_val, kab_norm_val), 0.0)
        probability_value = float(row.get("probability") or 0.0)
        exposure = probability_value * pop_value
        risk_label = _classify_risk(exposure)
        mitigation = _get_mitigation_actions(risk_label)
        entry = {
            "grid_id": str(row.get("grid_id")),
            "month": row.get("month_label") or "",
            "probability": probability_value,
            "risk_classification": risk_label,
            "weekly_count": int(row.get("weekly_count") or 0),
        }
        year_value = row.get("year")
        if pd.notna(year_value):
            entry["year"] = int(year_value)
        month_number = row.get("month_number")
        if pd.notna(month_number):
            entry["month_number"] = int(month_number)
        grid_lat = row.get("grid_lat")
        grid_lon = row.get("grid_lon")
        if pd.notna(grid_lat):
            entry["grid_lat"] = float(grid_lat)
        if pd.notna(grid_lon):
            entry["grid_lon"] = float(grid_lon)
        if centroid_lat is not None:
            entry["centroid_lat"] = float(centroid_lat)
        if centroid_lon is not None:
            entry["centroid_lon"] = float(centroid_lon)
        if pd.notna(row.get("provinsi")):
            entry["provinsi"] = row["provinsi"]
        if pd.notna(row.get("kab_kota")):
            entry["kab_kota"] = row["kab_kota"]
        month_start = row.get("month_start")
        if pd.notna(month_start):
            try:
                entry["month_start"] = pd.to_datetime(month_start).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                pass
        entry["pop_exposed"] = int(round(pop_value))
        entry["exposure"] = exposure
        entry["risk"] = exposure
        entry["mitigation_recommendations"] = mitigation
        entry["decision_support_note"] = DECISION_SUPPORT_NOTE
        entry["probability_field"] = probability_field
        entry.update(cause_props)
        entry.update(_get_aftershock_properties(entry["grid_id"]))
        results.append(entry)

    bounds_payload = {
        "min_lon": requested_bbox[0],
        "min_lat": requested_bbox[1],
        "max_lon": requested_bbox[2],
        "max_lat": requested_bbox[3],
    }

    return JsonResponse(
        {
            "model": model_key,
            "probability_field": probability_field,
            "source": source_path.name,
            "results": results,
            "meta": {
                "grid_count": int(monthly_df["grid_id"].nunique()),
                "month_count": int(monthly_df["month_label"].nunique()),
                "bbox": bounds_payload,
            },
        }
    )


@require_GET
def prediksi_bulanan(request):
    """
    Disagregasi probabilitas tahunan menjadi bulanan (tanpa retraining).
    Rumus: P_bulan = 1 - (1 - P_tahun) ** (1/12)
    """
    model_key = request.GET.get("model", "rf").lower()
    if model_key not in MODEL_FIELDS:
        model_key = "rf"

    probability_field = MODEL_FIELDS[model_key]
    if not PRED_PATH.exists():
        return JsonResponse({"error": "grid_predictions.csv tidak ditemukan."}, status=404)

    try:
        requested_bbox = _get_requested_bbox(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    df = pd.read_csv(PRED_PATH)
    required_cols = {"grid_id", "grid_lat", "grid_lon", probability_field, "year", "target_year"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return JsonResponse({"error": f"Kolom hilang: {', '.join(missing)}"}, status=500)

    df = _filter_predictions_by_bbox(df, requested_bbox)
    if df.empty:
        return JsonResponse({"error": "Tidak ada data dalam batas koordinat diminta."}, status=404)

    grid_lookup = _load_grid_lookup()
    pop_lookup = _get_population_lookup()

    monthly_records: list[dict] = []
    exposure_by_month = [0.0 for _ in range(12)]
    prob_sum_by_month = [0.0 for _ in range(12)]
    prob_count_by_month = [0 for _ in range(12)]
    peak_month_global = None
    peak_prob_global = -1.0
    hist_norm = _get_monthly_hist_norm()

    target_year_val = int(df["target_year"].dropna().max()) if "target_year" in df else None

    for _, row in df.iterrows():
        grid_id = str(row.get("grid_id"))
        prob_year = _clamp_probability(row.get(probability_field))
        base_lat = row.get("grid_lat")
        base_lon = row.get("grid_lon")
        centroid_lat = float(base_lat) + GRID_SIZE / 2.0 if pd.notna(base_lat) else None
        centroid_lon = float(base_lon) + GRID_SIZE / 2.0 if pd.notna(base_lon) else None

        # per-grid historical pattern
        bmkg_df = _load_bmkg_west_java()
        cell_mask = (
            pd.to_numeric(bmkg_df["lat"], errors="coerce").between(float(base_lat), float(base_lat) + GRID_SIZE)
            & pd.to_numeric(bmkg_df["lon"], errors="coerce").between(float(base_lon), float(base_lon) + GRID_SIZE)
        )
        cell_df = bmkg_df.loc[cell_mask]
        if cell_df.empty or cell_df["month"].nunique() <= 1:
            monthly_base = hist_norm  # fallback province pattern
        else:
            counts = cell_df["month"].value_counts().reindex(range(1, 13), fill_value=0).tolist()
            var = float(np.var(counts))
            if var <= 0:
                monthly_base = hist_norm
            else:
                total_c = sum(counts)
                monthly_base = [c / total_c for c in counts] if total_c else hist_norm

        monthly_raw = [n * prob_year for n in monthly_base]
        total_raw = sum(monthly_raw)
        scale = 1.0
        if total_raw > prob_year and total_raw > 0:
            scale = prob_year / total_raw
        monthly_probs = [max(0.0, min(1.0, val * scale)) for val in monthly_raw]

        prov_norm = kab_norm = prov_label = kab_label = None
        if grid_lookup:
            meta = grid_lookup.get(str(grid_id), {})
            prov_label = meta.get("provinsi")
            kab_label = meta.get("kab_kota")
            prov_norm = meta.get("prov_norm")
            kab_norm = meta.get("kab_norm")

        pop_val = 0.0
        if prov_norm and kab_norm:
            pop_val = pop_lookup.get((prov_norm, kab_norm), 0.0)

        cause_props = _get_cause_properties(centroid_lat, centroid_lon)

        for month_idx, month_name in enumerate(MONTH_NAMES, start=1):
            prob_month_weighted = monthly_probs[month_idx - 1]
            exposure = prob_month_weighted * pop_val
            exposure_by_month[month_idx - 1] += exposure
            prob_sum_by_month[month_idx - 1] += prob_month_weighted
            prob_count_by_month[month_idx - 1] += 1
            risk_class = _classify_probability(prob_month_weighted)
            entry = {
                "grid_id": grid_id,
                "month_number": month_idx,
                "month_name": month_name,
                "probability_month": prob_month_weighted,
                "prob": prob_month_weighted,
                "probability_year": prob_year,
                "risk_class": risk_class,
                "pop_exposed": int(round(pop_val)),
                "exposure": exposure,
                "year": int(row.get("year")) if pd.notna(row.get("year")) else None,
                "target_year": int(row.get("target_year")) if pd.notna(row.get("target_year")) else None,
            }
            if prov_label:
                entry["provinsi"] = prov_label
            if kab_label:
                entry["kab_kota"] = kab_label
            if base_lat is not None:
                entry["grid_lat"] = float(base_lat)
            if base_lon is not None:
                entry["grid_lon"] = float(base_lon)
            if centroid_lat is not None:
                entry["centroid_lat"] = float(centroid_lat)
            if centroid_lon is not None:
                entry["centroid_lon"] = float(centroid_lon)
            entry.update(cause_props)
            entry.update(_get_aftershock_properties(grid_id))
            monthly_records.append(entry)

            if prob_month_weighted > peak_prob_global:
                peak_prob_global = prob_month_weighted
                peak_month_global = month_name

    exposure_monthly = [
        {"month_name": MONTH_NAMES[i], "exposure": exposure_by_month[i]} for i in range(12)
    ]

    monthly_probs = []
    for i in range(12):
        avg_prob = 0.0
        if prob_count_by_month[i]:
            avg_prob = prob_sum_by_month[i] / prob_count_by_month[i]
        monthly_probs.append({"bulan": MONTH_NAMES[i], "prob": avg_prob})

    response_payload = {
        "region": "Jawa Barat",
        "year": target_year_val or 2026,
        "monthly": monthly_probs,
        "peak_month": peak_month_global,
        "exposure_monthly": exposure_monthly,
        "results": monthly_records,
    }

    return JsonResponse(response_payload)

def main():
    print("Baca data grid tahunan dari:", GRID_PATH)
    grid = pd.read_csv(GRID_PATH)

    # kita ambil tahun TERAKHIR yang punya data lengkap sebagai fitur
    last_year = int(grid["year"].max())
    print("Tahun fitur terakhir:", last_year)

    df = grid[grid["year"] == last_year].copy()

    # pastikan kolom fitur ada
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print("Kolom fitur yang hilang:", missing)
        return

    X = df[FEATURE_COLS].astype(float)

    # load model
    print("Load model dari:", MODELS_DIR)
    rf_model = joblib.load(MODELS_DIR / "rf_model.pkl")
    svm_model = joblib.load(MODELS_DIR / "svm_model.pkl")
    poisson_model = joblib.load(MODELS_DIR / "poisson_model.pkl")

    # prediksi probabilitas gempa (tahun berikutnya)
    rf_prob = rf_model.predict_proba(X)[:, 1]
    svm_prob = svm_model.predict_proba(X)[:, 1]

    # Poisson model prediksi lambda (rata-rata jumlah kejadian)
    lam = poisson_model.predict(X)
    # Probabilitas minimal 0, batasi juga supaya tidak minus kalau ada error numerik
    lam = np.clip(lam, a_min=0, a_max=None)
    poisson_prob = 1 - np.exp(-lam)  # P(X>=1) = 1 - exp(-lambda)

    # buat dataframe output
    df_out = df[["grid_id", "grid_lat", "grid_lon", "year"]].copy()
    df_out["target_year"] = df_out["year"] + 1  # tahun yang diprediksi
    df_out["rf_prob"] = rf_prob
    df_out["svm_prob"] = svm_prob
    df_out["poisson_prob"] = poisson_prob

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print("Prediksi per grid disimpan di:", OUT_PATH)
    print("Jumlah grid:", len(df_out))


if __name__ == "__main__":
    main()
