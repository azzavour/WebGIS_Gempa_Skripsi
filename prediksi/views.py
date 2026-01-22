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

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "data" / "processed" / "grid_predictions.csv"
GRID_SIZE = 1.0

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

# kolom fitur yang dipakai (sama seperti di train_models.py)
FEATURE_COLS = ["event_count", "mean_mag", "max_mag", "mean_depth", "event_occur"]

RISK_SCALE = [
    (1_000_000, "Ekstrem"),
    (500_000, "Sangat Tinggi"),
    (200_000, "Tinggi"),
    (50_000, "Menengah"),
    (10_000, "Rendah"),
    (0, "Sangat Rendah"),
]

_GRID_LOOKUP_CACHE: dict[str, dict[str, str]] | None = None
_POP_LOOKUP_CACHE: dict[tuple[str, str], float] | None = None
_ADMIN_FEATURES_CACHE: list[dict] | None = None


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
        features.append(
            {
                "geometry": geometry,
                "bbox": _geometry_bbox(geometry),
                "provinsi": prov_label,
                "kab_kota": kab_label,
                "prov_norm": _normalize_province(prov_label),
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
        risk = probability * pop_value
        risk_label = _classify_risk(risk)
        stats["probability"]["min"] = min(stats["probability"]["min"], probability)
        stats["probability"]["max"] = max(stats["probability"]["max"], probability)
        stats["risk"]["min"] = min(stats["risk"]["min"], risk)
        stats["risk"]["max"] = max(stats["risk"]["max"], risk)
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
            "risk": risk,
            "risk_label": risk_label,
            "rf_prob": _clamp_probability(row["rf_prob"]),
            "svm_prob": _clamp_probability(row["svm_prob"]),
            "poisson_prob": _clamp_probability(row["poisson_prob"]),
            "probability_field": MODEL_FIELDS[model_key],
        }
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

    features = []
    for _, row in df.iterrows():
        grid_lat = float(row["grid_lat"])
        grid_lon = float(row["grid_lon"])
        lat = grid_lat + 0.5
        lon = grid_lon + 0.5

        props = {
            "grid_id": str(row["grid_id"]),
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "year": int(row["year"]),
            "target_year": int(row["target_year"]),
            "rf_prob": float(row["rf_prob"]),
            "svm_prob": float(row["svm_prob"]),
            "poisson_prob": float(row["poisson_prob"]),
        }

        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )

    return JsonResponse({"type": "FeatureCollection", "features": features})


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
