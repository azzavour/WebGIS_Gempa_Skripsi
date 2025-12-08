import joblib
import numpy as np
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRED_PATH = BASE_DIR / "data" / "processed" / "grid_predictions.csv"
GRID_SIZE = 1.0

GRID_PATH = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"
TRAIN_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = BASE_DIR / "ml_models"
OUT_PATH = PRED_PATH
MODEL_SCORE_PATH = BASE_DIR / "data" / "processed" / "model_scores.csv"

MODEL_FIELDS = {
    "rf": "rf_prob",
    "svm": "svm_prob",
    "poisson": "poisson_prob",
}

# kolom fitur yang dipakai (sama seperti di train_models.py)
FEATURE_COLS = ["event_count", "mean_mag", "max_mag", "mean_depth", "event_occur"]


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
        properties = {
            "grid_id": str(row["grid_id"]),
            "year": _as_int(row["year"]),
            "target_year": _as_int(row["target_year"]),
            "rf_prob": float(row["rf_prob"]),
            "svm_prob": float(row["svm_prob"]),
            "poisson_prob": float(row["poisson_prob"]),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": properties,
            }
        )

    return JsonResponse(
        {
            "type": "FeatureCollection",
            "features": features,
            "meta": {"model": model_key, "property": MODEL_FIELDS[model_key]},
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
