import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GRID_PATH   = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"
TRAIN_PATH  = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR  = BASE_DIR / "ml_models"
OUT_PATH    = BASE_DIR / "data" / "processed" / "grid_predictions.csv"

# kolom fitur yang dipakai (sama seperti di train_models.py)
FEATURE_COLS = ["event_count", "mean_mag", "max_mag", "mean_depth", "event_occur"]


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

    # Poisson → prediksi lambda (rata-rata jumlah kejadian)
    lam = poisson_model.predict(X)
    # Probabilitas minimal 0, batasi juga supaya tidak minus kalau ada error numerik
    lam = np.clip(lam, a_min=0, a_max=None)
    poisson_prob = 1 - np.exp(-lam)  # P(X>=1) = 1 - e^(-λ)

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
