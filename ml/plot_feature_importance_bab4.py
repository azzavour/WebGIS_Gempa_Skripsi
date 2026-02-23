"""
Plot Feature Importance Random Forest untuk Bab 4 (Jawa Barat).

- Memuat training_dataset.csv untuk nama kolom fitur
- Memuat model rf_model.pkl
- Menyelaraskan urutan fitur dengan model.feature_names_in_
- Menyimpan bar chart horizontal DPI 300 ke data/processed/bab4_outputs/rf_feature_importance_final.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = BASE_DIR / "ml_models"
OUT_DIR = BASE_DIR / "data" / "processed" / "bab4_outputs"
OUT_PATH = OUT_DIR / "rf_feature_importance_final.png"


FEATURE_LABEL_MAP = {
    "event_count": "Jumlah Kejadian (Tahun Berjalan)",
    "event_occur": "Ada Kejadian Tahun Berjalan (0/1)",
    "mean_mag": "Magnitudo Rata-rata",
    "max_mag": "Magnitudo Maksimum",
    "mean_depth": "Kedalaman Rata-rata",
    "depth_mean": "Kedalaman Rata-rata",
    "grid_lat": "Latitude Grid",
    "grid_lon": "Longitude Grid",
    "year": "Tahun",
    "month": "Bulan",
    "week": "Minggu",
    "freq_4w": "Frekuensi Gempa 4 Minggu",
    "max_mag_prev": "Magnitudo Maksimum Sebelumnya",
    "mean_depth_prev": "Kedalaman Rata-rata Sebelumnya",
}


def prettify(name: str) -> str:
    return FEATURE_LABEL_MAP.get(name, name.replace("_", " ").title())


def load_features():
    df = pd.read_csv(DATA_PATH)
    # alias kedalaman jika perlu
    if "depth_mean" not in df.columns and "mean_depth" in df.columns:
        df["depth_mean"] = df["mean_depth"]
    if "mean_depth" not in df.columns and "depth_mean" in df.columns:
        df["mean_depth"] = df["depth_mean"]
    feature_cols = [c for c in df.columns if c not in {"event_count_next", "event_next", "target"}]
    return df[feature_cols]


def load_model():
    import joblib

    return joblib.load(MODELS_DIR / "rf_model.pkl")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X = load_features()
    model = load_model()

    # Selaraskan urutan fitur
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Kolom fitur hilang di dataset: {missing}")
        X_use = X[cols]
        feature_names = cols
    else:
        feature_names = list(X.columns)
        X_use = X

    importances = model.feature_importances_
    data = pd.DataFrame({"feature": feature_names, "importance": importances})
    data["label_id"] = data["feature"].apply(prettify)
    data = data.sort_values("importance", ascending=True)  # ascending for horizontal bar

    plt.figure(figsize=(8, max(4, len(data) * 0.45)))
    bar_color = "#38bdf8"  # biru muda tegas
    plt.barh(data["label_id"], data["importance"], color=bar_color, edgecolor="#0f172a", alpha=0.9)
    plt.xlabel("Skor Kepentingan", fontsize=11)
    plt.ylabel("Variabel Prediktor", fontsize=11)
    plt.title("Tingkat Kepentingan Fitur (Feature Importance) - Model Random Forest", fontsize=13, fontweight="bold")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot feature importance disimpan di: {OUT_PATH}")


if __name__ == "__main__":
    main()
