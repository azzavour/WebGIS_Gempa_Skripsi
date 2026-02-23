"""
Plot kurva ROC-AUC untuk Bab 4 (Random Forest vs SVM) – khusus Jawa Barat.

Langkah:
- Memuat dataset training_dataset.csv
- Memisah fitur dan target (event_count_next > 0)
- Memakai model terlatih rf_model.pkl & svm_model.pkl
- Menghitung ROC dan AUC, lalu menyimpan plot DPI 300
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = BASE_DIR / "ml_models"
OUT_DIR = BASE_DIR / "data" / "processed" / "bab4_outputs"
OUT_PATH = OUT_DIR / "roc_curves_final.png"

# Fitur kandidat mengikuti pipeline training
CANDIDATE_FEATURES = [
    "event_count",
    "mean_mag",
    "mean_depth",
    "max_mag",
    "depth_mean",
    "grid_lat",
    "grid_lon",
    "year",
    "event_occur",
    "month",
    "week",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH)

    # Sinkronisasi nama kolom kedalaman
    if "depth_mean" not in df.columns and "mean_depth" in df.columns:
        df["depth_mean"] = df["mean_depth"]
    if "mean_depth" not in df.columns and "depth_mean" in df.columns:
        df["mean_depth"] = df["depth_mean"]

    # Target biner: ada minimal 1 kejadian berikutnya
    if "event_count_next" not in df.columns:
        raise ValueError("Kolom 'event_count_next' tidak ditemukan di training_dataset.csv")
    df["target"] = (df["event_count_next"] > 0).astype(int)

    feature_cols = [c for c in CANDIDATE_FEATURES if c in df.columns]
    if not feature_cols:
        raise ValueError("Tidak ada fitur yang cocok ditemukan di training_dataset.csv")

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)
    return X, y, feature_cols


def time_based_split(X: pd.DataFrame, y: pd.Series):
    sort_cols = [c for c in ["year", "month", "week"] if c in X.columns]
    if sort_cols:
        df_sorted = pd.concat([X, y], axis=1).sort_values(sort_cols).reset_index(drop=True)
    else:
        df_sorted = pd.concat([X, y], axis=1).reset_index(drop=True)
    n = len(df_sorted)
    train_size = int(n * 0.7)
    X_train = df_sorted.iloc[:train_size, :-1]
    y_train = df_sorted.iloc[:train_size, -1]
    X_test = df_sorted.iloc[train_size:, :-1]
    y_test = df_sorted.iloc[train_size:, -1]
    return X_train, X_test, y_train, y_test


def load_models():
    import joblib

    rf = joblib.load(MODELS_DIR / "rf_model.pkl")
    svm = joblib.load(MODELS_DIR / "svm_model.pkl")
    return rf, svm


def compute_roc(model, X_test, y_test):
    # Sesuaikan urutan/kolom fitur dengan model saat training
    if hasattr(model, "feature_names_in_"):
        needed = list(model.feature_names_in_)
        missing = [c for c in needed if c not in X_test.columns]
        if missing:
            raise ValueError(f"Kolom fitur hilang untuk model {type(model).__name__}: {missing}")
        X_aligned = X_test[needed]
    else:
        X_aligned = X_test

    proba = model.predict_proba(X_aligned)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_val = auc(fpr, tpr)
    return fpr, tpr, auc_val


def plot_roc(rf_curve, svm_curve):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    fpr_rf, tpr_rf, auc_rf = rf_curve
    fpr_svm, tpr_svm, auc_svm = svm_curve

    plt.plot(fpr_rf, tpr_rf, color="#0ea5e9", linewidth=2.2, label=f"Random Forest (AUC = {auc_rf:.2f})")
    plt.plot(fpr_svm, tpr_svm, color="#ef4444", linewidth=2.2, label=f"SVM (AUC = {auc_svm:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Tebakan Acak (AUC = 0.50)")

    plt.title("Perbandingan Kurva ROC: Random Forest vs SVM", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR) / Sensitivitas", fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    return OUT_PATH


def main():
    print("Memuat dataset:", DATA_PATH)
    X, y, features = load_dataset()
    print("Fitur dipakai:", features)

    X_train, X_test, y_train, y_test = time_based_split(X, y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    rf_model, svm_model = load_models()

    rf_curve = compute_roc(rf_model, X_test, y_test)
    svm_curve = compute_roc(svm_model, X_test, y_test)

    out_path = plot_roc(rf_curve, svm_curve)
    print("Plot ROC disimpan di:", out_path)


if __name__ == "__main__":
    main()
