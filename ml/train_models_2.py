import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

# =========================
# Konfigurasi path
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(DATA_PROCESSED, "bab4_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_path = os.path.join(DATA_PROCESSED, "training_dataset.csv")

print(">> Membaca training_dataset.csv ...")
df = pd.read_csv(training_path)

print("\n=== INFO training_dataset ===")
print(df.info())
print(df.head())

# =========================
# Siapkan fitur & target
# =========================
# Target: apakah minggu berikutnya ada minimal 1 gempa (M>=4.5)
# event_count_next > 0 -> 1, else 0
if "event_count_next" not in df.columns:
    raise ValueError("Kolom 'event_count_next' tidak ditemukan di training_dataset.csv")

df["target"] = (df["event_count_next"] > 0).astype(int)

print("\nKolom yang tersedia di training_dataset:")
print(df.columns.tolist())

# Jika kedalaman pakai nama lain (misal 'mean_depth'), buat alias 'depth_mean'
if "depth_mean" not in df.columns and "mean_depth" in df.columns:
    df["depth_mean"] = df["mean_depth"]

# Kandidat fitur yang ideal
candidate_features = [
    "event_count",
    "max_mag",
    "depth_mean",
    "grid_lat",
    "grid_lon",
    "year",
    "month",
    "week",
]

# Pakai hanya yang benar-benar ada di dataset
feature_cols = [c for c in candidate_features if c in df.columns]

print("Fitur yang dipakai untuk model:", feature_cols)

if not feature_cols:
    raise ValueError("Tidak ada satu pun fitur yang cocok ditemukan di training_dataset.csv")

X = df[feature_cols].astype(float)
y = df["target"].astype(int)

# =========================
# Split train-test berbasis waktu (year, week)
# =========================
sort_cols = [c for c in ["year", "month", "week"] if c in df.columns]

if sort_cols:
    print("\nMengurutkan data berdasarkan kolom waktu:", sort_cols)
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)
else:
    print("\nTidak ada kolom waktu eksplisit (month/week), pakai urutan indeks saja.")
    df_sorted = df.reset_index(drop=True)

X_sorted = df_sorted[feature_cols]
y_sorted = df_sorted["target"]

n = len(df_sorted)
train_size = int(n * 0.7)  # 70% data pertama untuk train, 30% terakhir untuk test

X_train = X_sorted.iloc[:train_size]
y_train = y_sorted.iloc[:train_size]
X_test = X_sorted.iloc[train_size:]
y_test = y_sorted.iloc[train_size:]

print(f"\n>> Total sampel : {n}")
print(f">> Train size   : {len(X_train)}")
print(f">> Test size    : {len(X_test)}")

# =========================
# Fungsi bantu evaluasi
# =========================
def evaluate_model(name, y_true, proba):
    """Hitung metrik evaluasi utama dan kembalikan dict."""
    proba = np.clip(proba, 1e-6, 1 - 1e-6)  # hindari 0/1 murni
    y_pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, proba)
    ll = log_loss(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    print(f"\n=== {name} ===")
    print(f"Akurasi      : {acc:.4f}")
    print(f"Brier Score  : {brier:.4f}")
    print(f"Log Loss     : {ll:.4f}")
    print(f"ROC AUC      : {auc:.4f}")

    return {
        "model": name,
        "accuracy": acc,
        "brier_score": brier,
        "log_loss": ll,
        "roc_auc": auc,
    }

model_scores = []
roc_data = {}

# =========================
# 1. Random Forest
# =========================
print("\n>> Melatih Random Forest ...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

proba_rf = rf.predict_proba(X_test)[:, 1]
scores_rf = evaluate_model("Random Forest", y_test, proba_rf)
model_scores.append(scores_rf)

fpr_rf, tpr_rf, _ = roc_curve(y_test, proba_rf)
roc_data["Random Forest"] = (fpr_rf, tpr_rf)

# Simpan feature importance
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

fi.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"), index=False)

plt.figure()
plt.bar(fi["feature"], fi["importance"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
fi_png = os.path.join(OUTPUT_DIR, "rf_feature_importance.png")
plt.savefig(fi_png, dpi=300)
plt.close()

print(f">> Feature importance RF disimpan ke: {fi_png}")

# =========================
# 2. Support Vector Machine (SVM RBF)
# =========================
print("\n>> Melatih SVM (RBF) ...")
svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)

proba_svm = svm.predict_proba(X_test)[:, 1]
scores_svm = evaluate_model("SVM (RBF)", y_test, proba_svm)
model_scores.append(scores_svm)

fpr_svm, tpr_svm, _ = roc_curve(y_test, proba_svm)
roc_data["SVM (RBF)"] = (fpr_svm, tpr_svm)

# =========================
# 3. Poisson Regressor (baseline)
# =========================
print("\n>> Melatih Poisson Regressor (baseline) ...")

# Poisson memodelkan laju kejadian λ; probabilitas >=1 event: p = 1 - exp(-λ)
poisson = PoissonRegressor(alpha=0.0, max_iter=1000)
poisson.fit(X_train, y_train)

lambda_pred = poisson.predict(X_test)
lambda_pred = np.clip(lambda_pred, 1e-6, 50)  # jaga numerik

proba_pois = 1 - np.exp(-lambda_pred)
scores_pois = evaluate_model("Poisson Regressor", y_test, proba_pois)
model_scores.append(scores_pois)

fpr_pois, tpr_pois, _ = roc_curve(y_test, proba_pois)
roc_data["Poisson Regressor"] = (fpr_pois, tpr_pois)

# =========================
# Simpan skor model ke CSV
# =========================
scores_df = pd.DataFrame(model_scores)
scores_csv_path = os.path.join(OUTPUT_DIR, "model_scores_bab4.csv")
scores_df.to_csv(scores_csv_path, index=False)
print(f"\n>> Tabel skor model disimpan ke: {scores_csv_path}")

# =========================
# Plot ROC curve 3 model
# =========================
plt.figure()
for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Kurva ROC – Perbandingan Model")
plt.legend()
plt.tight_layout()
roc_png = os.path.join(OUTPUT_DIR, "roc_curves.png")
plt.savefig(roc_png, dpi=300)
plt.close()

print(f">> Kurva ROC disimpan ke: {roc_png}")
print("\nSelesai STEP 2: training & evaluasi model untuk BAB 4.")
