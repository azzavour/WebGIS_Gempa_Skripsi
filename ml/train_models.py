import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    brier_score_loss,
    mean_squared_error,
)
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "training_dataset.csv"
MODELS_DIR = BASE_DIR / "ml_models"
RESULTS_PATH = BASE_DIR / "data" / "processed" / "model_scores.csv"


def load_data():
    print("Baca data training dari:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # fitur yang dipakai
    feature_cols = ["event_count", "mean_mag", "max_mag", "mean_depth", "event_occur"]

    # buang baris yang ada NaN di fitur atau target
    df = df.dropna(subset=feature_cols + ["event_count_next"])

    # pastikan tipe numerik
    for col in feature_cols + ["event_count_next"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols + ["event_count_next"])

    # X = fitur
    X = df[feature_cols].astype(float)

    # y_bin = target biner (ada gempa tahun depan atau tidak)
    df["event_next_binary"] = (df["event_count_next"] > 0).astype(int)
    y_bin = df["event_next_binary"]

    # y_count = target jumlah kejadian (buat model Poisson)
    y_count = df["event_count_next"].astype(float)

    print("Unique label y_bin:", sorted(y_bin.unique()))
    return X, y_bin, y_count


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_svm(X_train, y_train):
    # SVM perlu scaling → pakai pipeline
    svm_clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42)),
        ]
    )
    svm_clf.fit(X_train, y_train)
    return svm_clf


def train_poisson(X_train, y_train_count):
    pr = PoissonRegressor(alpha=1.0, max_iter=1000)
    pr.fit(X_train, y_train_count)
    return pr


def evaluate_classifier(model, X_test, y_test, name):
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    brier = brier_score_loss(y_test, prob)

    print(f"\n=== {name} ===")
    print("Accuracy :", acc)
    print("F1-score :", f1)
    print("Brier    :", brier)

    return {
        "model": name,
        "accuracy": acc,
        "f1_score": f1,
        "brier_score": brier,
    }


def evaluate_poisson(model, X_test, y_true):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_true, y_pred)

    print("\n=== Poisson Regressor ===")
    print("MSE (count) :", mse)

    return {
        "model": "Poisson",
        "mse_count": mse,
    }


def main():
    X, y_bin, y_count = load_data()

    # split train/test 80/20
    X_train, X_test, y_train_bin, y_test_bin, y_train_count, y_test_count = train_test_split(
        X, y_bin, y_count, test_size=0.2, random_state=42, stratify=y_bin
    )

    # 1. Random Forest
    rf_model = train_random_forest(X_train, y_train_bin)

    # 2. SVM
    svm_model = train_svm(X_train, y_train_bin)

    # 3. Poisson
    poisson_model = train_poisson(X_train, y_train_count)

    # evaluasi
    scores = []
    scores.append(evaluate_classifier(rf_model, X_test, y_test_bin, "RandomForest"))
    scores.append(evaluate_classifier(svm_model, X_test, y_test_bin, "SVM"))
    scores.append(evaluate_poisson(poisson_model, X_test, y_test_count))

    # simpan model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, MODELS_DIR / "rf_model.pkl")
    joblib.dump(svm_model, MODELS_DIR / "svm_model.pkl")
    joblib.dump(poisson_model, MODELS_DIR / "poisson_model.pkl")
    print("\nModel disimpan di folder:", MODELS_DIR)

    # simpan skor ke CSV
    df_scores = pd.DataFrame(scores)
    df_scores.to_csv(RESULTS_PATH, index=False)
    print("Skor model disimpan di:", RESULTS_PATH)


if __name__ == "__main__":
    main()
