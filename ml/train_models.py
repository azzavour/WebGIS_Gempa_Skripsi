import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def load_data():
    """
    Ganti fungsi ini dengan baca dataset BMKG-mu.
    Contoh di sini: kita buat data dummy.
    """
    data = {
        'magnitude_mean': [4.5, 5.2, 4.8, 6.0, 5.5, 4.0, 4.3, 5.8],
        'depth_mean': [10, 20, 15, 30, 25, 5, 12, 40],
        'frequency_week': [1, 3, 2, 4, 3, 0, 1, 5],
        # label: 1 = terjadi gempa signifikan minggu depan, 0 = tidak
        'next_week_event': [0, 1, 0, 1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    return df


def main():
    df = load_data()
    X = df[['magnitude_mean', 'depth_mean', 'frequency_week']]
    y = df['next_week_event']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    print("RF Brier:", brier_score_loss(y_test, rf_prob))

    # SVM dengan probabilitas
    svm = SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )
    svm.fit(X_train, y_train)
    svm_prob = svm.predict_proba(X_test)[:, 1]
    print("SVM Brier:", brier_score_loss(y_test, svm_prob))

    # Poisson sebagai baseline (di sini kita pakai target count,
    # tapi untuk demo kita pakai label sama y; di kasus aslimu bisa beda)
    poisson = PoissonRegressor(alpha=1.0, max_iter=1000)
    poisson.fit(X_train, y_train)
    pois_pred = poisson.predict(X_test)
    print("Poisson pred sample:", pois_pred[:5])

    # Simpan model ke folder 'ml_models'
    models_dir = BASE_DIR / 'ml_models'
    models_dir.mkdir(exist_ok=True)

    joblib.dump(rf, models_dir / 'rf_model.pkl')
    joblib.dump(svm, models_dir / 'svm_model.pkl')
    joblib.dump(poisson, models_dir / 'poisson_model.pkl')
    print("Model disimpan di:", models_dir)


if __name__ == '__main__':
    main()
