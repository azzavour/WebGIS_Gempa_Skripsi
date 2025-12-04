from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'ml_models'

rf_model = joblib.load(MODELS_DIR / 'rf_model.pkl')
svm_model = joblib.load(MODELS_DIR / 'svm_model.pkl')
poisson_model = joblib.load(MODELS_DIR / 'poisson_model.pkl')


def predict_for_grid(magnitude_mean, depth_mean, frequency_week):
    X = [[magnitude_mean, depth_mean, frequency_week]]
    rf_prob = rf_model.predict_proba(X)[0][1]
    svm_prob = svm_model.predict_proba(X)[0][1]
    pois_rate = poisson_model.predict(X)[0]  # bisa kamu konversi ke prob
    return rf_prob, svm_prob, pois_rate
