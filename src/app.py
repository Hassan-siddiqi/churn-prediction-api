import numpy as np
from joblib import load
from fastapi import FastAPI, HTTPException
from .schemas import ChurnInput

app = FastAPI(title="Churn Prediction API")

# Load saved model + scaler (must run train.py first)
try:
    model = load("models/model.joblib")
    scaler = load("models/scaler.joblib")
except Exception:
    model = None
    scaler = None

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health():
    ok = (model is not None) and (scaler is not None)
    return {"status": "ok" if ok else "model_not_loaded"}

@app.post("/predict")
def predict(data: ChurnInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Run: python src/train.py")

    features = np.array([[
        data.tenure,
        data.monthly_charges,
        data.total_charges,
        data.senior_citizen
    ]], dtype=float)

    features_scaled = scaler.transform(features)

    prob = float(model.predict_proba(features_scaled)[0][1])  # churn probability
    pred = 1 if prob >= 0.5 else 0

    return {
        "prediction": pred,
        "churn_probability": round(prob, 4),
        "meaning": "Likely to churn" if pred == 1 else "Likely to stay"
    }