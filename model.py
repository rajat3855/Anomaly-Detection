"""
model.py — Model Loading & Prediction
Industrial Sensor Anomaly Detection | Rajat Bhati
"""

import os
import json
import math
import joblib
import numpy as np

MODEL_PATH   = "models/best_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"
FEATURES_PATH = "models/feature_cols.pkl"


def load_artifacts():
    """Load model, scaler, and feature list from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "No trained model found. Run: python scripts/02_train.py first."
        )
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return model, scaler, feature_cols, metrics


# Load once at import time
model, scaler, feature_cols, metrics_data = load_artifacts()
MODEL_NAME = metrics_data.get("best_model", "Random Forest")


def engineer_features(sensor_data: dict) -> np.ndarray:
    """
    Build the full feature vector from raw sensor inputs.
    Adds engineered features: Power and Temp Diff.
    """
    air_temp    = sensor_data["air_temp_k"]
    proc_temp   = sensor_data["process_temp_k"]
    rpm         = sensor_data["rotational_speed_rpm"]
    torque      = sensor_data["torque_nm"]
    tool_wear   = sensor_data["tool_wear_min"]

    # Engineered features
    power     = rpm * torque * (2 * math.pi / 60)
    temp_diff = proc_temp - air_temp

    # Update dict so it can be logged
    sensor_data["power"]     = round(power, 4)
    sensor_data["temp_diff"] = round(temp_diff, 4)

    feature_vector = np.array([[
        air_temp, proc_temp, rpm, torque, tool_wear, power, temp_diff
    ]])
    return feature_vector


def predict(sensor_data: dict) -> dict:
    """
    Run prediction on raw sensor inputs.

    Args:
        sensor_data: dict with keys:
            air_temp_k, process_temp_k, rotational_speed_rpm,
            torque_nm, tool_wear_min

    Returns:
        dict with prediction, label, confidence, model_used
    """
    X = engineer_features(sensor_data)
    X_scaled = scaler.transform(X)

    prediction  = int(model.predict(X_scaled)[0])
    proba       = model.predict_proba(X_scaled)[0]
    confidence  = float(proba[prediction])

    return {
        "prediction":    prediction,
        "label":         "Failure" if prediction == 1 else "Normal",
        "confidence":    round(confidence, 4),
        "confidence_pct": round(confidence * 100, 1),
        "model_used":    MODEL_NAME,
        "probabilities": {
            "normal":  round(float(proba[0]), 4),
            "failure": round(float(proba[1]), 4),
        }
    }


def get_model_metrics() -> dict:
    """Return saved evaluation metrics for all models."""
    return metrics_data
