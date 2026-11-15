"""
main.py — FastAPI Application
Industrial Sensor Anomaly Detection | Rajat Bhati
-------------------------------------------------
Run: uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model    import predict, get_model_metrics
from app.database import log_prediction, get_history, get_summary_stats

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Industrial Sensor Anomaly Detection API",
    description=(
        "Real-time machine failure prediction using sensor data. "
        "Built with FastAPI, scikit-learn, and SQLite. | Rajat Bhati"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────────────────────────
class SensorInput(BaseModel):
    air_temp_k: float = Field(
        ..., ge=290, le=310,
        description="Air temperature in Kelvin (typical: 295–305 K)",
        example=298.1
    )
    process_temp_k: float = Field(
        ..., ge=300, le=320,
        description="Process temperature in Kelvin (typical: 305–315 K)",
        example=308.6
    )
    rotational_speed_rpm: float = Field(
        ..., ge=1000, le=3000,
        description="Rotational speed in RPM (typical: 1200–2800)",
        example=1551.0
    )
    torque_nm: float = Field(
        ..., ge=1, le=100,
        description="Torque in Newton-metres (typical: 20–70 Nm)",
        example=42.8
    )
    tool_wear_min: float = Field(
        ..., ge=0, le=300,
        description="Tool wear in minutes (typical: 0–250 min)",
        example=108.0
    )

class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    confidence_pct: float
    model_used: str
    probabilities: dict
    engineered_features: dict

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running and model is loaded."""
    return {
        "status": "healthy",
        "model": "loaded",
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_failure(sensor: SensorInput):
    """
    Submit sensor readings and receive a real-time anomaly prediction.

    - **Normal (0)**: Machine operating within expected parameters.
    - **Failure (1)**: Anomaly detected — maintenance recommended.
    """
    sensor_dict = sensor.dict()

    try:
        result = predict(sensor_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Log to SQLite
    log_prediction(
        sensor_data=sensor_dict,
        prediction=result["prediction"],
        confidence=result["confidence"],
        model_used=result["model_used"]
    )

    return {
        **result,
        "engineered_features": {
            "power_watts": round(sensor_dict.get("power", 0), 2),
            "temp_diff_k": round(sensor_dict.get("temp_diff", 0), 2),
        }
    }


@app.get("/history", tags=["Data"])
def prediction_history(limit: int = 50):
    """
    Retrieve the last N predictions from the database.
    Default: 50 records. Max recommended: 500.
    """
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="Limit must be between 1 and 1000")
    records = get_history(limit=limit)
    return {
        "count": len(records),
        "records": records
    }


@app.get("/stats", tags=["Data"])
def summary_stats():
    """Return high-level prediction statistics."""
    return get_summary_stats()


@app.get("/metrics", tags=["Model"])
def model_metrics():
    """
    Return saved evaluation metrics for all trained models
    (accuracy, precision, recall, F1, ROC-AUC).
    """
    return get_model_metrics()


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Industrial Sensor Anomaly Detection API",
        "author": "Rajat Bhati",
        "docs": "/docs",
        "endpoints": ["/predict", "/history", "/stats", "/metrics", "/health"]
    }
