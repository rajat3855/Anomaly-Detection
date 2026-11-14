"""
database.py — SQLite Prediction Logger
Industrial Sensor Anomaly Detection | Rajat Bhati
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = "data/predictions.db"
os.makedirs("data", exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create predictions table if not exists."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp               TEXT NOT NULL,
            air_temp_k              REAL,
            process_temp_k          REAL,
            rotational_speed_rpm    REAL,
            torque_nm               REAL,
            tool_wear_min           REAL,
            power                   REAL,
            temp_diff               REAL,
            prediction              INTEGER,
            prediction_label        TEXT,
            confidence              REAL,
            model_used              TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(sensor_data: dict, prediction: int, confidence: float, model_used: str):
    """Insert a prediction record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (
            timestamp, air_temp_k, process_temp_k, rotational_speed_rpm,
            torque_nm, tool_wear_min, power, temp_diff,
            prediction, prediction_label, confidence, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        sensor_data.get("air_temp_k"),
        sensor_data.get("process_temp_k"),
        sensor_data.get("rotational_speed_rpm"),
        sensor_data.get("torque_nm"),
        sensor_data.get("tool_wear_min"),
        sensor_data.get("power"),
        sensor_data.get("temp_diff"),
        prediction,
        "Failure" if prediction == 1 else "Normal",
        round(confidence, 4),
        model_used
    ))
    conn.commit()
    conn.close()


def get_history(limit: int = 100):
    """Fetch the last N predictions."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, air_temp_k, process_temp_k, rotational_speed_rpm,
               torque_nm, tool_wear_min, prediction_label, confidence, model_used
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    columns = [
        "id", "timestamp", "air_temp_k", "process_temp_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
        "prediction_label", "confidence", "model_used"
    ]
    return [dict(zip(columns, row)) for row in rows]


def get_summary_stats():
    """Return high-level stats for the dashboard."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 1")
    failures = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 0")
    normals = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0] or 0
    conn.close()
    return {
        "total": total,
        "failures": failures,
        "normals": normals,
        "failure_rate": round(failures / total * 100, 1) if total > 0 else 0,
        "avg_confidence": round(avg_conf * 100, 1)
    }


# Initialize DB on import
init_db()
