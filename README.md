# 🏭 Industrial Sensor Anomaly Detection System



---

## 📌 Project Overview

This system ingests real-time sensor readings from industrial machinery and predicts whether a failure is imminent — enabling proactive maintenance before costly breakdowns occur.

It mirrors the kind of Industrial AI systems built at companies like ITHENA, where sensor data from IoT devices is processed using ML models to generate actionable business insights.

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| Data Analysis | Python, Pandas, NumPy, Matplotlib, Seaborn |
| Machine Learning | scikit-learn (Logistic Regression, Random Forest, Isolation Forest) |
| Feature Engineering | Power (W), Temperature Differential (K) |
| API Backend | FastAPI, Pydantic, Uvicorn |
| Database | SQLite (prediction logging) |
| Dashboard | Streamlit, Plotly |

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)
- 10,000 samples of industrial sensor readings
- 5 sensor features: Air Temp, Process Temp, RPM, Torque, Tool Wear
- 5 failure types: TWF, HDF, PWF, OSF, RNF
- Class imbalance: ~3.4% failure rate (real-world distribution)

---

## 🗂️ Project Structure

```
anomaly-detection/
├── data/
│   ├── raw/             ← Place dataset here (ai4i2020.csv)
│   └── predictions.db   ← Auto-created SQLite DB
├── scripts/
│   ├── 01_eda.py        ← Exploratory Data Analysis
│   └── 02_train.py      ← Model Training & Evaluation
├── app/
│   ├── main.py          ← FastAPI REST API
│   ├── model.py         ← Inference logic
│   └── database.py      ← SQLite logging
├── dashboard/
│   └── app.py           ← Streamlit dashboard
├── models/              ← Saved after training
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metrics.json
├── outputs/             ← Auto-created charts
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset
Place the AI4I 2020 CSV into:
```
data/raw/ai4i2020.csv
```

### 3. Run EDA
```bash
python scripts/01_eda.py
```
Charts saved to `outputs/eda/`

### 4. Train models
```bash
python scripts/02_train.py
```
Models saved to `models/`

### 5. Start the API
```bash
uvicorn app.main:app --reload
```
Interactive docs at: http://localhost:8000/docs

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard at: http://localhost:8501

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | API health check |
| POST | `/predict` | Submit sensor data, get prediction |
| GET | `/history` | Retrieve past predictions |
| GET | `/stats` | Summary statistics |
| GET | `/metrics` | Model evaluation metrics |

### Example Prediction Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "air_temp_k": 298.1,
    "process_temp_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 108
  }'
```

### Example Response
```json
{
  "prediction": 0,
  "label": "Normal",
  "confidence": 0.9732,
  "confidence_pct": 97.3,
  "model_used": "Random Forest",
  "probabilities": { "normal": 0.9732, "failure": 0.0268 },
  "engineered_features": { "power_watts": 6951.0, "temp_diff_k": 10.5 }
}
```

---

## 🧠 ML Models Compared

| Model | Role | Notes |
|---|---|---|
| **Logistic Regression** | Supervised baseline | Fast, interpretable |
| **Random Forest** | Supervised ensemble | Best F1, handles imbalance |
| **Isolation Forest** | Unsupervised anomaly detection | No labels needed |

Best model selected by **F1 Score** — the right metric for imbalanced industrial datasets where missing a failure is more costly than a false alarm.

---

## 🔬 Feature Engineering

Beyond raw sensors, two derived features are computed:

- **Power (W)** = RPM × Torque × (2π / 60) — captures mechanical stress
- **Temp Diff (K)** = Process Temp − Air Temp — captures heat dissipation

Both improve model discrimination as confirmed by feature importance analysis.

---

## 💡 Real-World Applicability

This same pipeline applies directly to:
- **CNC machine failure prediction**
- **Industrial motor health monitoring**
- **Predictive maintenance for automotive manufacturing lines**
- **Quality control in production environments**


