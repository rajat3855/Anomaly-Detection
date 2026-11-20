"""
02_train.py — Model Training & Evaluation
Industrial Sensor Anomaly Detection | Rajat Bhati
-------------------------------------------------
Run: python scripts/02_train.py
Output: saves trained models to models/ and charts to outputs/models/
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("models", exist_ok=True)
sns.set_theme(style="darkgrid")

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET = "Machine failure"

# ── Load & Feature Engineer ───────────────────────────────────────────────────
def load_data():
    candidates = [
        "data/raw/ai4i2020.csv",
        "data/raw/predictive_maintenance.csv",
        "data/raw/dataset.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("Dataset not found in data/raw/")

print("── LOADING DATA ──────────────────────────────────────────")
df = load_data()

# Engineer extra features (same as EDA)
df["Power"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * np.pi / 60)
df["Temp Diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]

ALL_FEATURES = FEATURE_COLS + ["Power", "Temp Diff [K]"]

X = df[ALL_FEATURES]
y = df[TARGET]
print(f"   Features: {ALL_FEATURES}")
print(f"   Samples: {len(X)} | Failures: {y.sum()} ({y.mean()*100:.1f}%)")

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(ALL_FEATURES, "models/feature_cols.pkl")
print("   ✅ Scaler saved to models/scaler.pkl")

# ── Train 3 Models ────────────────────────────────────────────────────────────
print("\n── TRAINING MODELS ───────────────────────────────────────")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        max_depth=10, random_state=42, n_jobs=-1
    ),
    "Isolation Forest": IsolationForest(
        contamination=0.034, random_state=42, n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n   Training: {name}")

    if name == "Isolation Forest":
        # Unsupervised — fit on training features only
        model.fit(X_train_scaled)
        raw_preds = model.predict(X_test_scaled)
        # Isolation Forest returns -1 (anomaly) / 1 (normal) → convert to 0/1
        y_pred = np.where(raw_preds == -1, 1, 0)
        # No predict_proba for IF, so use decision function (inverted)
        scores = -model.decision_function(X_test_scaled)
        # Normalize to [0,1]
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        scores = model.predict_proba(X_test_scaled)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, scores)

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "scores": scores,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
    }

    print(f"   Accuracy : {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall   : {rec:.4f}")
    print(f"   F1 Score : {f1:.4f}")
    print(f"   ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal','Failure'])}")

# ── Model Comparison Chart ────────────────────────────────────────────────────
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
model_names = list(results.keys())
x = np.arange(len(metrics))
width = 0.25
colors = ["#3498DB", "#2ECC71", "#E74C3C"]

fig, ax = plt.subplots(figsize=(12, 6))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/models/01_model_comparison.png", dpi=150)
plt.close()
print("→ Saved: outputs/models/01_model_comparison.png")

# ── Confusion Matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, model_names):
    cm = confusion_matrix(y_test, results[name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"],
                linewidths=0.5, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/models/02_confusion_matrices.png", dpi=150)
plt.close()
print("→ Saved: outputs/models/02_confusion_matrices.png")

# ── ROC Curves ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#3498DB", "#2ECC71", "#E74C3C"]
for name, color in zip(model_names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]["scores"])
    auc = results[name]["roc_auc"]
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("outputs/models/03_roc_curves.png", dpi=150)
plt.close()
print("→ Saved: outputs/models/03_roc_curves.png")

# ── Feature Importance (Random Forest) ───────────────────────────────────────
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=ALL_FEATURES).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = ["#E74C3C" if imp == importances.max() else "#3498DB" for imp in importances]
importances.plot(kind="barh", ax=ax, color=colors_bar, edgecolor="none")
ax.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
for bar, val in zip(ax.patches, importances.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/models/04_feature_importance.png", dpi=150)
plt.close()
print("→ Saved: outputs/models/04_feature_importance.png")

# ── Pick Best Model & Save ────────────────────────────────────────────────────
# Best by F1 (best for imbalanced classification)
best_name = max(
    [n for n in model_names if n != "Isolation Forest"],
    key=lambda n: results[n]["f1"]
)
best_model = results[best_name]["model"]
joblib.dump(best_model, "models/best_model.pkl")

# Save metrics summary
metrics_summary = {
    name: {k: v for k, v in res.items() if k not in ["model", "y_pred", "scores"]}
    for name, res in results.items()
}
with open("models/metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": metrics_summary}, f, indent=2)

print(f"\n✅ Best model: {best_name}")
print(f"   F1 Score : {results[best_name]['f1']:.4f}")
print(f"   ROC-AUC  : {results[best_name]['roc_auc']:.4f}")
print("   Saved to: models/best_model.pkl")
print("   Metrics : models/metrics.json")
print("\n🎉 Training complete. Run the FastAPI app next: uvicorn app.main:app --reload")
