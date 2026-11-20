"""
01_eda.py — Exploratory Data Analysis
Industrial Sensor Anomaly Detection | Rajat Bhati
-------------------------------------------------
Run: python scripts/01_eda.py
Output: saves all charts to outputs/eda/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs("outputs/eda", exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
TARGET = "Machine failure"

# ── Load Dataset ──────────────────────────────────────────────────────────────
def load_data():
    candidates = [
        "data/raw/ai4i2020.csv",
        "data/raw/predictive_maintenance.csv",
        "data/raw/dataset.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✅ Loaded: {path}  ({len(df)} rows, {len(df.columns)} cols)")
            return df
    raise FileNotFoundError(
        "Dataset not found. Place it in data/raw/ and rename to ai4i2020.csv"
    )

df = load_data()

# ── Basic Info ────────────────────────────────────────────────────────────────
print("\n── DATASET OVERVIEW ──────────────────────────────────────")
print(df.info())
print("\n── FIRST 5 ROWS ──────────────────────────────────────────")
print(df.head())
print("\n── MISSING VALUES ────────────────────────────────────────")
print(df.isnull().sum())
print("\n── STATISTICAL SUMMARY ───────────────────────────────────")
print(df[FEATURE_COLS].describe().round(2))

# ── Class Distribution ────────────────────────────────────────────────────────
total = len(df)
failures = df[TARGET].sum()
normal = total - failures
print(f"\n── CLASS DISTRIBUTION ────────────────────────────────────")
print(f"   Normal    : {normal} ({normal/total*100:.1f}%)")
print(f"   Failure   : {failures} ({failures/total*100:.1f}%)")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Normal", "Failure"], [normal, failures],
       color=["#2ECC71", "#E74C3C"], edgecolor="white", linewidth=1.5)
ax.set_title("Class Distribution: Machine Failure", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
for bar, val in zip(ax.patches, [normal, failures]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{val}\n({val/total*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda/01_class_distribution.png", dpi=150)
plt.close()
print("   → Saved: outputs/eda/01_class_distribution.png")

# ── Sensor Feature Distributions ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
colors = ["#3498DB", "#9B59B6", "#E67E22", "#1ABC9C", "#E74C3C"]

for i, col in enumerate(FEATURE_COLS):
    ax = axes[i]
    for label, color, ls in zip([0, 1], ["#2ECC71", "#E74C3C"], ["-", "--"]):
        subset = df[df[TARGET] == label][col]
        subset.hist(ax=ax, bins=40, alpha=0.6, color=color,
                    label="Normal" if label == 0 else "Failure",
                    edgecolor="none", density=True)
    ax.set_title(col, fontsize=11, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

axes[-1].set_visible(False)
plt.suptitle("Sensor Distributions: Normal vs Failure", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("outputs/eda/02_sensor_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: outputs/eda/02_sensor_distributions.png")

# ── Correlation Heatmap ───────────────────────────────────────────────────────
corr_cols = FEATURE_COLS + [TARGET]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            annot_kws={"size": 10, "weight": "bold"})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.xticks(rotation=30, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/eda/03_correlation_heatmap.png", dpi=150)
plt.close()
print("   → Saved: outputs/eda/03_correlation_heatmap.png")

# ── Boxplots by Failure Type ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 5))
for i, col in enumerate(FEATURE_COLS):
    sns.boxplot(data=df, x=TARGET, y=col, ax=axes[i],
                palette={"0": "#2ECC71", "1": "#E74C3C"})
    axes[i].set_title(col.split(" [")[0], fontsize=10, fontweight="bold")
    axes[i].set_xlabel("")
    axes[i].set_xticklabels(["Normal", "Failure"], fontsize=9)

plt.suptitle("Sensor Values: Normal vs Failure (Boxplots)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda/04_boxplots.png", dpi=150)
plt.close()
print("   → Saved: outputs/eda/04_boxplots.png")

# ── Failure Type Breakdown ────────────────────────────────────────────────────
failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_counts = df[failure_cols].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(failure_counts.index, failure_counts.values,
               color=["#E74C3C", "#E67E22", "#F39C12", "#9B59B6", "#3498DB"])
ax.set_xlabel("Number of Occurrences")
ax.set_title("Failure Type Breakdown", fontsize=14, fontweight="bold")
for bar, val in zip(bars, failure_counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/eda/05_failure_types.png", dpi=150)
plt.close()
print("   → Saved: outputs/eda/05_failure_types.png")

# ── Power & Temp Difference Features ─────────────────────────────────────────
df["Power"] = df["Rotational speed [rpm]"] * df["Torque [Nm]"] * (2 * np.pi / 60)
df["Temp Diff [K]"] = df["Process temperature [K]"] - df["Air temperature [K]"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["Power", "Temp Diff [K]"]):
    for label, color in zip([0, 1], ["#2ECC71", "#E74C3C"]):
        df[df[TARGET] == label][col].hist(
            ax=ax, bins=40, alpha=0.6, color=color,
            label="Normal" if label == 0 else "Failure",
            density=True, edgecolor="none"
        )
    ax.set_title(f"Engineered Feature: {col}", fontsize=11, fontweight="bold")
    ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda/06_engineered_features.png", dpi=150)
plt.close()
print("   → Saved: outputs/eda/06_engineered_features.png")

print("\n✅ EDA complete. All charts saved to outputs/eda/")
print("   Key insight: Check which sensors show the largest gap between")
print("   Normal and Failure distributions — those drive model accuracy.")
