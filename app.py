"""
dashboard/app.py — Streamlit Dashboard
Industrial Sensor Anomaly Detection | Rajat Bhati
-------------------------------------------------
Run: streamlit run dashboard/app.py
Make sure FastAPI is running first: uvicorn app.main:app --reload
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Industrial Anomaly Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .normal-badge {
        background: #1a4731;
        color: #2ECC71;
        padding: 8px 24px;
        border-radius: 24px;
        font-size: 22px;
        font-weight: bold;
        display: inline-block;
    }
    .failure-badge {
        background: #4a1a1a;
        color: #E74C3C;
        padding: 8px 24px;
        border-radius: 24px;
        font-size: 22px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 Anomaly Detection")
    st.markdown("**Industrial Predictive Maintenance**")
    st.markdown("---")

    health = api_get("/health")
    if health and health.get("status") == "healthy":
        st.success("✅ API Connected")
    else:
        st.error("❌ API Offline — start FastAPI first:\n`uvicorn app.main:app --reload`")

    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    - 📊 3 ML models trained
    - 🔧 AI4I 2020 dataset
    - ⚡ Sub-200ms inference
    - 💾 SQLite prediction logs
    """)
    st.markdown("---")
    st.markdown("Built by **Rajat Bhati**")

# ── Main Layout ───────────────────────────────────────────────────────────────
st.title("🏭 Industrial Sensor Anomaly Detection")
st.markdown("Real-time machine failure prediction | Enter sensor readings to detect anomalies.")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Live Prediction", "📊 History & Analytics", "📈 Model Metrics"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Enter Sensor Readings")

    col1, col2, col3 = st.columns(3)

    with col1:
        air_temp = st.slider(
            "🌡️ Air Temperature (K)",
            min_value=290.0, max_value=310.0,
            value=298.1, step=0.1,
            help="Typical range: 295–305 K"
        )
        process_temp = st.slider(
            "🔥 Process Temperature (K)",
            min_value=300.0, max_value=320.0,
            value=308.6, step=0.1,
            help="Typical range: 305–315 K"
        )

    with col2:
        rpm = st.slider(
            "⚙️ Rotational Speed (RPM)",
            min_value=1000.0, max_value=3000.0,
            value=1551.0, step=1.0,
            help="Typical range: 1200–2800 RPM"
        )
        torque = st.slider(
            "🔩 Torque (Nm)",
            min_value=1.0, max_value=100.0,
            value=42.8, step=0.1,
            help="Typical range: 20–70 Nm"
        )

    with col3:
        tool_wear = st.slider(
            "🛠️ Tool Wear (min)",
            min_value=0.0, max_value=300.0,
            value=108.0, step=1.0,
            help="Minutes of use since last replacement"
        )

        # Show computed derived features live
        import math
        power = rpm * torque * (2 * math.pi / 60)
        temp_diff = process_temp - air_temp
        st.markdown("**Derived Features**")
        st.info(f"⚡ Power: **{power:,.0f} W**")
        st.info(f"🌡️ Temp Diff: **{temp_diff:.1f} K**")

    st.markdown("---")

    if st.button("🔍 Run Prediction", type="primary", use_container_width=True):
        payload = {
            "air_temp_k": air_temp,
            "process_temp_k": process_temp,
            "rotational_speed_rpm": rpm,
            "torque_nm": torque,
            "tool_wear_min": tool_wear
        }
        with st.spinner("Running inference..."):
            result = api_post("/predict", payload)

        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            st.markdown("### Prediction Result")
            res_col1, res_col2, res_col3 = st.columns([2, 1, 1])

            with res_col1:
                if result["prediction"] == 0:
                    st.markdown(
                        '<div style="text-align:center; padding:20px;">'
                        '<span class="normal-badge">✅ NORMAL OPERATION</span>'
                        '</div>', unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div style="text-align:center; padding:20px;">'
                        '<span class="failure-badge">⚠️ FAILURE DETECTED</span>'
                        '</div>', unsafe_allow_html=True
                    )

            with res_col2:
                st.metric("Confidence", f"{result['confidence_pct']}%")
                st.metric("Model", result["model_used"])

            with res_col3:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["probabilities"]["failure"] * 100,
                    title={"text": "Failure Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#E74C3C" if result["prediction"] == 1 else "#2ECC71"},
                        "steps": [
                            {"range": [0, 30], "color": "#1a4731"},
                            {"range": [30, 70], "color": "#4a3a1a"},
                            {"range": [70, 100], "color": "#4a1a1a"},
                        ],
                        "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
                    }
                ))
                fig.update_layout(height=200, margin=dict(t=30, b=0, l=20, r=20),
                                  paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: HISTORY & ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Prediction History")

    stats = api_get("/stats")
    if stats:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", stats["total"])
        m2.metric("Failures Detected", stats["failures"])
        m3.metric("Normal Readings", stats["normals"])
        m4.metric("Failure Rate", f"{stats['failure_rate']}%")

    history = api_get("/history?limit=200")
    if history and history["count"] > 0:
        df = pd.DataFrame(history["records"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        st.markdown("#### Prediction Timeline")
        fig = px.scatter(
            df, x="timestamp", y="confidence",
            color="prediction_label",
            color_discrete_map={"Normal": "#2ECC71", "Failure": "#E74C3C"},
            size="confidence", hover_data=["torque_nm", "rotational_speed_rpm"],
            title="Prediction Confidence Over Time"
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
                          font_color="white", height=350)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Failure Rate Pie Chart")
            counts = df["prediction_label"].value_counts()
            fig2 = px.pie(
                values=counts.values, names=counts.index,
                color=counts.index,
                color_discrete_map={"Normal": "#2ECC71", "Failure": "#E74C3C"},
                hole=0.4
            )
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=300)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("#### Torque Distribution")
            fig3 = px.histogram(
                df, x="torque_nm", color="prediction_label",
                barmode="overlay", opacity=0.7,
                color_discrete_map={"Normal": "#2ECC71", "Failure": "#E74C3C"},
                nbins=30, title="Torque: Normal vs Failure"
            )
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
                               font_color="white", height=300)
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Recent Predictions Table")
        display_df = df.head(20)[["timestamp", "prediction_label", "confidence",
                                   "air_temp_k", "process_temp_k",
                                   "rotational_speed_rpm", "torque_nm", "tool_wear_min"]].copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x*100:.1f}%")
        display_df.columns = ["Time", "Result", "Confidence", "Air Temp (K)",
                               "Process Temp (K)", "RPM", "Torque (Nm)", "Tool Wear (min)"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions yet. Go to the Live Prediction tab and run some predictions!")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL METRICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Evaluation Results")
    metrics = api_get("/metrics")

    if metrics:
        best = metrics.get("best_model", "N/A")
        st.success(f"✅ Best Model: **{best}** (selected by highest F1 Score)")

        results = metrics.get("results", {})
        rows = []
        for model_name, vals in results.items():
            if isinstance(vals, dict) and "accuracy" in vals:
                rows.append({
                    "Model": model_name,
                    "Accuracy": f"{vals['accuracy']*100:.1f}%",
                    "Precision": f"{vals['precision']*100:.1f}%",
                    "Recall": f"{vals['recall']*100:.1f}%",
                    "F1 Score": f"{vals['f1']*100:.1f}%",
                    "ROC-AUC": f"{vals['roc_auc']:.3f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("#### Metric Comparison Chart")
            metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            fig = go.Figure()
            colors = ["#3498DB", "#2ECC71", "#E74C3C"]
            for (name, vals), color in zip(results.items(), colors):
                if isinstance(vals, dict) and "accuracy" in vals:
                    fig.add_trace(go.Bar(
                        name=name,
                        x=[m.replace("_", " ").title() for m in metric_keys],
                        y=[vals.get(m, 0) for m in metric_keys],
                        marker_color=color
                    ))
            fig.update_layout(
                barmode="group", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.1)",
                font_color="white", yaxis_range=[0, 1.1],
                legend=dict(bgcolor="rgba(0,0,0,0.3)")
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### How to Interpret")
        st.markdown("""
        | Metric | What it measures |
        |---|---|
        | **Accuracy** | Overall correct predictions |
        | **Precision** | Of predicted failures, how many were real |
        | **Recall** | Of all actual failures, how many we caught |
        | **F1 Score** | Balance of Precision & Recall (best for imbalanced data) |
        | **ROC-AUC** | Model's ability to discriminate between classes |

        > For industrial safety, **Recall is critical** — missing a real failure is more costly than a false alarm.
        """)
    else:
        st.warning("Could not load metrics. Ensure the FastAPI server is running.")
