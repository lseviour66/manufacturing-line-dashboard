import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import timedelta

# ============================
#   GREEN PROFESSIONAL THEME
# ============================
st.markdown("""
<style>

:root {
    --primary-green: #1B5E20;
    --secondary-green: #4CAF50;
    --light-green: #E8F5E9;
    --text-light: #FFFFFF;
}

body {
    background-color: var(--light-green);
}

h1 {
    color: var(--primary-green);
    font-weight: 700;
}

h2, h3 {
    color: var(--secondary-green);
    font-weight: 600;
}

div[data-testid="metric-container"] {
    background-color: var(--primary-green);
    padding: 15px;
    border-radius: 10px;
    color: var(--text-light);
    box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
}

section[data-testid="stSidebar"] {
    background-color: var(--primary-green);
}

section[data-testid="stSidebar"] * {
    color: var(--text-light) !important;
}

.stButton>button {
    background-color: var(--secondary-green);
    color: var(--text-light);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: var(--primary-green);
}

</style>
""", unsafe_allow_html=True)

# ============================
#   LOAD FULL DATA (DAY-FIRST)
# ============================
full_sensors = pd.read_csv(
    "machine_sensors.csv",
    parse_dates=["timestamp"],
    dayfirst=True
)

full_production = pd.read_csv(
    "production_logs.csv",
    parse_dates=["timestamp"],
    dayfirst=True
)

# ============================
#   SESSION STATE INIT
# ============================
if "index" not in st.session_state:
    st.session_state.index = 0

if "sensors" not in st.session_state:
    st.session_state.sensors = full_sensors.iloc[0:0].copy()

if "production" not in st.session_state:
    st.session_state.production = full_production.iloc[0:0].copy()

if "df" not in st.session_state:
    st.session_state.df = None

# ============================
#   RISK SCORING
# ============================
def compute_risk(df):
    required_cols = [
        "cycle_time_sec",
        "temperature_c",
        "vibration_mm_s",
        "error_code",
        "downtime_min",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    risk = []

    for _, row in df.iterrows():
        score = 0

        cycle = float(row["cycle_time_sec"]) if pd.notna(row["cycle_time_sec"]) else 0
        temp = float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else 0
        vib = float(row["vibration_mm_s"]) if pd.notna(row["vibration_mm_s"]) else 0
        downtime = float(row["downtime_min"]) if pd.notna(row["downtime_min"]) else 0
        error = row["error_code"] if pd.notna(row["error_code"]) else "OK"

        if cycle > 2.0:
            score += 2
        elif cycle > 1.6:
            score += 1

        if temp > 75:
            score += 2
        elif temp > 70:
            score += 1

        if vib > 3.5:
            score += 2
        elif vib > 2.8:
            score += 1

        if error == "FAULT":
            score += 3
        elif error == "WARN":
            score += 1

        if downtime > 10:
            score += 2
        elif downtime > 5:
            score += 1

        risk.append(score)

    df["risk_score"] = risk
    return df

# ============================
#   MERGE + FULL FIELD FILLING
# ============================
def build_df():
    df = pd.merge(
        st.session_state.sensors,
        st.session_state.production,
        on=["timestamp", "machine_id"],
        how="outer"
    ).sort_values("timestamp")

    fill_cols = [
        "timestamp", "machine_id",
        "cycle_time_sec", "temperature_c", "vibration_mm_s",
        "units_produced", "scrap_units", "downtime_min",
        "error_code"
    ]

    for col in fill_cols:
        df[col] = df[col].ffill().bfill()

    df = compute_risk(df)
    return df

# ============================
#   REAL-TIME REPLAY ENGINE
# ============================
def replay_next_row():
    i = st.session_state.index

    if i >= len(full_sensors):
        return

    next_sensor = full_sensors.iloc[[i]]
    next_prod = full_production.iloc[[i]]

    st.session_state.sensors = pd.concat(
        [st.session_state.sensors, next_sensor],
        ignore_index=True
    )

    st.session_state.production = pd.concat(
        [st.session_state.production, next_prod],
        ignore_index=True
    )

    st.session_state.index += 1
    st.session_state.df = build_df()

# ============================
#   RISK EXPLANATION & DISPLAY
# ============================
def explain_risk(row):
    reasons = []

    if row["cycle_time_sec"] > 2.0:
        reasons.append("🔥 Cycle time above 2.0s")
    elif row["cycle_time_sec"] > 1.6:
        reasons.append("⚠️ Cycle time trending high")

    if row["temperature_c"] > 75:
        reasons.append("🔥 Temperature above 75°C")
    elif row["temperature_c"] > 70:
        reasons.append("⚠️ Temperature trending high")

    if row["vibration_mm_s"] > 3.5:
        reasons.append("🔥 Vibration above 3.5 mm/s")
    elif row["vibration_mm_s"] > 2.8:
        reasons.append("⚠️ Vibration trending high")

    if row["error_code"] == "FAULT":
        reasons.append("🛑 Machine fault detected")
    elif row["error_code"] == "WARN":
        reasons.append("⚠️ Warning condition active")

    if row["downtime_min"] > 10:
        reasons.append("🔥 Downtime above 10 minutes")
    elif row["downtime_min"] > 5:
        reasons.append("⚠️ Downtime increasing")

    if not reasons:
        return "Stable conditions"

    return ", ".join(reasons)

def risk_icon_and_color(level):
    if level == "HIGH":
        return "🔥 HIGH", "red"
    if level == "MEDIUM":
        return "⚠️ MEDIUM", "orange"
    return "🟢 LOW", "green"

def predict_risk(df, steps=3):
    if len(df) < 5:
        return []

    recent = df["risk_score"].tail(5).values
    slope = (recent[-1] - recent[0]) / 4

    predictions = []
    next_val = recent[-1]

    for _ in range(steps):
        next_val += slope
        predictions.append(max(0, round(next_val)))

    return predictions

# ============================
#   STREAMLIT UI
# ============================
st.title("Manufacturing Line Health Dashboard (Real-Time Replay)")

df = build_df()

# ----- Risk level + explanation -----
if len(df) > 0:
    latest = df.iloc[-1]
    rv = latest["risk_score"]

    if rv >= 8:
        risk_level = "HIGH"
    elif rv >= 4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    label, color = risk_icon_and_color(risk_level)

    st.markdown(
        f"<h3 style='color:{color};font-weight:700;'>Risk Level: {label}</h3>",
        unsafe_allow_html=True
    )

    if risk_level == "HIGH":
        st.markdown(f"**Reason:** {explain_risk(latest)}")

    with st.expander("Risk Details"):
        st.write("### Latest Risk Breakdown")
        st.write(explain_risk(latest))

        st.write("### Raw Values")
        st.json({
            "timestamp": str(latest["timestamp"]),
            "machine_id": latest["machine_id"],
            "cycle_time_sec": float(latest["cycle_time_sec"]),
            "temperature_c": float(latest["temperature_c"]),
            "vibration_mm_s": float(latest["vibration_mm_s"]),
            "error_code": latest["error_code"],
            "downtime_min": float(latest["downtime_min"]),
            "risk_score": int(latest["risk_score"])
        })

    preds = predict_risk(df)
    if preds:
        st.subheader("Predicted Risk (Next 3 readings)")
        st.write(f"➡️ {preds}")

# ============================
#   2×2 GRID OF CHARTS
# ============================
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if len(df) > 0:
    with col1:
        st.subheader("Cycle Time (seconds)")
        st.line_chart(df.set_index("timestamp")[["cycle_time_sec"]])

    with col2:
        st.subheader("Temperature & Vibration")
        st.line_chart(df.set_index("timestamp")[["temperature_c", "vibration_mm_s"]])

    with col3:
        st.subheader("Units Produced")
        st.line_chart(df.set_index("timestamp")[["units_produced"]])

    with col4:
        st.subheader("Risk Score")
        st.line_chart(df.set_index("timestamp")[["risk_score"]])

# ============================
#   RISK TIMELINE
# ============================
if len(df) > 1:
    st.subheader("Risk Timeline (Last 20 readings)")
    st.line_chart(df.set_index("timestamp")[["risk_score"]].tail(20))

# ============================
#   REAL-TIME REPLAY CONTROL
# ============================
st.subheader("Real-Time Replay")

auto = st.checkbox("Replay data in real time")

if auto:
    replay_next_row()
    time.sleep(2)
    st.rerun()

# ============================
#   RAW DATA TABLE
# ============================
st.subheader("Raw Data")
st.dataframe(df)
