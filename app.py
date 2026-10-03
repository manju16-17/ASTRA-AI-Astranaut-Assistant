import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(page_title="ASTRA-AI", page_icon="🚀", layout="wide")

# ---------- Data simulation ----------
def generate_data(n=120):
    rng = np.random.default_rng(42)
    t = pd.date_range(datetime.now(), periods=n, freq="min")
    df = pd.DataFrame({
        "Time": t,
        "Oxygen (%)": 20.8 + rng.normal(0, .15, n),
        "CO2 (ppm)": 900 + rng.normal(0, 90, n),
        "Cabin Pressure (kPa)": 101.2 + rng.normal(0, .7, n),
        "Temperature (°C)": 23 + rng.normal(0, .7, n),
        "Humidity (%)": 48 + rng.normal(0, 2, n),
        "Radiation (mSv)": .28 + rng.normal(0, .035, n),
        "Battery (%)": np.linspace(88, 75, n) + rng.normal(0, .5, n),
    })
    # Simulated anomalies for an interesting demo
    df.loc[92:98, "Oxygen (%)"] -= np.linspace(2, 4, 7)
    df.loc[102:108, "CO2 (ppm)"] += np.linspace(500, 900, 7)
    df.loc[65:70, "Cabin Pressure (kPa)"] -= np.linspace(3, 5, 6)
    return df

df = generate_data()

features = ["Oxygen (%)","CO2 (ppm)","Cabin Pressure (kPa)",
            "Temperature (°C)","Humidity (%)","Radiation (mSv)","Battery (%)"]

model = IsolationForest(contamination=0.08, random_state=42)
df["ML Status"] = model.fit_predict(df[features])
df["ML Status"] = df["ML Status"].map({1:"NORMAL", -1:"ANOMALY"})

# ---------- Styling ----------
st.markdown("""
<style>
.main {background:#050914;}
.block-container {padding-top:1.5rem;}
h1,h2,h3 {letter-spacing:.5px;}
.metric-card {
    padding:14px; border-radius:14px; background:#0d1626;
    border:1px solid #263754; text-align:center;
}
.alert {
    padding:12px 16px; border-radius:12px; margin:7px 0;
    background:#21151a; border-left:5px solid #ff4b4b;
}
.ok {
    padding:12px 16px; border-radius:12px; margin:7px 0;
    background:#102018; border-left:5px solid #21c77a;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 ASTRA-AI")
st.caption("Intelligent Astronaut Assistance & Spacecraft Health Monitoring System")
st.divider()

latest = df.iloc[-1]
anomalies = df[df["ML Status"] == "ANOMALY"]

# ---------- KPI row ----------
cols = st.columns(7)
metrics = [
    ("O₂", f"{latest['Oxygen (%)']:.1f}%", "🫁"),
    ("CO₂", f"{latest['CO2 (ppm)']:.0f}", "🌫️"),
    ("Pressure", f"{latest['Cabin Pressure (kPa)']:.1f}", "🛰️"),
    ("Temp", f"{latest['Temperature (°C)']:.1f}°C", "🌡️"),
    ("Humidity", f"{latest['Humidity (%)']:.0f}%", "💧"),
    ("Radiation", f"{latest['Radiation (mSv)']:.2f}", "☢️"),
    ("Battery", f"{latest['Battery (%)']:.0f}%", "🔋")
]
for c, (name, value, icon) in zip(cols, metrics):
    c.markdown(f"<div class='metric-card'><h3>{icon}</h3><b>{name}</b><br>{value}</div>", unsafe_allow_html=True)

st.write("")

# ---------- Mission status ----------
left, right = st.columns([1.5, 1])

with left:
    st.subheader("📈 Spacecraft Telemetry")
    parameter = st.selectbox("Select telemetry", features)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Time"], y=df[parameter], mode="lines", name=parameter))
    fig.update_layout(height=360, margin=dict(l=10,r=10,t=25,b=10),
                      paper_bgcolor="#050914", plot_bgcolor="#050914",
                      font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("🤖 ML Health Analysis")
    if len(anomalies) == 0:
        st.markdown("<div class='ok'>🟢 No anomalies detected.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='alert'>🔴 {len(anomalies)} unusual telemetry readings detected by Isolation Forest.</div>", unsafe_allow_html=True)
        st.write("Latest detected events:")
        st.dataframe(anomalies[["Time"] + features].tail(6), hide_index=True, use_container_width=True)

st.subheader("🧠 Mission AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

def assistant_answer(q):
    ql = q.lower()
    if "oxygen" in ql or "o2" in ql:
        return f"Current simulated oxygen level is **{latest['Oxygen (%)']:.1f}%**. The ML monitor is checking for unusual changes."
    if "co2" in ql:
        return f"Current simulated CO₂ is **{latest['CO2 (ppm)']:.0f} ppm**. A sustained rise can trigger a telemetry anomaly."
    if "battery" in ql:
        return f"Battery is currently **{latest['Battery (%)']:.0f}%** in this simulation."
    if "temperature" in ql or "temp" in ql:
        return f"Cabin temperature is **{latest['Temperature (°C)']:.1f}°C**."
    if "pressure" in ql:
        return f"Cabin pressure is **{latest['Cabin Pressure (kPa)']:.1f} kPa**."
    if "radiation" in ql:
        return f"Radiation reading is **{latest['Radiation (mSv)']:.2f} mSv** in the simulated telemetry."
    if "status" in ql or "mission" in ql:
        return f"Mission status: **{'⚠️ ATTENTION REQUIRED' if len(anomalies)>0 else '🟢 NOMINAL'}**. The anomaly detector has flagged {len(anomalies)} readings in the current simulated dataset."
    if "anomal" in ql or "problem" in ql or "warning" in ql:
        return f"The ML system found **{len(anomalies)} anomalous readings**. Simulated events include oxygen drop, CO₂ increase and cabin-pressure deviation."
    if "help" in ql:
        return "Try asking: *What is the oxygen level?*, *Show mission status*, *Any anomalies?*, *What is the battery level?*"
    return "I can monitor spacecraft telemetry and answer questions about oxygen, CO₂, pressure, temperature, radiation, battery, mission status and detected anomalies."

prompt = st.chat_input("Ask ASTRA-AI about the spacecraft...")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    answer = assistant_answer(prompt)
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()

st.divider()
st.subheader("📋 Recent Telemetry")
st.dataframe(df[["Time"] + features + ["ML Status"]].tail(12), hide_index=True, use_container_width=True)

st.caption("Educational simulation only — sensor values and anomaly events are synthetic.")
