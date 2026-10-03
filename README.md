# 🚀 ASTRA-AI — Intelligent Astronaut Assistant

ASTRA-AI is an educational Machine Learning project that simulates an astronaut assistant and spacecraft health-monitoring dashboard.

## Features
- Simulated spacecraft telemetry
- Oxygen, CO₂, pressure, temperature, humidity, radiation and battery monitoring
- Isolation Forest anomaly detection
- Interactive Plotly telemetry graphs
- Mission health status
- AI-style astronaut chat assistant
- Automatic warning generation
- Futuristic Streamlit dashboard

## ML Algorithm
**Isolation Forest** is used for unsupervised anomaly detection. It identifies telemetry patterns that differ from the normal operating data.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Suggested presentation flow
1. Problem: astronauts need continuous monitoring of spacecraft health.
2. Solution: ASTRA-AI combines telemetry monitoring with ML anomaly detection.
3. Data: simulated spacecraft sensor readings.
4. ML: Isolation Forest detects unusual patterns.
5. Assistant: users can ask questions about spacecraft status.
6. Dashboard: graphs, metrics and alerts make the information easy to understand.
7. Future scope: real NASA/ESA datasets, speech recognition, text-to-speech, RAG over spacecraft manuals, and edge deployment.

## Important
This is a student/educational simulation. It is not a real spacecraft safety system.
