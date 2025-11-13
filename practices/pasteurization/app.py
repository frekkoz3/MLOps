#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit dashboard for pasteurization process sensors.
Reads from a Flask API (serving.py) /stream endpoint.
Now supports dynamic IP and port configuration.
"""

import streamlit as st
import pandas as pd
import requests
import json
import time
import matplotlib.pyplot as plt

from river import linear_model, optim, preprocessing, metrics, drift

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Pasteurization Dashboard", layout="wide")
st.title("Pasteurization Process – Live Sensor Dashboard")

# Sidebar configuration
st.sidebar.header("🔌 Connection Settings")

default_ip = st.session_state.get("ip", "127.0.0.1")
default_port = st.session_state.get("port", "8001")

ip = st.sidebar.text_input("Server IP", default_ip)
port = st.sidebar.text_input("Server Port", default_port)

# Save for persistence
st.session_state.ip = ip
st.session_state.port = port

# Construct the stream URL dynamically
STREAM_URL = f"http://{ip}:{port}/stream"

st.sidebar.markdown(f"**Connected to:** `{STREAM_URL}`")

# -----------------------------
# PARAMETERS
# -----------------------------
REFRESH_INTERVAL = st.sidebar.slider("Refresh interval (seconds)", 0.1, 5.0, 1.0)
MAX_POINTS = st.sidebar.slider("Number of samples to show", 100, 1000, 300)

# Fixed y-axis ranges for each sensor
Y_RANGES = {
    "T": (0, 80),
    "pH": (6.0, 7.2),
    "Kappa": (4.0, 5.5),
    "Mu": (1.4, 2.4),
    "Tau": (0.0, 1.5),
    "Q_in": (-0.1, 2.0),
    "Q_out": (-0.1, 2.0),
    "P": (0.8, 1.6),
    "dTdt": (-1.0, 1.0),
}
SENSORS = list(Y_RANGES.keys())

# -----------------------------
# FUNCTIONS
# -----------------------------
@st.cache_resource
def get_stream(url):
    """Create a persistent streaming connection."""
    try:
        return requests.get(url, stream=True, timeout=10)
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def stream_data(url):
    """Generator to read data lines from Flask SSE endpoint."""
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            for line in r.iter_lines():
                if line and line.startswith(b"data:"):
                    payload = line.replace(b"data: ", b"").decode("utf-8")
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        st.error(f"Stream error: {e}")

# -------------------------------
# River Model Setup
# -------------------------------
model = { sensor : preprocessing.StandardScaler() | linear_model.LinearRegression(optimizer=optim.SGD(0.05)) for sensor in SENSORS } # learning rate 0.01
mae = { sensor : metrics.MAE() for sensor in SENSORS }
adwin = { sensor : drift.ADWIN(clock=4, grace_period=32) for sensor in SENSORS }

# -----------------------------
# INITIALIZE DATAFRAME
# -----------------------------
PREDICTED_SENSORS = [f"predicted_{sensor}" for sensor in SENSORS]
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp"] + SENSORS + PREDICTED_SENSORS)

placeholder = st.empty()
st.info("🟢 Waiting for live data...")

# -----------------------------
# LIVE PLOTTING LOOP
# -----------------------------
data_gen = stream_data(STREAM_URL)

for sample in data_gen:

    df = st.session_state.data

    # -- COMPUTING PREDICTIONS --
    # This prediction is just an univariate 
    for i, sensor in enumerate(SENSORS):
        if len(df[sensor]) > 2:
            # --- ONLINE LEARNING STEP ---
            x = {sensor : df[sensor].iloc[-1]}
            y = sample[sensor]
            y_pred = model[sensor].predict_one({sensor : y})
            model[sensor].learn_one(x, y)
            y_pred_yesterday = df[f"predicted_{sensor}"].iloc[-1]
            mae[sensor].update(y, y_pred_yesterday)
            error = abs(y - (y_pred or 0))
            adwin[sensor].update(error)
            drift_flag = adwin[sensor].drift_detected
            if drift_flag:
                model[sensor] = preprocessing.StandardScaler() | linear_model.LinearRegression(optimizer=optim.SGD(0.05), intercept_init=y) 
            sample[f"predicted_{sensor}"] = y_pred
            sample[f"drift_history_{sensor}"] = drift_flag
            sample[f"mae_{sensor}"] = error
        else:
            model[sensor] = preprocessing.StandardScaler() | linear_model.LinearRegression(optimizer=optim.SGD(0.05), intercept_init=sample[sensor]) 
            sample[f"predicted_{sensor}"] = Y_RANGES[sensor][0]
            sample[f"drift_history_{sensor}"] = False
            sample[f"mae_{sensor}"] = Y_RANGES[sensor][0]

    df = pd.concat([df, pd.DataFrame([sample])], ignore_index=True)
    st.session_state.data = df

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]
    
    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(14, 8)) # axes note: axes are np.ndarray. they're ordered in row-column style
    axes = axes.flatten()
    # -- PLOTTING RESULTS -- 
    for i, sensor in enumerate(SENSORS):
        ax = axes[i]
        ax.plot(df["timestamp"], df[sensor], label="true value", linewidth=1.0, c = "blue")
        ax.plot(df["timestamp"] + 1, df[f"predicted_{sensor}"], label = "prediction", linewidth = 1.0, c = "orange")
        # ax.plot(df["timestamp"], df[f"mae_{sensor}"], label="MAE", linewidth = 0.5, c="green")
        drift_points = df.loc[df[f"drift_history_{sensor}"] == True, ["timestamp", f"{sensor}"]]
        ax.scatter(drift_points["timestamp"], drift_points[f"{sensor}"], c="red", label="Detected Drift")
        ax.set_title(sensor)
        ax.set_ylim(Y_RANGES[sensor])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")

    plt.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)

    time.sleep(REFRESH_INTERVAL)
