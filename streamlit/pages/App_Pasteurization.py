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

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="Pasteurization Dashboard", layout="wide")
st.title("Pasteurization Process – Live Sensor Dashboard")

# Sidebar configuration
st.sidebar.header("🔌 Connection Settings")

default_ip = st.session_state.get("ip", "127.0.0.1")
default_port = st.session_state.get("port", "8000")

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
    "Q_in": (0.0, 2.0),
    "Q_out": (0.0, 2.0),
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


# -----------------------------
# INITIALIZE DATAFRAME
# -----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp"] + SENSORS)

placeholder = st.empty()
st.info("🟢 Waiting for live data...")

# -----------------------------
# LIVE PLOTTING LOOP
# -----------------------------
data_gen = stream_data(STREAM_URL)

for sample in data_gen:
    df = st.session_state.data
    df = pd.concat([df, pd.DataFrame([sample])], ignore_index=True)

    if len(df) > MAX_POINTS:
        df = df.iloc[-MAX_POINTS:]

    st.session_state.data = df

    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, sensor in enumerate(SENSORS):
        ax = axes[i]
        ax.plot(df["timestamp"], df[sensor], label=sensor, linewidth=1.5)
        ax.set_title(sensor)
        ax.set_ylim(Y_RANGES[sensor])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")

    plt.tight_layout()
    placeholder.pyplot(fig)
    plt.close(fig)

    time.sleep(REFRESH_INTERVAL)
