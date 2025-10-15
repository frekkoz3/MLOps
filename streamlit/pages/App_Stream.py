import time
import requests
import pandas as pd
import streamlit as st
from river import linear_model, optim, preprocessing, metrics, drift
import matplotlib.pyplot as plt

# -------------------------------
# 1. Streamlit Configuration
# -------------------------------
st.set_page_config(
    page_title="Online Weather Prediction (River + Streamlit)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🌤️ Online Weather Prediction – Streamlit + River")
st.markdown("""
This app demonstrates **online learning** using the [River](https://riverml.xyz) library.  
A simple **linear regression model** predicts real-time **temperature** based on **wind speed**
data obtained from the [Open-Meteo API](https://open-meteo.com/).
""")

# -------------------------------
# 2. River Model Setup
# -------------------------------
model = preprocessing.StandardScaler() | linear_model.LinearRegression(optimizer=optim.SGD(0.01)) #learning rate 0.01
mae = metrics.MAE()
adwin = drift.ADWIN()

URL = "https://api.open-meteo.com/v1/forecast?latitude=48.85&longitude=2.35&current_weather=true"

# -------------------------------
# 3. Streamlit Layout Placeholders
# -------------------------------
placeholder_metrics = st.empty()
placeholder_chart = st.empty()
placeholder_drift = st.empty()

# Initialize history dataframe
data_history = pd.DataFrame(columns=["timestamp", "temperature", "wind", "prediction", "mae", "drift_detected"])

# -------------------------------
# 4. Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Streaming Control")
st.sidebar.divider()

# Unique key for checkbox prevents duplication error
run_checkbox = st.sidebar.checkbox("Start Streaming", value=False, key="run_checkbox")
update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 30, 5)
st.sidebar.caption("Live data fetched from Open-Meteo API (Paris, France).")

# -------------------------------
# 5. Main Execution
# -------------------------------
if run_checkbox:
    st.sidebar.success("✅ Streaming active. Uncheck to stop.")
    while st.session_state["run_checkbox"]:
        try:
            # --- Fetch live data ---
            response = requests.get(URL)
            response.raise_for_status()
            data = response.json()["current_weather"]
            temp = data["temperature"]
            wind = data["windspeed"]

            # --- Online learning step ---
            x = {"wind": wind}
            y = temp
            y_pred = model.predict_one(x)
            model.learn_one(x, y)
            mae.update(y, y_pred)
            error = abs(y - (y_pred or 0))
            adwin.update(error)
            drift_flag = adwin.drift_detected

            # --- Update data history ---
            new_row = {
                "timestamp": pd.Timestamp.now(),
                "temperature": y,
                "wind": wind,
                "prediction": y_pred,
                "mae": mae.get(),
                "drift_detected": drift_flag,
            }
            new_row_df = pd.DataFrame([new_row])
            if data_history.empty:
                data_history = new_row_df
            else:
                data_history = pd.concat([data_history, new_row_df]).tail(100)

            # --- Display metrics ---
            with placeholder_metrics.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("🌡️ Actual Temperature (°C)", f"{y:.2f}")
                c2.metric("💨 Wind Speed (km/h)", f"{wind:.2f}")
                c3.metric("📉 Mean Absolute Error", f"{mae.get():.3f}")

            # --- Prediction plot ---
            with placeholder_chart.container():
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data_history["timestamp"], data_history["temperature"], label="Actual Temperature", color="tab:blue")
                ax.plot(data_history["timestamp"], data_history["prediction"], label="Predicted Temperature", color="tab:orange")
                ax.set_xlabel("Time")
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)  # prevents memory leak warning

            # --- Drift detection plot ---
            with placeholder_drift.container():
                st.markdown("### 🧭 Drift Detection (ADWIN)")
                drift_points = data_history[data_history["drift_detected"] == True]
                fig2, ax2 = plt.subplots(figsize=(10, 2))
                ax2.plot(data_history["timestamp"], data_history["mae"], label="MAE", color="tab:green")
                ax2.scatter(drift_points["timestamp"], drift_points["mae"], color="red", label="Detected Drift")
                ax2.set_ylabel("MAE")
                ax2.legend()
                st.pyplot(fig2)
                plt.close(fig2)

            # --- Pause before next update ---
            time.sleep(update_interval)

        except requests.exceptions.RequestException as e:
            st.error(f"⚠️ API Error: {e}")
            time.sleep(10)
        except Exception as e:
            st.error(f"⚠️ Unexpected error: {e}")
            time.sleep(5)

    st.warning("🛑 Streaming stopped by user.")

else:
    st.info("▶️ Enable **Start Streaming** in the sidebar to begin live online prediction.")

# -------------------------------
# 6. Footer
# -------------------------------
st.markdown("---")
st.caption("Developed for 440MI – University of Trieste | Demonstration of Online Learning with Streamlit & River.")
