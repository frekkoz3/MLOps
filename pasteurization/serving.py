#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, Response, jsonify, request
import time, json
import numpy as np
import pandas as pd

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from synth_sensors import (
    seg_idle, seg_fill, seg_heatup, seg_hold, seg_cool, seg_discharge,
    PROD_RANGES, DT, rng, mu_of_T, Q_IN_DESIGN, Q_OUT_DESIGN, P_BASE, P_VAR
)

app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------
SEGMENT_FUNCS = {
    "Idle": seg_idle,
    "Fill": seg_fill,
    "HeatUp": seg_heatup,
    "Hold": seg_hold,
    "Cool": seg_cool,
    "Discharge": seg_discharge,
}

def build_timeline():
    """Build random durations for each state and cumulative boundaries."""
    timeline = []
    t0 = 0
    for state, (tmin, tmax) in PROD_RANGES.items():
        dur = int(rng.integers(tmin, tmax))
        timeline.append((state, t0, t0 + dur))
        t0 += dur
    return timeline, t0  # total duration


def simulate_point(t_global: float, ctx=None):
    """
    Simulate one timestamp (in seconds) during a production batch.
    Returns a dict of sensor readings for that time.
    """
    if ctx is None:
        ctx = {"last_T": 10.0, "inlet_T": 8.0}

    timeline, total_dur = build_timeline()
    t_mod = t_global % total_dur  # wrap around if beyond total duration

    for state, t_start, t_end in timeline:
        if t_start <= t_mod < t_end:
            frac = (t_mod - t_start) / (t_end - t_start)
            n = int((t_end - t_start) / DT)
            seg_func = SEGMENT_FUNCS[state]

            # segment args differ for first ones
            if state == "Idle":
                args = (n, ctx["last_T"])
            elif state == "Fill":
                args = (n, ctx["inlet_T"])
            elif state in ["Cool", "HeatUp"]:
                args = (n, ctx["last_T"])
            else:
                args = (n,)

            T, pH, K, Mu, Tau, Qin, Qout, P = seg_func(*args)
            i = min(int(frac * n), n - 1)
            dTdt = float(np.gradient(T)[i])

            return {
                "timestamp": float(t_global),
                "state": state,
                "T": float(T[i]),
                "pH": float(pH[i]),
                "Kappa": float(K[i]),
                "Mu": float(Mu[i]),
                "Tau": float(Tau[i]),
                "Q_in": float(Qin[i]),
                "Q_out": float(Qout[i]),
                "P": float(P[i]),
                "dTdt": dTdt,
            }

    # fallback (shouldn't happen)
    return {"error": f"time {t_global}s outside batch range"}


# -----------------------------
# Flask Endpoints
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "Synthetic Pasteurization Stream API",
        "routes": ["/batch", "/stream", "/point?t=<seconds>"],
    })


@app.route("/batch")
def one_batch():
    from synth_signals import simulate_batch
    df, _ = simulate_batch(1)
    return Response(df.to_json(orient="records"), mimetype="application/json")


@app.route("/stream")
def stream():
    from synth_sensors import simulate_batch
    STREAM_DT = 1.0
    BATCH_ID = 1
    df, _ = simulate_batch(BATCH_ID)

    def generate():
        for _, row in df.iterrows():
            data = row.to_dict()
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(STREAM_DT)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/point")
def point():
    """Return a single simulated reading at a given timestamp (seconds)."""
    try:
        t = float(request.args.get("t", 0))
    except ValueError:
        return jsonify({"error": "Invalid t parameter"}), 400

    data = simulate_point(t)
    return jsonify(data)


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True, threaded=True)
