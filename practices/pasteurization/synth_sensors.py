#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic multivariate signals for a pasteurization tank lifecycle:
Production (Idle → Fill → HeatUp → Hold → Cool → Discharge)

Sensors (7): 
  - Temperature (°C)
  - pH
  - Kappa (mS/cm, conductivity)
  - Mu (cP, viscosity)
  - Tau (relative turbidity)
  - Q_in, Q_out (flow)
  - P (pressure)

Derived:
  - dT/dt (temperature change rate)

Output:
  - synthetic_pasteurization_signals.csv
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# -------------------- CONFIG --------------------
DT = 1.0               # seconds per sample
N_BATCHES = 1000
RNG_SEED = 42

# Production setpoint
T_SP = 72.0            # °C
BAND_T = 0.2           # ± around setpoint

# Cooling target
T_COOL_LOW, T_COOL_HIGH = 4.0, 10.0

# Baselines (approx.)
PH_RAW, PH_VAR = 6.65, 0.01
KAPPA_MILK, KAPPA_VAR = 4.8, 0.10
VISC_COLD, VISC_HOT = 2.2, 1.6
TAU_BASE, TAU_VAR = 1.0, 0.05
Q_IN_DESIGN, Q_OUT_DESIGN = 1.5, 1.5
P_BASE, P_VAR = 1.2, 0.10

# Duration ranges (seconds)
PROD_RANGES = {
    "Idle":      (60, 120),
    "Fill":      (30, 60),
    "HeatUp":    (30, 45),
    "Hold":      (15, 25),
    "Cool":      (30, 60),
    "Discharge": (60, 120),
}

# -------------------- UTILS --------------------
rng = np.random.default_rng(RNG_SEED)

def ramp(start: float, end: float, n: int) -> np.ndarray:
    return np.linspace(start, end, n)

def smooth_noise(n: int, scale: float) -> np.ndarray:
    steps = rng.normal(0, scale, size=n)
    return np.cumsum(steps) / max(n, 1)

def mu_of_T(T: np.ndarray) -> np.ndarray:
    return np.interp(T, [10, 72], [VISC_COLD, VISC_HOT])

# -------------------- SEGMENTS (PRODUCTION) --------------------
def seg_idle(n, last_T):
    T = last_T + rng.normal(0, 0.05, n) + smooth_noise(n, 0.005)
    pH = rng.normal(PH_RAW, PH_VAR, n)
    K = rng.normal(KAPPA_MILK, KAPPA_VAR, n)
    Mu = mu_of_T(T) + rng.normal(0, 0.05, n)
    Tau = TAU_BASE * 0.2 + rng.normal(0, TAU_VAR * 0.3, n)
    Qin = np.zeros(n)
    Qout = np.zeros(n)
    P = rng.normal(P_BASE, P_VAR, n)
    return T, pH, K, Mu, Tau, Qin, Qout, P

def seg_fill(n, inlet_T):
    T = inlet_T + rng.normal(0, 0.05, n)
    pH = rng.normal(PH_RAW, PH_VAR, n)
    K = ramp(3.0, KAPPA_MILK, n) + rng.normal(0, 0.05, n)
    Tau = ramp(TAU_BASE * 0.3, TAU_BASE, n) + rng.normal(0, TAU_VAR, n)
    Mu = mu_of_T(T) + rng.normal(0, 0.05, n)
    Qin = Q_IN_DESIGN + rng.normal(0, 0.05, n)
    Qout = np.zeros(n)
    P = rng.normal(P_BASE, P_VAR, n)
    return T, pH, K, Mu, Tau, Qin, Qout, P

def seg_heatup(n, last_T):
    T = ramp(last_T, T_SP, n) + rng.normal(0, 0.05, n)
    pH = rng.normal(PH_RAW, PH_VAR, n)
    K = rng.normal(KAPPA_MILK, KAPPA_VAR, n)
    Mu = mu_of_T(T) + rng.normal(0, 0.05, n)
    Tau = TAU_BASE + rng.normal(0, TAU_VAR, n)
    Qin = np.zeros(n)
    Qout = np.zeros(n)
    P = rng.normal(P_BASE, P_VAR, n)
    return T, pH, K, Mu, Tau, Qin, Qout, P

def seg_hold(n):
    T = T_SP + rng.normal(0, BAND_T / 3, n) + smooth_noise(n, 0.002)
    pH = rng.normal(PH_RAW, PH_VAR, n)
    K = rng.normal(KAPPA_MILK, KAPPA_VAR, n)
    Mu = mu_of_T(T) + rng.normal(0, 0.05, n)
    Tau = TAU_BASE + rng.normal(0, TAU_VAR * 0.5, n)
    Qin = np.zeros(n)
    Qout = np.zeros(n)
    P = rng.normal(P_BASE, P_VAR, n)
    return T, pH, K, Mu, Tau, Qin, Qout, P

def seg_cool(n, last_T):
    target = 0.5 * (T_COOL_LOW + T_COOL_HIGH)
    T = ramp(last_T, target, n) + rng.normal(0, 0.05, n)
    pH = rng.normal(PH_RAW, PH_VAR, n)
    K = rng.normal(KAPPA_MILK, KAPPA_VAR, n)
    Mu = mu_of_T(T) + rng.normal(0, 0.05, n)
    Tau = TAU_BASE + rng.normal(0, TAU_VAR * 0.5, n)
    Qin = np.zeros(n)
    Qout = np.zeros(n)
    P = rng.normal(P_BASE, P_VAR, n)
    return T, pH, K, Mu, Tau, Qin, Qout, P

def seg_discharge(n):
    target = rng.uniform(T_COOL_LOW, T_COOL_HIGH)
    T = target + rng.normal(0, 0.05, n)
    pH = rng.normal(PH_RAW, PH_VAR, n)
    K = rng.normal(KAPPA_MILK, KAPPA_VAR, n)
    Mu = mu_of_T(T) + rng.normal(0, 0.05, n)
    Tau = ramp(TAU_BASE, TAU_BASE * 0.4, n) + rng.normal(0, TAU_VAR, n)
    Qin = np.zeros(n)
    Qout = Q_OUT_DESIGN + rng.normal(0, 0.05, n)
    P = rng.normal(P_BASE, P_VAR, n)
    return T, pH, K, Mu, Tau, Qin, Qout, P

# -------------------- SIMULATION --------------------
def simulate_production_cycle(dur, ctx):
    frames = []

    # Idle
    n = max(1, int(dur["Idle"] / DT))
    T, pH, K, Mu, Tau, Qin, Qout, P = seg_idle(n, ctx["last_T"])
    frames.append(("Idle", T, pH, K, Mu, Tau, Qin, Qout, P))
    last_T = T[-1]

    # Fill
    n = max(1, int(dur["Fill"] / DT))
    T, pH, K, Mu, Tau, Qin, Qout, P = seg_fill(n, ctx["inlet_T"])
    frames.append(("Fill", T, pH, K, Mu, Tau, Qin, Qout, P))
    last_T = T[-1]

    # HeatUp
    n = max(1, int(dur["HeatUp"] / DT))
    T, pH, K, Mu, Tau, Qin, Qout, P = seg_heatup(n, last_T)
    frames.append(("HeatUp", T, pH, K, Mu, Tau, Qin, Qout, P))
    last_T = T[-1]

    # Hold
    n = max(1, int(dur["Hold"] / DT))
    T, pH, K, Mu, Tau, Qin, Qout, P = seg_hold(n)
    frames.append(("Hold", T, pH, K, Mu, Tau, Qin, Qout, P))
    last_T = T[-1]

    # Cool
    n = max(1, int(dur["Cool"] / DT))
    T, pH, K, Mu, Tau, Qin, Qout, P = seg_cool(n, last_T)
    frames.append(("Cool", T, pH, K, Mu, Tau, Qin, Qout, P))
    last_T = T[-1]

    # Discharge
    n = max(1, int(dur["Discharge"] / DT))
    T, pH, K, Mu, Tau, Qin, Qout, P = seg_discharge(n)
    frames.append(("Discharge", T, pH, K, Mu, Tau, Qin, Qout, P))

    ctx["last_T"] = float(T[-1])
    return frames

def to_df(frames, start_t, batch_id):
    """Concatenate frames into a single DataFrame with timestamps."""
    parts = []
    t = start_t
    for (state, T, pH, K, Mu, Tau, Qin, Qout, P) in frames:
        n = len(T)
        ts = np.arange(n) * DT + t
        part = pd.DataFrame({
            "timestamp": ts,
            "batch_id": batch_id,
            "state": state,
            "T": T,
            "pH": pH,
            "Kappa": K,
            "Mu": Mu,
            "Tau": Tau,
            "Q_in": Qin,
            "Q_out": Qout,
            "P": P,
        })
        parts.append(part)
        t = float(ts[-1] + DT)
    df = pd.concat(parts, ignore_index=True)
    return df, t

def simulate_batch(bid, start_time=0.0):
    pdur = {k: int(rng.integers(*PROD_RANGES[k])) for k in PROD_RANGES}
    ctx = {
        "last_T": float(rng.uniform(8, 12)),
        "inlet_T": float(rng.uniform(6, 12))
    }

    frames = simulate_production_cycle(pdur, ctx)
    df, t = to_df(frames, start_time, bid)
    df["dTdt"] = df["T"].diff().fillna(0.0) / DT
    return df, t

def simulate_all(n_batches=N_BATCHES):
    all_df = []
    t = 0.0
    for b in range(1, n_batches + 1):
        dfb, t = simulate_batch(b, start_time=t)
        all_df.append(dfb)
        t += 60.0  # 1 minute between batches
    return pd.concat(all_df, ignore_index=True)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    df = simulate_all(N_BATCHES)
    path = "synthetic_pasteurization_signals.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {path} (rows={len(df):,})")
    print(df.head(3))
