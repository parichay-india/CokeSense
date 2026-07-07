"""CokeSense — calibrated coke-quality early warning (CRI / CSR).

Streamlit implementation of the serving loop in the paper (Algorithm 1):
hybrid ridge point prediction + rolling split-conformal interval.
Run locally:      streamlit run app.py
Streamlit Cloud:  point the app at this repository, main file app.py.
"""
from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RAW_FEATURES = ["Coke_WM", "Coke_Ash", "Coke_VM", "Coke_FC",
                "M40", "M10", "Coke_+80mm", "Coke_-25mm"]
NICE = {"Coke_WM": "Total moisture (%)", "Coke_Ash": "Ash (%)",
        "Coke_VM": "Volatile matter (%)", "Coke_FC": "Fixed carbon (%)",
        "M40": "Micum M40 (%)", "M10": "Micum M10 (%)",
        "Coke_+80mm": "+80 mm fraction (%)", "Coke_-25mm": "-25 mm fraction (%)"}

EMBER, STEEL, CHARCOAL, SAND = "#E2572B", "#3D6EA5", "#22262B", "#EDE6DC"


def load_artifacts():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    models = {t: joblib.load(ROOT / "models" / f"hybrid_ridge_{t}.joblib")
              for t in ("CRI", "CSR")}
    return meta, models


def engineered(row: dict) -> dict:
    row = dict(row)
    row["FC_Ash_ratio"] = row["Coke_FC"] / max(row["Coke_Ash"], 1e-6)
    row["Strength_idx"] = row["M40"] - row["M10"]
    row["Ash_VM"] = row["Coke_Ash"] * row["Coke_VM"]
    return row


def rolling_q(buffer: list[float], alpha: float) -> float:
    s = np.sort(np.asarray(buffer, dtype=float))
    n = len(s)
    return float(s[min(ceil((n + 1) * (1 - alpha)), n) - 1])


def predict_one(models, meta, target: str, raw: dict,
                y_lag1: float, y_roll7: float, alpha: float = 0.10):
    feats = engineered(raw)
    feats["y_lag1"], feats["y_roll7"] = y_lag1, y_roll7
    X = pd.DataFrame([feats])[meta["features"]]
    yhat = float(models[target].predict(X)[0])
    q = rolling_q(meta["targets"][target]["buffer_tail"], alpha)
    return yhat, q


# ----------------------------------------------------------------- UI
if __name__ == "__main__":
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="CokeSense", page_icon="🔥", layout="wide")
    meta, models = (st.cache_resource(load_artifacts))()

    st.markdown(
        f"""
        <div style='background:{CHARCOAL};padding:1.1rem 1.4rem;border-radius:14px;
                    border-left:6px solid {EMBER};margin-bottom:0.8rem'>
          <span style='color:{SAND};font-size:1.75rem;font-weight:700'>CokeSense</span>
          <span style='color:{EMBER};font-size:1.05rem'>&nbsp;· calibrated coke-quality early warning</span><br>
          <span style='color:#9AA4AE;font-size:0.92rem'>
            Hybrid soft sensor with a rolling conformal band. Point estimates come from
            routine coke measurements joined to the most recent laboratory feedback;
            the interval is the 90% rolling split-conformal band of the accompanying paper.
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Routine coke measurements")
        raw = {}
        for f in RAW_FEATURES:
            lo, hi = meta["raw_feature_ranges"][f]
            d = meta["raw_feature_defaults"][f]
            raw[f] = st.slider(NICE[f], float(round(lo, 2)), float(round(hi, 2)),
                               float(round(d, 2)), 0.01)
        st.header("Laboratory feedback")
        st.caption("Enter the last confirmed values and the mean of the last seven "
                   "determinations; defaults are the final records of the released dataset.")
        lag = {t: st.number_input(f"Last confirmed {t}",
                                  value=float(round(meta["targets"][t]["default_lag1"], 2)),
                                  step=0.1, format="%.2f") for t in ("CRI", "CSR")}
        roll = {t: st.number_input(f"7-day mean {t}",
                                   value=float(round(meta["targets"][t]["default_roll7"], 2)),
                                   step=0.1, format="%.2f") for t in ("CRI", "CSR")}
        level = st.radio("Interval level", ["90%", "80%"], horizontal=True)
        alpha = 0.10 if level == "90%" else 0.20

    cols = st.columns(2)
    ranges = {"CRI": (16, 30), "CSR": (55, 74)}
    better = {"CRI": "lower", "CSR": "higher"}
    for col, t, accent in zip(cols, ("CRI", "CSR"), (EMBER, STEEL)):
        yhat, q = predict_one(models, meta, t, raw, lag[t], roll[t], alpha)
        lo_r, hi_r = ranges[t]
        with col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=yhat,
                number={"suffix": f"  ± {q:.2f}", "font": {"size": 42}},
                title={"text": f"{t}  <span style='font-size:0.62em;color:#889'>"
                               f"({better[t]} is better)</span>"},
                gauge={
                    "axis": {"range": [lo_r, hi_r]},
                    "bar": {"color": accent, "thickness": 0.28},
                    "steps": [{"range": [max(lo_r, yhat - q), min(hi_r, yhat + q)],
                               "color": "rgba(140,150,160,0.35)"}],
                    "threshold": {"line": {"color": "#C0392B", "width": 3},
                                  "thickness": 0.9, "value": yhat},
                },
            ))
            fig.update_layout(height=300, margin=dict(l=28, r=28, t=54, b=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;color:#788'>{level} rolling conformal band: "
                f"<b>[{yhat - q:.2f}, {yhat + q:.2f}]</b></div>",
                unsafe_allow_html=True,
            )

    with st.expander("Method, calibration state, and caveats"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                "**Model.** Ridge regression on eleven coke-side features "
                "(eight raw, three engineered) joined to the last confirmed index "
                "and its seven-day mean, fitted on the first 80% of a 1,251-day "
                "plant record (2012-2023).\n\n"
                "**Interval.** Split-conformal half-width recomputed from the most "
                "recent 100 absolute residuals whose laboratory confirmations have "
                "arrived (paper Eq. 5, W = 100). Replayed over 2019-2023 this held "
                "90.4% (CRI) and 92.4% (CSR) empirical coverage at the nominal 90%.")
        with c2:
            for t in ("CRI", "CSR"):
                m = meta["targets"][t]
                st.markdown(
                    f"**{t}** - buffer of {len(m['buffer_tail'])} residuals to "
                    f"{m['last_date']}; current q90 = {m['q90']:.2f}, "
                    f"q80 = {m['q80']:.2f}; training rows = {m['train_rows']}.")
            st.markdown(
                "**Caveats.** A screening aid, not a replacement for the reaction "
                "test; retrain and re-seed the buffer when the plant regime or the "
                "blend strategy changes. Scikit-learn is pinned in "
                "`requirements.txt`; artifacts load only under the pinned version.")

    st.caption("CokeSense · paper under double-blind review; citation and DOI will "
               "appear here after acceptance.")
