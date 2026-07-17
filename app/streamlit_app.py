"""Streamlit demo for the Spaceship Titanic classifier.

The model behind this page is the from-scratch NumPy network in
``src/scratch_nn.py`` — no TensorFlow at inference, just a matrix multiply per
layer against weights loaded from a .npz file.

Run locally:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.data import Preprocessor  # noqa: E402
from src.scratch_nn import NeuralNetwork  # noqa: E402

MODEL_PATH = config.MODELS_DIR / "nn_v2.npz"
PRE_PATH = config.MODELS_DIR / "preprocessor_v2.json"

st.set_page_config(page_title="Spaceship Titanic — Transport Predictor", page_icon="🚀", layout="wide")


@st.cache_resource
def load_model() -> tuple[NeuralNetwork, Preprocessor]:
    return NeuralNetwork.load(MODEL_PATH), Preprocessor.load(PRE_PATH)


@st.cache_data
def load_metrics() -> dict:
    import json

    path = config.MODELS_DIR / "metrics_v2.json"
    return json.loads(path.read_text()) if path.exists() else {}


if not MODEL_PATH.exists() or not PRE_PATH.exists():
    st.error(
        "No trained model found. Train one first:\n\n"
        "```\npython -m src.train --model v2\n```"
    )
    st.stop()

model, pre = load_model()
metrics = load_metrics()

st.title("🚀 Spaceship Titanic — Transport Predictor")
st.caption(
    "Predicts whether a passenger was transported to an alternate dimension. "
    "Served by a neural network written from scratch in NumPy — hand-rolled "
    "forward pass, backprop, Xavier init and dropout. No deep-learning framework at inference."
)

if metrics:
    a, b, c = st.columns(3)
    a.metric("Validation accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("F1 score", f"{metrics['f1']:.3f}")
    c.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Passenger record")

    c1, c2 = st.columns(2)
    home = c1.selectbox("Home planet", ["Earth", "Europa", "Mars", "Unknown"])
    dest = c2.selectbox(
        "Destination", ["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22", "Unknown"]
    )

    c3, c4 = st.columns(2)
    age = c3.slider("Age", 0, 80, 27)
    group_size = c4.slider(
        "Travel group size", 1, 8, 1, help="Passengers booked under the same group ID."
    )

    c5, c6, c7 = st.columns([1, 1, 1])
    deck = c5.selectbox("Cabin deck", list("ABCDEFGT") + ["Unknown"], index=5)
    side = c6.selectbox("Cabin side", ["P (port)", "S (starboard)", "Unknown"])
    cabin_num = c7.number_input("Cabin number", min_value=0, max_value=2000, value=100)

    c8, c9 = st.columns(2)
    cryo = c8.toggle(
        "In cryosleep",
        value=False,
        help="Passengers in cryosleep are confined to their cabins and cannot spend.",
    )
    vip = c9.toggle("VIP passenger", value=False)

    st.markdown("**Amenity spending**")
    if cryo:
        st.info("Cryosleep passengers are confined to their cabins — all spending is zero.")
        spend = dict.fromkeys(config.SPEND_COLUMNS, 0.0)
    else:
        s1, s2, s3 = st.columns(3)
        s4, s5 = st.columns(2)
        spend = {
            "RoomService": s1.number_input("Room service", 0.0, 30000.0, 0.0, step=50.0),
            "FoodCourt": s2.number_input("Food court", 0.0, 30000.0, 0.0, step=50.0),
            "ShoppingMall": s3.number_input("Shopping mall", 0.0, 30000.0, 0.0, step=50.0),
            "Spa": s4.number_input("Spa", 0.0, 30000.0, 0.0, step=50.0),
            "VRDeck": s5.number_input("VR deck", 0.0, 30000.0, 0.0, step=50.0),
        }

record = pd.DataFrame(
    [
        {
            "PassengerId": "0001_01",
            "HomePlanet": None if home == "Unknown" else home,
            "CryoSleep": bool(cryo),
            "Cabin": None if deck == "Unknown" else f"{deck}/{int(cabin_num)}/{side[0]}",
            "Destination": None if dest == "Unknown" else dest,
            "Age": float(age),
            "VIP": bool(vip),
            **spend,
            "GroupSize": float(group_size),
        }
    ]
)

with right:
    st.subheader("Prediction")

    proba = float(model.predict_proba(pre.transform(record)).ravel()[0])
    transported = proba >= 0.5

    if transported:
        st.success(f"### Transported\nProbability: **{proba:.1%}**")
    else:
        st.warning(f"### Not transported\nProbability: **{proba:.1%}**")

    st.progress(proba)
    st.caption("Decision threshold: 0.50")

    # Occlusion-style sensitivity: re-score the passenger with one field reset to
    # a neutral value and report how far the probability moves. It is a cheap,
    # honest read on what this particular prediction hinges on -- not a global
    # feature importance.
    st.markdown("**What is driving this prediction?**")
    neutral = {
        "CryoSleep": False,
        "VIP": False,
        "Age": 27.0,
        "GroupSize": 1.0,
        "HomePlanet": None,
        "Destination": None,
        "Cabin": None,
    }
    deltas = []
    for field, value in neutral.items():
        probe = record.copy()
        probe.loc[0, field] = value
        if field == "CryoSleep" and value is False:
            for s in config.SPEND_COLUMNS:
                probe.loc[0, s] = record.loc[0, s]
        shifted = float(model.predict_proba(pre.transform(probe)).ravel()[0])
        deltas.append({"Feature": field, "Shift": proba - shifted})

    probe = record.copy()
    for s in config.SPEND_COLUMNS:
        probe.loc[0, s] = 0.0
    shifted = float(model.predict_proba(pre.transform(probe)).ravel()[0])
    deltas.append({"Feature": "Amenity spending", "Shift": proba - shifted})

    chart = (
        pd.DataFrame(deltas)
        .assign(Magnitude=lambda d: d["Shift"].abs())
        .sort_values("Magnitude", ascending=False)
        .head(5)
        .set_index("Feature")
    )
    st.bar_chart(chart["Shift"], horizontal=True, height=240)
    st.caption(
        "Change in transport probability versus a baseline passenger with that "
        "field reset to a neutral value."
    )

with st.expander("How this model works"):
    st.markdown(
        f"""
**Architecture** — `{' → '.join(str(s) for s in model.layer_sizes)}`: {len(model.layer_sizes) - 2}
hidden ReLU layers of 128 units with dropout, sigmoid output, trained on binary
cross-entropy with mini-batch SGD.

**Written from scratch** — forward propagation, backpropagation, Xavier
initialisation, inverted dropout and the SGD update are all implemented directly
in NumPy in `src/scratch_nn.py`. Backprop is verified against numerical
gradients (max relative error ~7e-08).

**The features that matter** — the strongest signal in this dataset is amenity
spending and cryosleep: a passenger in cryosleep spends nothing and is
transported far more often than not. The original 2024 version of this project
dropped all five spending columns, which cost it roughly 8 points of accuracy.
See the README for the full before/after.
"""
    )
