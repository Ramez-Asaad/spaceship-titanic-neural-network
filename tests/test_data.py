"""Tests for both preprocessing pipelines.

The v2 pipeline has to survive a single row of user input from the Streamlit
app, with fields left blank, so most of these tests poke at that path.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import (
    NUMERIC_FEATURES,
    Preprocessor,
    add_group_size,
    report_pipeline,
    split_cabin,
)


def _row(**overrides) -> pd.DataFrame:
    base = dict(
        PassengerId="0001_01",
        HomePlanet="Earth",
        CryoSleep=False,
        Cabin="F/100/S",
        Destination="TRAPPIST-1e",
        Age=27.0,
        VIP=False,
        RoomService=0.0,
        FoodCourt=0.0,
        ShoppingMall=0.0,
        Spa=0.0,
        VRDeck=0.0,
        GroupSize=1.0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


# -- helpers ---------------------------------------------------------------

def test_split_cabin_parses_deck_num_side():
    out = split_cabin(pd.Series(["B/12/P", "F/0/S"]))
    assert out["Cabin_Deck"].tolist() == ["B", "F"]
    assert out["Cabin_Num"].tolist() == [12.0, 0.0]
    assert out["Cabin_Side"].tolist() == ["P", "S"]


def test_split_cabin_handles_missing():
    out = split_cabin(pd.Series([None]))
    assert out["Cabin_Deck"].iloc[0] == "Unknown"
    assert out["Cabin_Side"].iloc[0] == "Unknown"


def test_add_group_size_counts_shared_group_ids():
    df = pd.DataFrame({config.ID: ["0001_01", "0001_02", "0002_01"]})
    assert add_group_size(df)["GroupSize"].tolist() == [2.0, 2.0, 1.0]


# -- v2 preprocessor -------------------------------------------------------

def test_preprocessor_roundtrip_preserves_output(tmp_path):
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    expected = pre.transform(df)

    path = tmp_path / "pre.json"
    pre.save(path)
    restored = Preprocessor.load(path)

    assert restored.feature_names == pre.feature_names
    assert np.allclose(restored.transform(df), expected)


def test_transform_accepts_a_single_row():
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    out = pre.transform(_row())
    assert out.shape == (1, len(pre.feature_names))
    assert np.all(np.isfinite(out))


def test_transform_tolerates_all_missing_categoricals():
    """The app lets every categorical be left as Unknown."""
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    out = pre.transform(_row(HomePlanet=None, Cabin=None, Destination=None, Age=np.nan))
    assert np.all(np.isfinite(out))


def test_cryosleep_inferred_from_spending_when_missing():
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    idx = pre.feature_names.index("CryoSleep")

    spent = pre.transform(_row(CryoSleep=None, RoomService=500.0))
    idle = pre.transform(_row(CryoSleep=None))
    assert spent[0, idx] == 0.0  # billed for room service -> was awake
    assert idle[0, idx] == 1.0  # billed nothing -> assumed asleep


def test_missing_spend_is_treated_as_zero():
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    idx = pre.feature_names.index("NoSpend")
    assert pre.transform(_row(Spa=np.nan))[0, idx] == 1.0


def test_unseen_category_falls_back_to_unknown():
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    out = pre.transform(_row(HomePlanet="Pluto"))
    assert out[0, pre.feature_names.index("HomePlanet_Unknown")] == 1.0
    assert out[0, pre.feature_names.index("HomePlanet_Earth")] == 0.0


def test_numeric_features_are_standardised():
    df = add_group_size(pd.read_csv(config.TRAIN_CSV))
    pre = Preprocessor()
    X = pd.DataFrame(pre.fit_transform(df), columns=pre.feature_names)
    assert np.allclose(X[NUMERIC_FEATURES].mean(), 0.0, atol=1e-6)
    assert np.allclose(X[NUMERIC_FEATURES].std(ddof=0), 1.0, atol=1e-6)


def test_feature_order_is_stable_across_calls():
    df = add_group_size(pd.read_csv(config.TRAIN_CSV).head(200))
    pre = Preprocessor().fit(df)
    assert pre.transform(_row()).shape[1] == pre.transform(df).shape[1]


# -- report pipeline -------------------------------------------------------

def test_report_pipeline_uses_exactly_six_features():
    raw = pd.read_csv(config.TRAIN_CSV).head(500)
    X, y, ids, _ = report_pipeline(raw, is_train=True)
    assert list(X.columns) == [
        "CryoSleep",
        "Cabin_Deck",
        "Cabin_Side",
        "Destination",
        "Age",
        "VIP",
    ]
    assert len(X) == len(y) == len(ids)


def test_report_pipeline_drops_zscore_outliers():
    raw = pd.read_csv(config.TRAIN_CSV)
    kept, _, _, _ = report_pipeline(raw, is_train=True, remove_outliers=True)
    all_rows, _, _, _ = report_pipeline(raw, is_train=True, remove_outliers=False)
    assert len(kept) < len(all_rows) == len(raw)


def test_report_pipeline_reuses_train_age_mean_on_test():
    raw = pd.read_csv(config.TRAIN_CSV).head(500)
    _, _, _, age_mean = report_pipeline(raw, is_train=True)
    test = pd.read_csv(config.TEST_CSV).head(50)
    X, y, _, reused = report_pipeline(test, is_train=False, age_mean=age_mean)
    assert reused == pytest.approx(age_mean)
    assert y is None  # test split has no Transported column
