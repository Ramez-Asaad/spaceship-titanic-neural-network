"""Preprocessing for the Spaceship Titanic dataset.

Two pipelines live here:

``Preprocessor`` (v2)
    The pipeline the shipped model uses. 33 features: one-hot categoricals,
    group structure recovered from the passenger ID, log1p on the skewed
    spending columns, and CryoSleep inferred from spending where it is missing.

``report_pipeline``
    A reproduction of the preprocessing described in the original Fall 24/25
    coursework report, kept so that result is still runnable: six features,
    mean/False imputation, ordinal cabin encoding and z-score outlier removal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def load_raw(split: str = "train") -> pd.DataFrame:
    """Read train.csv or test.csv straight from data/."""
    path = config.TRAIN_CSV if split == "train" else config.TEST_CSV
    return pd.read_csv(path)


def split_cabin(cabin: pd.Series) -> pd.DataFrame:
    """Cabin is 'deck/num/side'. Missing cabins become deck/side 'Unknown'."""
    filled = cabin.fillna("Unknown/-1/Unknown")
    parts = filled.str.split("/", expand=True)
    parts.columns = ["Cabin_Deck", "Cabin_Num", "Cabin_Side"]
    parts["Cabin_Num"] = pd.to_numeric(parts["Cabin_Num"], errors="coerce")
    return parts


def add_group_size(df: pd.DataFrame) -> pd.DataFrame:
    """PassengerId is 'gggg_pp' where gggg is a travel group.

    Group size is a dataset-level count, so it is computed here rather than
    inside Preprocessor.transform (which must also work on a single row).
    """
    out = df.copy()
    group = out[config.ID].str.split("_").str[0]
    out["GroupSize"] = group.map(group.value_counts()).astype(float)
    return out


# --------------------------------------------------------------------------
# Pipeline 1: faithful reproduction of the report
# --------------------------------------------------------------------------

def report_pipeline(
    df: pd.DataFrame,
    is_train: bool = True,
    age_mean: float | None = None,
    remove_outliers: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None, pd.Series, float]:
    """Reproduce the original report's preprocessing exactly.

    Returns (X, y, passenger_ids, age_mean). ``age_mean`` is returned so the
    test split can reuse the value learned on train.
    """
    d = df.copy()

    if age_mean is None:
        age_mean = float(d["Age"].mean())

    d["CryoSleep"] = (d["CryoSleep"] == True).astype(int)  # noqa: E712  (NaN -> False)
    d["VIP"] = (d["VIP"] == True).astype(int)  # noqa: E712  (NaN -> False)
    d["Age"] = d["Age"].fillna(age_mean).astype(int)
    d["Destination"] = d["Destination"].fillna("Unknown")

    cabin = split_cabin(d["Cabin"])
    d["Cabin_Deck"] = cabin["Cabin_Deck"].map(config.DECK_MAP).fillna(
        config.DECK_MAP["Unknown"]
    )
    d["Cabin_Side"] = cabin["Cabin_Side"].map(config.SIDE_MAP).fillna(
        config.SIDE_MAP["Unknown"]
    )
    d["Destination"] = d["Destination"].map(config.DESTINATION_MAP).fillna(
        config.DESTINATION_MAP["Unknown"]
    )

    features = ["CryoSleep", "Cabin_Deck", "Cabin_Side", "Destination", "Age", "VIP"]

    if is_train and remove_outliers:
        numeric = d[features].astype(float)
        z = (numeric - numeric.mean()) / numeric.std(ddof=0)
        keep = (z.abs() <= 3).all(axis=1)
        d = d.loc[keep]

    X = d[features].astype(float).reset_index(drop=True)
    ids = d[config.ID].reset_index(drop=True)
    y = (
        d[config.TARGET].astype(bool).astype(int).reset_index(drop=True)
        if config.TARGET in d.columns
        else None
    )
    return X, y, ids, age_mean


# --------------------------------------------------------------------------
# Pipeline 2: the reworked preprocessor
# --------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Age",
    "GroupSize",
    "Cabin_Num",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "TotalSpend",
]
BINARY_FEATURES = ["CryoSleep", "VIP", "IsAlone", "NoSpend"]
ONEHOT_SPECS = {
    "HomePlanet": ["Earth", "Europa", "Mars", "Unknown"],
    "Destination": ["55 Cancri e", "PSO J318.5-22", "TRAPPIST-1e", "Unknown"],
    "Cabin_Deck": list("ABCDEFGT") + ["Unknown"],
    "Cabin_Side": ["P", "S", "Unknown"],
}


class Preprocessor:
    """Fit on train, transform anything, including a single row from the app.

    Everything learned from the data (medians, means, stds) is stored in
    ``self.stats`` and serialises to plain JSON, so inference needs only numpy
    and pandas. No scikit-learn, no pickle.
    """

    def __init__(self) -> None:
        self.stats: dict = {}
        self.feature_names: list[str] = []

    # -- feature construction ------------------------------------------------
    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        if "GroupSize" not in d.columns:
            d["GroupSize"] = 1.0
        d["IsAlone"] = (d["GroupSize"] <= 1).astype(int)

        cabin = split_cabin(d["Cabin"]) if "Cabin" in d.columns else None
        if cabin is not None:
            d["Cabin_Deck"] = cabin["Cabin_Deck"]
            d["Cabin_Side"] = cabin["Cabin_Side"]
            d["Cabin_Num"] = cabin["Cabin_Num"]
        d["Cabin_Num"] = d["Cabin_Num"].replace(-1, np.nan)

        # An unbilled amenity means the passenger did not use it, so a missing
        # spend is a true zero rather than an unknown.
        for col in config.SPEND_COLUMNS:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

        d["TotalSpend"] = d[config.SPEND_COLUMNS].sum(axis=1)
        d["NoSpend"] = (d["TotalSpend"] == 0).astype(int)

        # Passengers in cryosleep are confined to their cabins and cannot spend
        # anything, so spending recovers most of the missing CryoSleep values:
        # anyone who was billed was awake. Where nothing was billed, cryosleep
        # is the likelier explanation.
        cryo = d["CryoSleep"]
        spent = d["TotalSpend"] > 0
        d["CryoSleep"] = np.where(cryo.notna(), cryo == True, ~spent).astype(int)  # noqa: E712

        d["VIP"] = (d["VIP"] == True).astype(int)  # noqa: E712  (NaN -> False)
        d["HomePlanet"] = d["HomePlanet"].fillna("Unknown")
        d["Destination"] = d["Destination"].fillna("Unknown")

        # Spending is heavily right-skewed; log1p keeps the tail from dominating
        # the standardised scale.
        for col in config.SPEND_COLUMNS + ["TotalSpend"]:
            d[col] = np.log1p(d[col].clip(lower=0))

        return d

    def _assemble(self, d: pd.DataFrame) -> pd.DataFrame:
        blocks = [d[NUMERIC_FEATURES].astype(float), d[BINARY_FEATURES].astype(float)]
        for col, categories in ONEHOT_SPECS.items():
            values = d[col].where(d[col].isin(categories), "Unknown")
            dummies = pd.DataFrame(
                {f"{col}_{c}": (values == c).astype(float) for c in categories},
                index=d.index,
            )
            blocks.append(dummies)
        return pd.concat(blocks, axis=1)

    # -- API -----------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        d = self._engineer(df)
        self.stats["medians"] = {
            c: float(d[c].median()) for c in ["Age", "Cabin_Num", "GroupSize"]
        }
        for col, med in self.stats["medians"].items():
            d[col] = d[col].fillna(med)

        matrix = self._assemble(d)
        self.feature_names = list(matrix.columns)
        self.stats["mean"] = matrix[NUMERIC_FEATURES].mean().tolist()
        self.stats["std"] = matrix[NUMERIC_FEATURES].std(ddof=0).replace(0, 1.0).tolist()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        d = self._engineer(df)
        for col, med in self.stats["medians"].items():
            d[col] = d[col].fillna(med)

        matrix = self._assemble(d).reindex(columns=self.feature_names, fill_value=0.0)
        mean = np.asarray(self.stats["mean"])
        std = np.asarray(self.stats["std"])
        matrix[NUMERIC_FEATURES] = (matrix[NUMERIC_FEATURES] - mean) / std
        return matrix.to_numpy(dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"stats": self.stats, "feature_names": self.feature_names}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        payload = json.loads(Path(path).read_text())
        obj = cls()
        obj.stats = payload["stats"]
        obj.feature_names = payload["feature_names"]
        return obj


def load_v2(split: str = "train") -> tuple[pd.DataFrame, pd.Series | None, pd.Series]:
    """Raw frame with group features attached, plus target and ids."""
    df = add_group_size(load_raw(split))
    y = (
        df[config.TARGET].astype(bool).astype(int)
        if config.TARGET in df.columns
        else None
    )
    return df, y, df[config.ID]
