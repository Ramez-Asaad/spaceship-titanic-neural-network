"""Phase 2 of the original project: the same ideas, using external libraries.

The report built three models here, and all three are reproduced as described,
including one that is wrong on purpose (see ``build_single_layer_relu``). They
exist so the from-scratch network in ``scratch_nn.py`` has something to be
checked against.

TensorFlow is an optional dependency. Import errors are raised only when a
Keras model is actually requested, so the from-scratch pipeline and the app
stay usable without it.
"""

from __future__ import annotations

import numpy as np

from . import config


def _require_keras():
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ImportError(
            "TensorFlow is needed for the Keras baselines. "
            "Install it with `pip install -r requirements-keras.txt`, or use the "
            "from-scratch model, which only needs numpy."
        ) from exc
    return keras


# --------------------------------------------------------------------------
# Baseline 1: logistic regression (scikit-learn)
# --------------------------------------------------------------------------

def fit_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 1000):
    """Unregularised logistic regression, as the report specified.

    Returns (model, weights, bias, log_loss). This is the linear floor: the
    neural network has to beat it to justify itself.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=max_iter)
    model.fit(X, y)
    loss = log_loss(y, model.predict_proba(X))
    return model, model.coef_, model.intercept_, loss


# --------------------------------------------------------------------------
# Baseline 2: single dense layer, ReLU output
# --------------------------------------------------------------------------

def build_single_layer_relu(n_features: int):
    """Reproduces the report's `Dense(1, activation='relu')` model verbatim.

    This model is broken by construction and is kept only to document the
    original. A ReLU output cannot express a probability: it is unbounded above
    and flat at zero below, so binary cross-entropy sees values outside (0, 1)
    and any passenger the layer maps to a negative pre-activation gets a hard
    zero with no gradient to recover from. `build_multi_layer` uses sigmoid,
    which is what this should have been.
    """
    keras = _require_keras()
    model = keras.Sequential(
        [keras.layers.Input(shape=(n_features,)), keras.layers.Dense(1, activation="relu")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# Baseline 3: the report's real architecture
# --------------------------------------------------------------------------

def build_multi_layer(
    n_features: int,
    hidden_units: int = config.REPORT_HPARAMS["hidden_units"],
    hidden_layers: int = config.REPORT_HPARAMS["hidden_layers"],
    dropout_rate: float = config.REPORT_HPARAMS["dropout_rate"],
    learning_rate: float = config.REPORT_HPARAMS["learning_rate"],
):
    """3 hidden ReLU layers with dropout, sigmoid output, Adam + BCE.

    This is the Keras twin of ``scratch_nn.NeuralNetwork``. The one deliberate
    difference is the optimiser: the report used Adam here but plain SGD in the
    from-scratch model, so the two are not expected to land on identical numbers.
    """
    keras = _require_keras()
    layers = [keras.layers.Input(shape=(n_features,))]
    for _ in range(hidden_layers):
        layers.append(keras.layers.Dense(hidden_units, activation="relu"))
        layers.append(keras.layers.Dropout(dropout_rate))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))

    model = keras.Sequential(layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    return model
