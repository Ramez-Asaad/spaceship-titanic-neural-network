"""Training entry point.

    python -m src.train --model v2       # reworked pipeline, from-scratch NN (default)
    python -m src.train --model report   # faithful reproduction of the 2024 report
    python -m src.train --model keras    # Keras twin, for comparison
    python -m src.train --model logreg   # linear baseline

The v2 run writes models/nn_v2.npz and models/preprocessor_v2.json, which are
what the Streamlit app loads.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

from . import config
from . import data as D
from .scratch_nn import NeuralNetwork


def _report_metrics(name: str, y_true, y_pred, y_proba) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }
    print(
        f"\n{name}: accuracy {metrics['accuracy']:.4f} | "
        f"f1 {metrics['f1']:.4f} | roc_auc {metrics['roc_auc']:.4f}"
    )
    return metrics


def train_v2(epochs: int, learning_rate: float, seed: int) -> dict:
    """Reworked pipeline + from-scratch network. This is the shipped model."""
    df, y, _ = D.load_v2("train")
    pre = D.Preprocessor()
    X = pre.fit_transform(df)
    y = y.to_numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"features {X.shape[1]} | train {len(X_tr)} | val {len(X_val)}")

    model = NeuralNetwork(
        [X.shape[1], 128, 128, 128, 1],
        learning_rate=learning_rate,
        dropout_rate=config.REPORT_HPARAMS["dropout_rate"],
        seed=seed,
    )
    model.fit(
        X_tr, y_tr, epochs=epochs, batch_size=32, X_val=X_val, y_val=y_val, verbose_every=10
    )

    proba = model.predict_proba(X_val).ravel()
    metrics = _report_metrics("v2 (from-scratch NN)", y_val, (proba >= 0.5).astype(int), proba)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODELS_DIR / "nn_v2.npz")
    pre.save(config.MODELS_DIR / "preprocessor_v2.json")
    (config.MODELS_DIR / "metrics_v2.json").write_text(json.dumps(metrics, indent=2))
    (config.MODELS_DIR / "history_v2.json").write_text(json.dumps(model.history, indent=2))
    print(f"saved -> {config.MODELS_DIR / 'nn_v2.npz'}")
    return metrics


def train_report(epochs: int, seed: int) -> dict:
    """The 2024 pipeline and hyperparameters, reproduced as documented."""
    raw = D.load_raw("train")
    X, y, _, _ = D.report_pipeline(raw, is_train=True)
    print(f"features {X.shape[1]} | rows kept {len(X)} of {len(raw)} after z-score filter")

    mean, std = X.mean(), X.std().replace(0, 1.0)
    X_scaled = ((X - mean) / std).to_numpy()
    y = y.to_numpy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = NeuralNetwork(
        [X_scaled.shape[1], 128, 128, 128, 1],
        learning_rate=config.REPORT_HPARAMS["learning_rate"],
        dropout_rate=config.REPORT_HPARAMS["dropout_rate"],
        seed=seed,
    )
    model.fit(
        X_tr, y_tr, epochs=epochs, batch_size=32, X_val=X_val, y_val=y_val, verbose_every=10
    )

    proba = model.predict_proba(X_val).ravel()
    metrics = _report_metrics("report repro", y_val, (proba >= 0.5).astype(int), proba)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODELS_DIR / "nn_report.npz")
    (config.MODELS_DIR / "metrics_report.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def train_keras(epochs: int, seed: int) -> dict:
    from .keras_models import build_multi_layer

    df, y, _ = D.load_v2("train")
    X = D.Preprocessor().fit_transform(df)
    y = y.to_numpy()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    model = build_multi_layer(X.shape[1])
    model.fit(
        X_tr, y_tr, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=2
    )
    proba = model.predict(X_val, verbose=0).ravel()
    return _report_metrics("keras multi-layer", y_val, (proba >= 0.5).astype(int), proba)


def train_logreg(seed: int) -> dict:
    from .keras_models import fit_logistic_regression

    df, y, _ = D.load_v2("train")
    X = D.Preprocessor().fit_transform(df)
    y = y.to_numpy()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model, _, _, loss = fit_logistic_regression(X_tr, y_tr)
    print(f"train log_loss {loss:.4f}")
    proba = model.predict_proba(X_val)[:, 1]
    return _report_metrics("logistic regression", y_val, (proba >= 0.5).astype(int), proba)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spaceship Titanic models.")
    parser.add_argument(
        "--model", choices=["v2", "report", "keras", "logreg"], default="v2"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.model == "v2":
        train_v2(args.epochs, args.learning_rate, args.seed)
    elif args.model == "report":
        train_report(args.epochs, args.seed)
    elif args.model == "keras":
        train_keras(args.epochs, args.seed)
    else:
        train_logreg(args.seed)


if __name__ == "__main__":
    main()
