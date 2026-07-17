"""Generate a Kaggle submission from the trained v2 model.

    python -m src.predict                      # -> submission.csv
    python -m src.predict --output my.csv

Output matches the competition's expected format: PassengerId, Transported
(boolean).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from . import data as D
from .scratch_nn import NeuralNetwork


def generate_submission(output: Path, threshold: float = 0.5) -> pd.DataFrame:
    model_path = config.MODELS_DIR / "nn_v2.npz"
    pre_path = config.MODELS_DIR / "preprocessor_v2.json"

    if not model_path.exists() or not pre_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run `python -m src.train --model v2` first."
        )

    model = NeuralNetwork.load(model_path)
    pre = D.Preprocessor.load(pre_path)

    df, _, ids = D.load_v2("test")
    proba = model.predict_proba(pre.transform(df)).ravel()

    submission = pd.DataFrame(
        {config.ID: ids, config.TARGET: (proba >= threshold)}
    )
    submission.to_csv(output, index=False)

    rate = float(submission[config.TARGET].mean())
    print(f"wrote {len(submission)} predictions -> {output}")
    print(f"predicted transported: {rate:.1%}")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a Kaggle submission file.")
    parser.add_argument("--output", type=Path, default=config.ROOT / "submission.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    generate_submission(args.output, args.threshold)


if __name__ == "__main__":
    main()
