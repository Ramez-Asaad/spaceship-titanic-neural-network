"""Generate the figures used in reports/technical-report.pdf.

    python -m src.figures

Writes PDF (vector, for LaTeX) and PNG copies into reports/figures/.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

# Categorical slots 1 and 6 from the reference palette. Validated for
# colour-vision deficiency: worst adjacent pair dE 24.7 (protan), well clear of
# the 8.0 target, so the two series stay distinct in print and for CVD readers.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"

FIG_DIR = config.REPORTS_DIR / "figures"


def _style(ax) -> None:
    """Recessive axes and grid, so the data carries the emphasis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {FIG_DIR / name}.pdf")


def training_curves() -> None:
    """Train vs validation loss, and validation accuracy, as two panels.

    Loss and accuracy live on separate axes rather than a twin y-axis: two
    scales on one frame invite false comparisons between unrelated units.
    """
    history = json.loads((config.MODELS_DIR / "history_v2.json").read_text())
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))

    ax1.plot(epochs, history["loss"], color=BLUE, linewidth=2, label="Training")
    ax1.plot(epochs, history["val_loss"], color=ORANGE, linewidth=2, label="Validation")
    ax1.set_xlabel("Epoch", fontsize=9, color=MUTED)
    ax1.set_ylabel("Binary cross-entropy", fontsize=9, color=MUTED)
    ax1.set_title("Loss", fontsize=10, color=INK, loc="left", pad=10)
    legend = ax1.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(MUTED)
    _style(ax1)

    ax2.plot(epochs, history["val_acc"], color=BLUE, linewidth=2)
    final = history["val_acc"][-1]
    ax2.annotate(
        f"{final:.1%}",
        xy=(len(history["val_acc"]), final),
        xytext=(-4, -14),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        ha="right",
    )
    ax2.set_xlabel("Epoch", fontsize=9, color=MUTED)
    ax2.set_ylabel("Accuracy", fontsize=9, color=MUTED)
    ax2.set_title("Validation accuracy", fontsize=10, color=INK, loc="left", pad=10)
    _style(ax2)

    _save(fig, "training-curves")


def model_comparison() -> None:
    """Accuracy by model. One series, so no legend: the title names it."""
    models = [
        ("2024 pipeline\n(6 features)", 0.7336),
        ("Logistic regression", 0.7775),
        ("Keras equivalent", 0.7970),
        ("From-scratch NN", 0.8079),
    ]
    labels = [m[0] for m in models]
    values = [m[1] for m in models]
    # Colour follows the entity, not its rank: the shipped model is the one
    # being highlighted, the rest are context.
    colors = [MUTED, MUTED, MUTED, BLUE]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    bars = ax.barh(labels, values, color=colors, height=0.6)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1%}",
            xy=(value, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )

    ax.set_xlim(0, 0.95)
    ax.set_xlabel("Validation accuracy", fontsize=9, color=MUTED)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.invert_yaxis()
    _style(ax)
    ax.grid(axis="y", visible=False)
    _save(fig, "model-comparison")


def main() -> None:
    training_curves()
    model_comparison()


if __name__ == "__main__":
    main()
