# Spaceship Titanic — Neural Network from Scratch

A binary classifier for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition: given a passenger record, predict whether they were transported to
an alternate dimension.

The network is **written from scratch in NumPy** — forward propagation,
backpropagation, Xavier initialisation, inverted dropout and mini-batch SGD are
all implemented directly, with no autograd and no deep-learning framework. Keras
and scikit-learn versions are included as baselines to check the hand-written
implementation against.

Originally coursework for Neural Networks (AIE231) at Alamein International
University, Fall 24/25. This repository is a rebuild: the original submission was
a report and a zip of data with no runnable code. Everything here was
reconstructed from the report, then **audited and improved** — see
[What the original got wrong](#what-the-original-got-wrong).

---

## Results

Validation accuracy on a stratified 20% holdout (8,693 training rows, seed 42):

| Model | Features | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Original report (as documented in 2024) | 6 | 73.9% | — | — |
| **Report reproduction** (this repo, faithful) | 6 | **73.4%** | 0.707 | 0.791 |
| Logistic regression (v2 features) | 33 | 77.8% | 0.781 | 0.860 |
| Keras twin (v2 features, Adam) | 33 | 79.7% | 0.800 | 0.886 |
| **v2 — from-scratch NN** ⭐ | 33 | **80.8%** | 0.804 | 0.896 |

The reproduction landing within 0.6 points of the report's original 73.9% is the
evidence that the reconstruction is faithful. The v2 model then adds **+7.4
points** over it.

Two rows are worth dwelling on:

- **Logistic regression** is the linear floor. The network clears it by 3 points,
  which is what justifies using a network at all — without this row, "80.8%"
  means nothing.
- **The from-scratch network edges out its own Keras twin** (80.8% vs 79.7%) on
  identical features. Not because hand-written NumPy is faster or smarter, but
  because the Keras model uses Adam, which fits the training set harder (86.1%
  train accuracy vs 79.7% validation) and generalises slightly worse here than
  plain SGD. It is a useful reminder that the better optimiser is not
  automatically the better model.

---

## What the original got wrong

The 2024 project scored 73.9%. Rebuilding it surfaced three mistakes, and fixing
them is most of the gain:

**1. It threw away the most predictive features.**
The report reduced the dataset to six columns, dropping `HomePlanet` and all five
amenity-spending columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`,
`VRDeck`). Spending is the single strongest signal in this dataset — a passenger
in cryosleep is confined to their cabin, spends exactly nothing, and is
transported far more often than not. Dropping those columns discarded the
relationship the problem turns on. Restoring them is worth roughly 6 points on
its own.

**2. It forced a false ordering onto categorical variables.**
Decks `A`–`T` were mapped to integers `0`–`7`, which tells the model deck `G` is
"greater than" deck `A` and that the gap between `A` and `B` equals the gap
between `F` and `G`. None of that is true. v2 one-hot encodes them instead.

**3. It deleted real data as "outliers".**
The report dropped every row with any |z-score| > 3, removing 228 passengers.
Amenity spending is heavily right-skewed, so a z-score filter mostly deletes
big spenders — who are precisely the most informative passengers. v2 keeps them
and applies `log1p` instead, which compresses the tail without discarding anyone.

There is also a genuine bug preserved in `src/keras_models.py`: the report's
single-layer baseline used `Dense(1, activation='relu')` for binary
classification. A ReLU output cannot express a probability — it is unbounded
above and flat at zero below, so cross-entropy sees values outside (0, 1) and any
passenger mapped below zero yields no gradient. It is reproduced as-written and
documented, rather than quietly corrected.

### What v2 does instead

- Keeps all 13 usable raw columns, engineered up to 33 features
- One-hot encodes `HomePlanet`, `Destination`, `Cabin_Deck`, `Cabin_Side`
- Splits `Cabin` into deck / number / side
- Derives `GroupSize` and `IsAlone` from the `gggg_pp` passenger ID
- `log1p` on all spending, plus `TotalSpend` and a `NoSpend` flag
- Recovers missing `CryoSleep` from spending: anyone billed for an amenity was awake
- Tuned to `lr=0.01` (the report's `0.001` with plain SGD had not converged; at
  `lr=0.05` the model overfits — 88.9% train against 79.9% validation)

---

## The from-scratch network

`src/scratch_nn.py` is the centrepiece. Architecture `33 → 128 → 128 → 128 → 1`:
three hidden ReLU layers with dropout, sigmoid output, binary cross-entropy.

Implemented by hand:

- **Xavier/Glorot uniform** initialisation, biases at zero
- **Forward propagation** with a numerically stable sigmoid (branches on sign so
  `exp` never overflows)
- **Backpropagation** — the `sigmoid + BCE` output gradient collapses to
  `(A - y)/m`, then the chain rule back through dropout masks and the ReLU derivative
- **Inverted dropout** — scaled at training time, so inference needs no correction
- **Mini-batch SGD** with per-epoch reshuffling

**Backprop is verified against numerical gradients**, not just assumed correct:

```python
net.gradient_check(X, y)   # -> 6.96e-08 max relative error
```

That check runs as part of the test suite (`tests/test_scratch_nn.py`). Central
differences agreeing with the analytic gradient to ~1e-8 is what proves the
derivation is right.

The trained model serialises to a ~280 KB `.npz`, and **the Streamlit app serves
this network** — inference is one matrix multiply per layer, numpy only.

---

## The web app

An interactive demo: enter a passenger record, get a transport probability plus a
breakdown of what drove the prediction.

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The "what is driving this prediction" panel re-scores the passenger with each
field reset to a neutral value and reports how far the probability moves. It is
an honest per-prediction sensitivity read, not a global feature importance — and
it makes the cryosleep effect visible: toggling cryosleep on a Europa passenger
moves the probability by about +0.20.

Sanity checks on the deployed model, which line up with the domain:

| Passenger | P(transported) |
|---|---|
| In cryosleep, no spending | 0.86 |
| Awake, spent 2,900 across amenities | 0.09 |
| Everything unknown | 0.40 (≈ the 50/50 base rate, appropriately uncertain) |

---

## Quickstart

```bash
git clone https://github.com/Ramez-Asaad/spaceship-titanic-neural-network.git
cd spaceship-titanic-neural-network
pip install -r requirements.txt

python -m src.train --model v2        # train the shipped model (~15s, CPU)
streamlit run app/streamlit_app.py    # launch the demo
pytest tests/ -q                      # 24 tests, including the gradient check
```

Other training targets:

```bash
python -m src.train --model report    # faithful 2024 reproduction (6 features, lr=0.001, 1000 epochs)
python -m src.train --model logreg    # linear baseline
python -m src.train --model keras     # Keras twin (needs: pip install -r requirements-keras.txt)
```

TensorFlow is optional and only used for the Keras baselines. The from-scratch
model, the tests and the app need nothing beyond numpy, pandas and streamlit.

---

## Repository layout

```
.
├── app/
│   └── streamlit_app.py     # interactive demo, serves the from-scratch NN
├── data/                    # Kaggle train/test splits
├── models/                  # trained weights + preprocessor stats (tracked, ~280 KB)
├── reports/
│   └── original-report.pdf  # the 2024 coursework report
├── src/
│   ├── config.py            # paths, and the report's documented hyperparameters
│   ├── data.py              # both pipelines: faithful report repro + v2
│   ├── scratch_nn.py        # ⭐ the from-scratch NumPy network
│   ├── keras_models.py      # Phase 2 baselines (Keras / scikit-learn)
│   ├── train.py             # training CLI
│   └── predict.py           # Kaggle submission generator
└── tests/                   # 24 tests
```

The preprocessor serialises to plain JSON rather than a pickle — statistics
learned on train (medians, means, stds) are stored as numbers, so inference needs
no scikit-learn and there is no version-coupled binary to break.

---

## Data

From the [Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic):
8,693 training rows and 4,277 test rows. The target is near-perfectly balanced
(50.4% / 49.6%), so plain accuracy is a fair headline metric here.

Every column has missing values (roughly 2% each) — handling them is a real part
of the problem, not an afterthought.

---

## Credits

Original coursework team (Neural Networks AIE231, Fall 24/25):

- **Ramez Ezzat** (22100506)
- **Alaaeldin Ibrahim** (22101463)
- **Rana Hossam** (22101478)
- **Ahmed Fathy** (22101981)

The 2024 report is preserved verbatim at `reports/original-report.pdf`. The
reconstruction, audit, v2 model and web app in this repository are by
[Ramez Ezzat](https://github.com/Ramez-Asaad).

## License

MIT — see [LICENSE](LICENSE).
