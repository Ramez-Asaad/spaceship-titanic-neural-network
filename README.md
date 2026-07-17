# Spaceship Titanic: Neural Network from Scratch

A binary classifier for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
competition. Given a passenger record, predict whether they were transported to
an alternate dimension.

The network is written from scratch in NumPy. Forward propagation,
backpropagation, Xavier initialisation, inverted dropout and mini-batch SGD are
implemented directly, with no autograd and no deep learning framework. Keras and
scikit-learn versions are included as baselines to check the hand-written
implementation against.

**[Try the live demo](https://spaceship--titanic.streamlit.app/)**, or run it locally:

```bash
pip install -r requirements.txt
python -m src.train --model v2        # train (~15s, CPU)
streamlit run app/streamlit_app.py    # interactive demo
pytest tests/ -q                      # 24 tests (pip install -r requirements-dev.txt)
```

The full write-up, including the backpropagation derivation, is in
[`reports/technical-report.pdf`](reports/technical-report.pdf).

## Results

Validation accuracy on a stratified 20% holdout of the 8,693 training rows,
seed 42:

| Model | Features | Accuracy | F1 | ROC AUC |
|---|---|---|---|---|
| Logistic regression | 33 | 77.8% | 0.781 | 0.860 |
| Keras equivalent (Adam) | 33 | 79.7% | 0.800 | 0.886 |
| From-scratch NumPy network | 33 | **80.8%** | 0.804 | 0.896 |

The logistic regression row is the linear floor. The network clears it by three
points, which is what justifies using a network at all.

The from-scratch network also comes in ahead of the Keras model built on the
same features. That is not because hand-written NumPy is faster or smarter. The
Keras model uses Adam, which fits the training set harder (86.1% train accuracy
against 79.7% validation) and generalises slightly worse here than plain SGD.

## The network

`src/scratch_nn.py` is the core of the project. Architecture is
`33 -> 128 -> 128 -> 128 -> 1`: three hidden ReLU layers with dropout, sigmoid
output, binary cross-entropy loss.

Implemented by hand:

- Xavier/Glorot uniform initialisation, biases at zero
- Forward propagation with a numerically stable sigmoid that branches on sign so
  `exp` never overflows
- Backpropagation. The `sigmoid + BCE` output gradient collapses to `(A - y)/m`,
  then the chain rule runs back through the dropout masks and the ReLU derivative
- Inverted dropout, scaled at training time so inference needs no correction
- Mini-batch SGD with per-epoch reshuffling

Backpropagation is verified against numerical gradients rather than assumed
correct:

```python
net.gradient_check(X, y)   # -> 6.96e-08 max relative error
```

The check runs as part of the test suite. Central differences agreeing with the
analytic gradient to roughly 1e-8 is what shows the derivation is right.

A trained model serialises to a 280 KB `.npz`. Inference is one matrix multiply
per layer.

## Features

The dataset gives 13 usable columns, engineered up to 33 features.

The strongest signal is amenity spending combined with cryosleep. A passenger in
cryosleep is confined to their cabin, spends exactly nothing, and is transported
far more often than not. Most of the feature work follows from that:

- `Cabin` splits into deck, number and side
- `HomePlanet`, `Destination`, `Cabin_Deck` and `Cabin_Side` are one-hot encoded.
  Deck labels A through T have no natural ordering, so encoding them as integers
  would assert a ranking and a uniform spacing that do not exist
- `GroupSize` and `IsAlone` come from the `gggg_pp` passenger ID, which encodes
  travel groups
- Spending gets `log1p`, plus a `TotalSpend` total and a `NoSpend` flag. Spending
  is heavily right-skewed, and `log1p` compresses the tail without discarding the
  big spenders, who are the most informative passengers
- Missing `CryoSleep` is recovered from spending. Anyone billed for an amenity
  was awake

Every column in the raw data has missing values, roughly 2% each, so imputation
is a real part of the problem rather than an afterthought.

The preprocessor serialises to plain JSON rather than a pickle. Statistics
learned on train (medians, means, standard deviations) are stored as numbers, so
inference needs no scikit-learn and there is no version-coupled binary to break.

## The web app

An interactive demo, [live at
spaceship--titanic.streamlit.app](https://spaceship--titanic.streamlit.app/).
Enter a passenger record, get a transport probability and a breakdown of what
drove it. Run it locally with:

```bash
streamlit run app/streamlit_app.py
```

The app serves the from-scratch network directly, so inference needs numpy and
nothing else. No TensorFlow.

The "what is driving this prediction" panel re-scores the passenger with each
field reset to a neutral value and reports how far the probability moves. It is a
per-prediction sensitivity read, not a global feature importance. Toggling
cryosleep on a Europa passenger moves the probability by about +0.20.

Model behaviour on a few hand-checked passengers, which lines up with the domain:

| Passenger | P(transported) |
|---|---|
| In cryosleep, no spending | 0.86 |
| Awake, spent 2,900 across amenities | 0.09 |
| Everything unknown | 0.40, near the 50/50 base rate |

## Training

```bash
python -m src.train --model v2        # the shipped model
python -m src.train --model logreg    # linear baseline
python -m src.train --model keras     # Keras equivalent (pip install -r requirements-keras.txt)
python -m src.train --model report    # the 2024 coursework pipeline, reproduced
python -m src.predict                 # write a Kaggle submission
```

Learning rate is 0.01. At 0.001 with plain SGD the model has not converged by the
end of training; at 0.05 it overfits, reaching 88.9% train against 79.9%
validation.

TensorFlow is optional and only used for the Keras baselines. The from-scratch
model, the tests and the app need nothing beyond numpy, pandas and streamlit.

## Layout

```
.
├── app/
│   └── streamlit_app.py     # interactive demo
├── data/                    # Kaggle train/test splits
├── models/                  # trained weights + preprocessor stats (tracked, 280 KB)
├── reports/
│   ├── technical-report.pdf # write-up of the current approach, with the maths
│   ├── technical-report.tex # its LaTeX source
│   └── figures/             # generated by src/figures.py
├── src/
│   ├── config.py            # paths and hyperparameters
│   ├── data.py              # preprocessing
│   ├── scratch_nn.py        # the from-scratch NumPy network
│   ├── keras_models.py      # Keras / scikit-learn baselines
│   ├── train.py             # training CLI
│   ├── predict.py           # Kaggle submission generator
│   └── figures.py           # report figures
└── tests/                   # 24 tests
```

[`reports/technical-report.pdf`](reports/technical-report.pdf) is the full
write-up: the backpropagation derivation, why sigmoid pairs with cross-entropy
rather than squared error, the gradient verification, and the error analysis.

## Data

From the [Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic):
8,693 training rows and 4,277 test rows. The target is near-perfectly balanced at
50.4% against 49.6%, so plain accuracy is a fair headline metric.

## History

This started as coursework for Neural Networks (AIE231) at Alamein International
University, Fall 24/25, by:

- Ramez Ezzat (22100506)
- Alaaeldin Ibrahim (22101463)
- Rana Hossam (22101478)
- Ahmed Fathy (22101981)

The repository has since been reworked. The original submission was a report and
a zip of data with no runnable code, so the model was rebuilt from the report,
then audited against the data. The audit found three problems: the amenity
spending columns and `HomePlanet` had been dropped, cabin decks were encoded as
integers, and a z-score filter was deleting 228 rows of real data. Fixing those
took the model from 73.4% to 80.8%. The original pipeline is still reproducible
with `python -m src.train --model report`.

## License

MIT, see [LICENSE](LICENSE).
