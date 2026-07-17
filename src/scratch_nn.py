"""A feed-forward neural network written from scratch in NumPy.

This is Phase 1 of the original project: no TensorFlow, no PyTorch, no autograd.
Forward propagation, backpropagation, Xavier initialisation, inverted dropout
and mini-batch SGD are all implemented directly, matching the architecture the
report specifies (3 hidden ReLU layers of 128 units, sigmoid output, binary
cross-entropy loss).

The Streamlit app serves this model rather than the Keras one, so what is
deployed is the hand-written backprop, with numpy as the only runtime
dependency.

Matrices are row-major: X is (n_samples, n_features), W is (n_in, n_out),
b is (1, n_out).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Activations, loss
# --------------------------------------------------------------------------

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Split on sign so exp() never overflows on large-magnitude inputs.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


# --------------------------------------------------------------------------
# Network
# --------------------------------------------------------------------------

class NeuralNetwork:
    """Fully-connected binary classifier trained with mini-batch SGD.

    Parameters
    ----------
    layer_sizes:
        Full topology including input and output, e.g. ``[33, 128, 128, 128, 1]``.
    learning_rate, dropout_rate, seed:
        Training knobs. ``dropout_rate`` is the fraction of units dropped from
        each hidden layer during training.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_acc": []}
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Xavier (Glorot) uniform for weights, zeros for biases.

        The report specifies Xavier. It is the standard choice for sigmoid but
        conservative for ReLU, where He initialisation is the usual pick; kept
        as-specified for faithfulness.
        """
        self.weights, self.biases = [], []
        for n_in, n_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.weights.append(self.rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.biases.append(np.zeros((1, n_out)))

    @property
    def n_layers(self) -> int:
        return len(self.weights)

    # -- forward -------------------------------------------------------------
    def _forward(self, X: np.ndarray, training: bool = False) -> dict:
        """Run a forward pass, caching everything backprop needs."""
        cache: dict = {"A": [X], "Z": [], "masks": []}
        A = X
        for i in range(self.n_layers):
            Z = A @ self.weights[i] + self.biases[i]
            cache["Z"].append(Z)

            if i == self.n_layers - 1:
                A = sigmoid(Z)
                cache["masks"].append(None)
            else:
                A = relu(Z)
                if training and self.dropout_rate > 0:
                    # Inverted dropout: scale at train time so that inference
                    # needs no correction factor.
                    keep = 1.0 - self.dropout_rate
                    mask = (self.rng.random(A.shape) < keep) / keep
                    A = A * mask
                    cache["masks"].append(mask)
                else:
                    cache["masks"].append(None)
            cache["A"].append(A)
        return cache

    # -- backward ------------------------------------------------------------
    def _backward(self, cache: dict, y: np.ndarray) -> tuple[list, list]:
        """Backpropagate and return gradients for every layer."""
        m = y.shape[0]
        grads_w: list[np.ndarray | None] = [None] * self.n_layers
        grads_b: list[np.ndarray | None] = [None] * self.n_layers

        # sigmoid + binary cross-entropy collapse to this clean output gradient.
        dZ = (cache["A"][-1] - y) / m

        for i in reversed(range(self.n_layers)):
            A_prev = cache["A"][i]
            grads_w[i] = A_prev.T @ dZ
            grads_b[i] = dZ.sum(axis=0, keepdims=True)

            if i > 0:
                dA_prev = dZ @ self.weights[i].T
                mask = cache["masks"][i - 1]
                if mask is not None:
                    dA_prev = dA_prev * mask
                dZ = dA_prev * relu_derivative(cache["Z"][i - 1])

        return grads_w, grads_b

    def _update(self, grads_w: list, grads_b: list) -> None:
        for i in range(self.n_layers):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    # -- training ------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose_every: int = 10,
    ) -> "NeuralNetwork":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n = X.shape[0]

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)
            X_shuffled, y_shuffled = X[order], y[order]

            epoch_loss, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                xb = X_shuffled[start : start + batch_size]
                yb = y_shuffled[start : start + batch_size]
                cache = self._forward(xb, training=True)
                epoch_loss += binary_cross_entropy(yb, cache["A"][-1])
                n_batches += 1
                self._update(*self._backward(cache, yb))

            self.history["loss"].append(epoch_loss / max(n_batches, 1))

            if X_val is not None and y_val is not None:
                probs = self.predict_proba(X_val)
                y_val_col = np.asarray(y_val, dtype=np.float64).reshape(-1, 1)
                self.history["val_loss"].append(binary_cross_entropy(y_val_col, probs))
                self.history["val_acc"].append(
                    float(np.mean((probs >= 0.5).astype(int) == y_val_col))
                )

            if verbose_every and (epoch % verbose_every == 0 or epoch == 1):
                msg = f"epoch {epoch:4d}/{epochs}  loss {self.history['loss'][-1]:.4f}"
                if self.history["val_acc"]:
                    msg += (
                        f"  val_loss {self.history['val_loss'][-1]:.4f}"
                        f"  val_acc {self.history['val_acc'][-1]:.4f}"
                    )
                print(msg)

        return self

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self._forward(X, training=False)["A"][-1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).reshape(-1, 1)
        return float(np.mean(self.predict(X) == y))

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"W{i}": w for i, w in enumerate(self.weights)}
        payload.update({f"b{i}": b for i, b in enumerate(self.biases)})
        payload["layer_sizes"] = np.array(self.layer_sizes)
        payload["hparams"] = np.array([self.learning_rate, self.dropout_rate, self.seed])
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        data = np.load(Path(path), allow_pickle=False)
        layer_sizes = data["layer_sizes"].tolist()
        lr, dropout, seed = data["hparams"].tolist()
        model = cls(layer_sizes, learning_rate=lr, dropout_rate=dropout, seed=int(seed))
        model.weights = [data[f"W{i}"] for i in range(len(layer_sizes) - 1)]
        model.biases = [data[f"b{i}"] for i in range(len(layer_sizes) - 1)]
        return model

    # -- verification --------------------------------------------------------
    def gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """Compare analytic gradients against numerical ones.

        Returns the max relative error across all weights. Anything below ~1e-6
        means backprop is doing what calculus says it should. Dropout is off for
        this check because it makes the loss stochastic.
        """
        saved_dropout = self.dropout_rate
        self.dropout_rate = 0.0
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        cache = self._forward(X, training=False)
        grads_w, _ = self._backward(cache, y)

        max_error = 0.0
        for i, W in enumerate(self.weights):
            it = np.nditer(W, flags=["multi_index"])
            checked = 0
            while not it.finished and checked < 20:
                idx = it.multi_index
                original = W[idx]

                W[idx] = original + eps
                loss_plus = binary_cross_entropy(y, self._forward(X)["A"][-1])
                W[idx] = original - eps
                loss_minus = binary_cross_entropy(y, self._forward(X)["A"][-1])
                W[idx] = original

                numerical = (loss_plus - loss_minus) / (2 * eps)
                analytic = grads_w[i][idx]
                denom = max(abs(numerical) + abs(analytic), 1e-12)
                max_error = max(max_error, abs(numerical - analytic) / denom)

                checked += 1
                it.iternext()

        self.dropout_rate = saved_dropout
        return max_error
