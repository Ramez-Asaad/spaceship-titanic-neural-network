"""Tests for the from-scratch network.

The gradient check is the important one: it is what proves the hand-written
backpropagation actually computes the derivative of the loss.
"""

import numpy as np
import pytest

from src.scratch_nn import (
    NeuralNetwork,
    binary_cross_entropy,
    relu,
    relu_derivative,
    sigmoid,
)


def test_sigmoid_is_stable_at_extremes():
    out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0, abs=1e-12)


def test_relu_and_derivative():
    z = np.array([-2.0, 0.0, 3.0])
    assert np.allclose(relu(z), [0.0, 0.0, 3.0])
    assert np.allclose(relu_derivative(z), [0.0, 0.0, 1.0])


def test_bce_rewards_confident_correct_predictions():
    y = np.array([[1.0], [0.0]])
    confident = binary_cross_entropy(y, np.array([[0.99], [0.01]]))
    wrong = binary_cross_entropy(y, np.array([[0.01], [0.99]]))
    assert confident < wrong


def test_xavier_init_shapes_and_zero_biases():
    net = NeuralNetwork([5, 8, 3, 1], seed=0)
    assert [w.shape for w in net.weights] == [(5, 8), (8, 3), (3, 1)]
    assert all(np.all(b == 0) for b in net.biases)
    # Glorot uniform bound for the first layer.
    assert np.abs(net.weights[0]).max() <= np.sqrt(6.0 / (5 + 8))


def test_gradient_check_matches_numerical_gradient():
    """Analytic backprop must agree with finite differences."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 5))
    y = rng.integers(0, 2, size=(16, 1)).astype(float)
    net = NeuralNetwork([5, 12, 12, 1], learning_rate=0.01, dropout_rate=0.2, seed=1)
    assert net.gradient_check(X, y) < 1e-6


def test_network_learns_a_separable_pattern():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    net = NeuralNetwork([4, 32, 1], learning_rate=0.05, dropout_rate=0.0, seed=3)
    net.fit(X, y, epochs=25, batch_size=32, verbose_every=0)
    assert net.score(X, y) > 0.95


def test_loss_decreases_during_training():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    net = NeuralNetwork([3, 16, 1], learning_rate=0.05, dropout_rate=0.0, seed=4)
    net.fit(X, y, epochs=20, batch_size=32, verbose_every=0)
    assert net.history["loss"][-1] < net.history["loss"][0]


def test_dropout_is_inactive_at_inference():
    """Two identical predict calls must return identical values."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(10, 4))
    net = NeuralNetwork([4, 16, 1], dropout_rate=0.5, seed=5)
    assert np.allclose(net.predict_proba(X), net.predict_proba(X))


def test_probabilities_are_in_unit_interval():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(30, 4)) * 100
    proba = NeuralNetwork([4, 16, 1], seed=6).predict_proba(X)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 4))
    net = NeuralNetwork([4, 8, 1], learning_rate=0.02, dropout_rate=0.3, seed=7)
    before = net.predict_proba(X)

    path = tmp_path / "net.npz"
    net.save(path)
    restored = NeuralNetwork.load(path)

    assert np.allclose(before, restored.predict_proba(X))
    assert restored.layer_sizes == net.layer_sizes
    assert restored.learning_rate == pytest.approx(net.learning_rate)
