import numpy as np

from nn.base import Parameter
from nn.layers import Linear
from nn.sequential import Sequential
from nn.losses import MSELoss, CrossEntropyLoss
from nn.optimizers import SGD, Momentum, Adam
from nn.metrics import Accuracy, MeanSquaredError
from nn.trainer import Trainer


def test_mse_loss_forward_backward():
    loss = MSELoss()
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_true = np.array([[0.0, 1.0], [2.0, 3.0]])

    l = loss.forward(y_pred, y_true)
    assert np.isclose(l, 1.0)
    grad = loss.backward()
    assert np.allclose(grad, np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_cross_entropy_loss_with_indices_and_one_hot():
    loss = CrossEntropyLoss()
    y_pred = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 2.0]])
    y_true_idx = np.array([2, 1])
    l1 = loss.forward(y_pred, y_true_idx)
    grad1 = loss.backward()
    assert l1 > 0
    assert grad1.shape == y_pred.shape

    y_true_oh = np.zeros_like(y_pred)
    y_true_oh[np.arange(2), y_true_idx] = 1
    l2 = loss.forward(y_pred, y_true_oh)
    grad2 = loss.backward()
    assert np.isclose(l1, l2)
    assert np.allclose(grad1, grad2)


def test_sgd_momentum_adam_updates():
    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([1.0, 1.0])

    sgd = SGD(lr=0.1)
    sgd.step([p])
    assert np.allclose(p.data, np.array([0.9, 1.9]))
    sgd.zero_grad([p])
    assert np.allclose(p.grad, np.zeros(2))

    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([1.0, 1.0])
    mom = Momentum(lr=0.1, momentum=0.9)
    mom.step([p])
    assert np.allclose(p.data, np.array([0.9, 1.9]))
    p.grad = np.array([1.0, 1.0])
    mom.step([p])
    assert np.allclose(p.data, np.array([0.71, 1.71]), atol=1e-8)

    p = Parameter(np.array([1.0, 2.0]))
    p.grad = np.array([1.0, 1.0])
    adam = Adam(lr=0.1)
    adam.step([p])
    assert p.data.shape == (2,)


def test_metrics_compute():
    acc = Accuracy()
    y_pred = np.array([[0.1, 0.9], [0.7, 0.3]])
    y_true = np.array([1, 0])
    acc.update(y_pred, y_true)
    assert np.isclose(acc.compute(), 1.0)

    mse = MeanSquaredError()
    y_pred = np.array([1.0, 2.0])
    y_true = np.array([1.5, 1.5])
    mse.update(y_pred, y_true)
    assert np.isclose(mse.compute(), 0.25)


def test_trainer_fit_reduces_loss():
    np.random.seed(0)
    model = Sequential([Linear(1, 1)])
    loss = MSELoss()
    opt = SGD(lr=0.1)
    trainer = Trainer(loss, opt)

    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    y = 2 * X

    hist = trainer.fit(model, X, y, epochs=40, batch_size=5)
    assert hist['loss'][-1] < hist['loss'][0]
