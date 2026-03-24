import numpy as np

from nn.activations import ReLU, Sigmoid, Tanh
from nn.layers import Linear
from nn.sequential import Sequential


def test_igor_block_forward_backward_and_parameters():
    np.random.seed(0)
    x = np.random.randn(5, 4)

    model = Sequential(
        [
            Linear(4, 8),
            ReLU(),
            Linear(8, 3),
            Sigmoid(),
        ]
    )

    out = model.forward(x)
    assert out.shape == (5, 3)

    grad_in = model.backward(np.ones_like(out))
    assert grad_in.shape == x.shape

    params = model.parameters()
    assert len(params) == 4
    for p in params:
        assert p.grad is not None
        assert p.grad.shape == p.data.shape


def test_tanh_shapes():
    act = Tanh()
    x = np.random.randn(2, 2)

    y = act.forward(x)
    grad = act.backward(np.ones_like(y))

    assert y.shape == (2, 2)
    assert grad.shape == (2, 2)
