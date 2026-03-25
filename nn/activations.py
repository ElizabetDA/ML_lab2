from nn.base import Module

import numpy as np


class ReLU(Module):
    def __init__(self):
        self._mask = None

    def forward(self, x):
        self._mask = x > 0
        return np.maximum(0.0, x)

    def backward(self, grad_output):
        if self._mask is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self._mask

    def parameters(self):
        return []


class Sigmoid(Module):
    def __init__(self):
        self._output = None

    def forward(self, x):
        self._output = 1.0 / (1.0 + np.exp(-x))
        return self._output

    def backward(self, grad_output):
        if self._output is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * self._output * (1.0 - self._output)

    def parameters(self):
        return []


class Tanh(Module):
    def __init__(self):
        self._output = None

    def forward(self, x):
        self._output = np.tanh(x)
        return self._output

    def backward(self, grad_output):
        if self._output is None:
            raise RuntimeError("forward must be called before backward")
        return grad_output * (1.0 - self._output**2)

    def parameters(self):
        return []
