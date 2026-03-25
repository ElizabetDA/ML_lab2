from nn.base import Module
from nn.base import Parameter

import numpy as np


class Linear(Module):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight = Parameter(np.random.randn(in_features, out_features) * scale)
        self.bias = Parameter(np.zeros((1, out_features)))
        self._last_input = None

    def forward(self, x):
        self._last_input = x
        return x @ self.weight.data + self.bias.data

    def backward(self, grad_output):
        if self._last_input is None:
            raise RuntimeError("forward must be called before backward")
        self.weight.grad = self._last_input.T @ grad_output
        self.bias.grad = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = grad_output @ self.weight.data.T
        return grad_input

    def parameters(self):
        return [self.weight, self.bias]
