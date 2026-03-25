import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)


class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)
