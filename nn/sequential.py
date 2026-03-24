from nn.base import Module

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []