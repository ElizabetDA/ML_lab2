import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params):
        for p in params:
            if p.grad is None:
                continue
            p.data = p.data - self.lr * p.grad

    def zero_grad(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.fill(0.0)


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, params):
        for p in params:
            if p.grad is None:
                continue
            key = id(p)
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(p.data)
            v = self.velocities[key]
            v = self.momentum * v - self.lr * p.grad
            p.data = p.data + v
            self.velocities[key] = v

    def zero_grad(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.fill(0.0)


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params):
        self.t += 1
        for p in params:
            if p.grad is None:
                continue
            key = id(p)
            if key not in self.m:
                self.m[key] = np.zeros_like(p.data)
                self.v[key] = np.zeros_like(p.data)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * p.grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (p.grad ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            p.data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.fill(0.0)
