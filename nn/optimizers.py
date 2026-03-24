class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params):
        raise NotImplementedError

    def zero_grad(self, params):
        raise NotImplementedError


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def step(self, params):
        raise NotImplementedError

    def zero_grad(self, params):
        raise NotImplementedError


class Adam:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params):
        raise NotImplementedError

    def zero_grad(self, params):
        raise NotImplementedError