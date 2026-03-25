import numpy as np


class MSELoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        self.y_pred = np.asarray(y_pred, dtype=np.float64)
        self.y_true = np.asarray(y_true, dtype=np.float64)
        if self.y_pred.shape != self.y_true.shape:
            raise ValueError('y_pred and y_true must have same shape for MSELoss')
        loss = np.mean((self.y_pred - self.y_true) ** 2)
        return loss

    def backward(self):
        if self.y_pred is None or self.y_true is None:
            raise RuntimeError('forward must be called before backward')
        n = self.y_true.size
        return 2.0 * (self.y_pred - self.y_true) / n


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def _softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true)

        if y_pred.ndim != 2:
            raise ValueError('y_pred must be a 2D array for CrossEntropyLoss')

        self.probs = self._softmax(y_pred)

        if y_true.ndim == 1:
            if y_true.shape[0] != y_pred.shape[0]:
                raise ValueError('y_true shape mismatch')
            self.y_true = np.zeros_like(self.probs)
            self.y_true[np.arange(y_pred.shape[0]), y_true] = 1.0
        elif y_true.ndim == 2:
            if y_true.shape != y_pred.shape:
                raise ValueError('y_true and y_pred must have same shape for one-hot targets')
            self.y_true = y_true.astype(np.float64)
        else:
            raise ValueError('y_true must be 1D indices or 2D one-hot vectors')

        clip_eps = 1e-15
        log_probs = np.log(np.clip(self.probs, clip_eps, 1.0))
        loss = -np.sum(self.y_true * log_probs) / y_pred.shape[0]
        return loss

    def backward(self):
        if self.probs is None or self.y_true is None:
            raise RuntimeError('forward must be called before backward')
        batch_size = self.probs.shape[0]
        return (self.probs - self.y_true) / batch_size
