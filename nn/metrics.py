import numpy as np


class Accuracy:
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.total = 0

    def update(self, y_pred, y_true):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        if y_pred.ndim > 1:
            y_pred_label = np.argmax(y_pred, axis=1)
        else:
            y_pred_label = (y_pred >= 0.5).astype(int)

        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_label = np.argmax(y_true, axis=1)
        else:
            y_true_label = y_true.reshape(-1).astype(int)

        if y_pred_label.shape != y_true_label.shape:
            raise ValueError('Shape mismatch for Accuracy.update')

        self.correct += np.sum(y_pred_label == y_true_label)
        self.total += y_true_label.shape[0]

    def compute(self):
        if self.total == 0:
            return 0.0
        return float(self.correct) / self.total


class MeanSquaredError:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sq_sum = 0.0
        self.count = 0

    def update(self, y_pred, y_true):
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        self.sq_sum += np.sum((y_pred - y_true) ** 2)
        self.count += y_pred.size

    def compute(self):
        if self.count == 0:
            return 0.0
        return float(self.sq_sum) / self.count
