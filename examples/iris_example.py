import os
import sys

import numpy as np
from sklearn.datasets import load_iris

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nn.activations import ReLU
from nn.layers import Linear
from nn.losses import CrossEntropyLoss
from nn.metrics import Accuracy
from nn.optimizers import Adam
from nn.sequential import Sequential
from nn.trainer import Trainer
from nn.data import train_test_split, normalize_features


def evaluate(model, X, y):
    metric = Accuracy()
    preds = model.forward(X)
    metric.update(preds, y)
    return metric.compute()


def main():
    print("start iris example")
    np.random.seed(42)

    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    X_train, X_test = normalize_features(X_train, X_test)

    model = Sequential([
        Linear(4, 16),
        ReLU(),
        Linear(16, 3),
    ])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.01)
    metric = Accuracy()
    trainer = Trainer(loss_fn=loss_fn, optimizer=optimizer, metric=metric, verbose=True)

    history = trainer.fit(
        model,
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
    )

    train_acc = evaluate(model, X_train, y_train)
    test_acc = evaluate(model, X_test, y_test)

    print("\n=== Training finished ===")
    print(f"Final train loss: {history['loss'][-1]:.6f}")
    if history["metric"]:
        print(f"Final train metric: {history['metric'][-1]:.6f}")
    print(f"Train accuracy: {train_acc:.6f}")
    print(f"Test accuracy: {test_acc:.6f}")


if __name__ == "__main__":
    main()
