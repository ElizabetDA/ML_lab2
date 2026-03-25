import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    X = np.asarray(X)
    y = np.asarray(y)

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]

    split_idx = int(len(X) * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def normalize_features(X_train, X_test=None):
    X_train = np.asarray(X_train, dtype=np.float64)
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    X_train_norm = (X_train - mean) / std

    if X_test is None:
        return X_train_norm

    X_test = np.asarray(X_test, dtype=np.float64)
    X_test_norm = (X_test - mean) / std
    return X_train_norm, X_test_norm


def batch_iterator(X, y, batch_size=32, shuffle=True):
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.arange(len(X))

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


def shuffle_data(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]
