import numpy as np


class Trainer:
    def __init__(self, loss_fn, optimizer, metric=None, verbose=False):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric = metric
        self.verbose = verbose

    def fit(self, model, X_train, y_train, epochs=100, batch_size=32):
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train)
        n_samples = X_train.shape[0]

        if n_samples != y_train.shape[0]:
            raise ValueError('X_train and y_train must have same number of samples')

        history = {'loss': [], 'metric': []}

        for epoch in range(1, epochs + 1):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            epoch_loss = 0.0

            if self.metric is not None:
                self.metric.reset()

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_idx = indices[start:end]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                y_pred = model.forward(X_batch)
                loss = self.loss_fn.forward(y_pred, y_batch)
                grad = self.loss_fn.backward()

                model.backward(grad)
                self.optimizer.step(model.parameters())
                self.optimizer.zero_grad(model.parameters())

                epoch_loss += loss * (end - start)

                if self.metric is not None:
                    self.metric.update(y_pred, y_batch)

            epoch_loss /= n_samples
            history['loss'].append(epoch_loss)

            if self.metric is not None:
                history['metric'].append(self.metric.compute())

            if self.verbose:
                msg = f'Epoch {epoch}/{epochs} loss={epoch_loss:.6f}'
                if self.metric is not None:
                    msg += f' metric={history["metric"][-1]:.6f}'
                print(msg)

        return history
