import numpy as np
import matplotlib.pyplot as plt
from nn.sequential import Sequential
from nn.layers import Linear
from nn.activations import Tanh
from nn.losses import MSELoss
from nn.metrics import MeanSquaredError
from nn.optimizers import Adam
from nn.trainer import Trainer
from nn.data import train_test_split, normalize_features


random_seed = 42


np.random.seed(random_seed)
n_samples = 300
X = np.random.uniform(-5, 5, size=(n_samples, 1))
y = 2 * X + 3 + np.random.normal(0, 0.8, size=(n_samples, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

model = Sequential([
    Linear(1, 16),
    Tanh(),
    Linear(16, 1)
])

loss = MSELoss()
optimizer = Adam(lr=0.01)
metric = MeanSquaredError()
trainer = Trainer(loss_fn=loss, optimizer=optimizer, metric=metric, verbose=True)

print("=" * 50)
print("Обучение модели регрессии (y = 2x + 3 + noise)")
print("=" * 50)

history = trainer.fit(
    model=model,
    X_train=X_train_norm,
    y_train=y_train,
    epochs=40,
    batch_size=32
)

y_pred = model.forward(X_test_norm)

test_loss = loss.forward(y_pred, y_test)

metric.reset()
metric.update(y_pred, y_test)
test_metric = metric.compute()

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ")
print("=" * 50)
print(f"Финальная ошибка на обучении (loss):     {history['loss'][-1]:.6f}")
if history['metric']:
    print(f"Финальная метрика на обучении (MSE):    {history['metric'][-1]:.6f}")
print(f"Ошибка на тестовой выборке (loss):        {test_loss:.6f}")
print(f"Метрика на тестовой выборке (MSE):        {test_metric:.6f}")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

X_test_orig = X_test
axes[0].scatter(X_test_orig, y_test, alpha=0.5, label='Истинные значения')
axes[0].scatter(X_test_orig, y_pred, alpha=0.5, label='Предсказания', marker='x')
axes[0].plot(X_test_orig, 2 * X_test_orig + 3, alpha=0.7, color="black", label='Истинная зависимость (без шума)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Предсказания модели')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['loss'], label='Train loss', color='blue')
if history['metric']:
    axes[1].plot(history['metric'], "--", label='Train MSE', color='orange')
axes[1].set_xlabel('Эпоха')
axes[1].set_ylabel('Значение')
axes[1].set_title('Кривые обучения')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_result.png', dpi=150)
plt.show()