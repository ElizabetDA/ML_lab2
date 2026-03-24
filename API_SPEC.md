# API Specification

## Назначение

Этот файл фиксирует единый интерфейс проекта, чтобы все участники команды писали код в одном формате и без расхождений по названиям классов, методов и логике взаимодействия модулей.

Все участники обязаны придерживаться именно этого API.  
Если кто-то хочет поменять интерфейс, это сначала согласуется с тимлидом, а потом меняется у всех.

---

## Структура проекта

```text
nn/
  __init__.py
  base.py
  layers.py
  activations.py
  losses.py
  optimizers.py
  sequential.py
  trainer.py
  data.py
  metrics.py
  utils.py

examples/
  iris_example.py
  mnist_example.py
  regression_example.py

demo/
  app.py
```

---

## Обязательные классы и интерфейсы

### 1. Parameter

Используется для хранения обучаемых параметров модели.

```python
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = None
```

Требования:
- `data` хранит значение параметра
- `grad` хранит градиент параметра
- `grad` должен быть совместим по размеру с `data`

---

### 2. Module

Базовый класс для всех слоёв и контейнеров.

```python
class Module:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []
```

Требования:
- каждый слой наследуется от `Module`
- каждый слой обязан реализовать `forward(x)`
- каждый слой обязан реализовать `backward(grad_output)`
- `parameters()` возвращает список обучаемых параметров слоя
- если у слоя нет параметров, возвращается пустой список

---

### 3. Linear

Полносвязный слой.

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        pass

    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def parameters(self):
        return []
```

Требования:
- хранит веса и смещения
- на `forward` выполняет линейное преобразование
- на `backward` считает градиенты по входу, весам и смещениям

---

### 4. Activation functions

Обязательные классы:
- `ReLU`
- `Sigmoid`
- `Tanh`

У всех должен быть одинаковый интерфейс:

```python
class ReLU(Module):
    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def parameters(self):
        return []
```

Требования:
- активации не имеют обучаемых параметров
- `parameters()` должен возвращать пустой список

---

### 5. Sequential

Контейнер для последовательного перечисления слоёв модели.

```python
class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass

    def parameters(self):
        return []
```

Требования:
- принимает список слоёв
- `forward` последовательно прогоняет вход через все слои
- `backward` идёт по слоям в обратном порядке
- `parameters()` возвращает список параметров всех слоёв модели

---

### 6. Loss functions

Обязательные классы:
- `MSELoss`
- `CrossEntropyLoss`

Интерфейс:

```python
class MSELoss:
    def forward(self, y_pred, y_true):
        pass

    def backward(self):
        pass
```

Требования:
- `forward(y_pred, y_true)` возвращает значение функции потерь
- `backward()` возвращает градиент функции потерь по предсказаниям модели

---

### 7. Optimizers

Обязательные классы:
- `SGD`
- `Momentum`
- `Adam` или `GradientClipping`

Интерфейс:

```python
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params):
        pass

    def zero_grad(self, params):
        pass
```

Требования:
- `step(params)` обновляет параметры модели
- `zero_grad(params)` обнуляет градиенты параметров
- оптимизаторы работают со списком `Parameter`, который приходит из `model.parameters()`

---

### 8. Trainer

```python
class Trainer:
    def fit(self, model, X_train, y_train, epochs=100, batch_size=32):
        pass
```

Требования:
- `fit(...)` запускает цикл обучения
- внутри используется модель, loss, optimizer и данные
- trainer должен поддерживать обучение по эпохам и батчам

---

## Общий цикл обучения

Все участники должны ориентироваться на следующий порядок вызовов:

```python
preds = model.forward(X_batch)
loss = loss_fn.forward(preds, y_batch)
grad = loss_fn.backward()
model.backward(grad)
optimizer.step(model.parameters())
optimizer.zero_grad(model.parameters())
```

Именно под этот сценарий должны стыковаться модули, loss-функции и оптимизаторы.

---

## Правила командной работы

- не менять названия классов и методов без согласования
- не писать альтернативные интерфейсы вроде `run()`, `compute()`, `train_step()` вместо согласованных методов
- если нужен новый метод, сначала согласовать это с тимлидом
- каждый участник отвечает за свои файлы, но должен соблюдать общий API

---

## Итог

Минимально в проекте должны быть реализованы:
- `Parameter`
- `Module`
- `Linear`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `MSELoss`
- `CrossEntropyLoss`
- `SGD`
- `Momentum`
- `Adam` или `GradientClipping`
- `Sequential`
- `Trainer.fit`
- `Dataset`
- `DataLoader`
- примеры на `Iris`, `MNIST/Fashion-MNIST`, синтетической регрессии
- wow-часть через `streamlit` или другую визуальную демонстрацию