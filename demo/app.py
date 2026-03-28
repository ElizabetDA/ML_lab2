import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from nn.sequential import Sequential
from nn.layers import Linear
from nn.activations import ReLU, Tanh, Sigmoid
from nn.losses import MSELoss, CrossEntropyLoss
from nn.metrics import MeanSquaredError, Accuracy
from nn.optimizers import SGD, Momentum, Adam
from nn.trainer import Trainer
from nn.data import train_test_split, normalize_features
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

st.set_page_config(
    page_title="MindLab Demo",
    layout="wide"
)

st.title("MindLab — Нейросетевой фреймворк на NumPy")

st.sidebar.header("Настройки эксперимента")

task_type = st.sidebar.selectbox(
    "Тип задачи",
    ["Регрессия", "Классификация"]
)

if task_type == "Регрессия":
    st.sidebar.markdown("### Параметры регрессии")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        a_coef = st.number_input("Коэффициент a", value=2.0, step=0.5, format="%.1f")
    with col_b:
        b_coef = st.number_input("Коэффициент b", value=3.0, step=0.5, format="%.1f")

    noise_level = st.sidebar.slider(
        "Уровень шума",
        min_value=0.0,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Стандартное отклонение нормального шума"
    )

    n_samples = st.sidebar.slider(
        "Количество примеров",
        min_value=100,
        max_value=1000,
        value=500,
        step=50
    )

else:
    dataset = st.sidebar.selectbox(
        "Датасет",
        [
            "digits (рукописные цифры, 8x8)",
            "iris (цветки ириса, 3 класса)",
            "wine (вина, 3 класса)",
            "breast_cancer (рак груди, 2 класса)"
        ]
    )

optimizer_name = st.sidebar.selectbox(
    "Оптимизатор",
    ["Adam", "SGD", "Momentum"]
)

st.sidebar.markdown("### Параметры оптимизатора")

if optimizer_name == "Adam":
    opt_col1, opt_col2 = st.sidebar.columns(2)

    with opt_col1:
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%.4f"
        )
        beta1 = st.number_input(
            "β₁ (momentum)",
            min_value=0.8,
            max_value=0.999,
            value=0.9,
            format="%.3f"
        )

    with opt_col2:
        beta2 = st.number_input(
            "β₂ (RMSprop)",
            min_value=0.9,
            max_value=0.9999,
            value=0.999,
            format="%.4f"
        )
        eps = st.number_input(
            "ε (стабильность)",
            min_value=1e-10,
            max_value=1e-6,
            value=1e-8,
            format="%.1e"
        )

elif optimizer_name == "SGD":
    learning_rate = st.sidebar.number_input(
        "Learning rate",
        min_value=0.0001,
        max_value=0.5,
        value=0.01,
        format="%.4f"
    )
else:
    opt_col1, opt_col2 = st.sidebar.columns(2)
    with opt_col1:
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.0001,
            max_value=0.5,
            value=0.01,
            format="%.4f"
        )
    with opt_col2:
        momentum = st.number_input(
            "Momentum",
            min_value=0.5,
            max_value=0.99,
            value=0.9,
            format="%.2f"
        )

st.sidebar.markdown("### Параметры обучения")
col1, col2 = st.sidebar.columns(2)
with col1:
    epochs = st.number_input("Эпохи", min_value=10, max_value=500, value=100, step=10)
    random_seed = st.number_input("Random seed", min_value=0, max_value=999, value=42)
with col2:
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=1)

train_button = st.sidebar.button("Запустить обучение", type="primary", use_container_width=True)

if not train_button:
    st.info("Настройте параметры в боковой панели и нажмите 'Запустить обучение'")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Регрессия")
        st.code("""
        model = Sequential([
            Linear(1, 32),
            Tanh(),
            Linear(32, 32),
            Tanh(),
            Linear(32, 1)
        ])
        """, language="python")

    with col2:
        st.subheader("Классификация")
        st.code("""
        model = Sequential([
            Linear(n_features, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, n_classes)
        ])
        """, language="python")

    st.markdown("---")
    st.caption(
        """**О фреймворке:**
- Все слои реализованы на NumPy
- Поддержка обратного распространения
- Оптимизаторы: SGD, Momentum, Adam
- Loss: MSELoss, CrossEntropyLoss"""
    )

else:
    st.subheader("Процесс обучения")
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Подготовка данных...")

    np.random.seed(random_seed)

    if task_type == "Регрессия":
        X = np.random.uniform(-5, 5, size=(n_samples, 1))
        y = a_coef * X + b_coef + np.random.normal(0, noise_level, size=(n_samples, 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed
        )

        X_train_norm, X_test_norm = normalize_features(X_train, X_test)

        model = Sequential([
            Linear(1, 32),
            Tanh(),
            Linear(32, 32),
            Tanh(),
            Linear(32, 1)
        ])

        loss_fn = MSELoss()
        metric = MeanSquaredError()
        is_classification = False

        st.info(f"**Регрессия:** y = {a_coef:.1f} * x + {b_coef:.1f} + шум(σ={noise_level:.1f}), {n_samples} примеров")

    else:
        if dataset == "digits (рукописные цифры, 8x8)":
            data = load_digits()
            X = data.data
            y = data.target
            n_features = 64
            n_classes = 10
            dataset_info = f"digits: {X.shape[0]} примеров, {n_features} признаков (8x8), {n_classes} классов"

        elif dataset == "iris (цветки ириса, 3 класса)":
            data = load_iris()
            X = data.data
            y = data.target
            n_features = 4
            n_classes = 3
            dataset_info = f"iris: {X.shape[0]} примеров, {n_features} признака, {n_classes} класса"

        elif dataset == "wine (вина, 3 класса)":
            data = load_wine()
            X = data.data
            y = data.target
            n_features = 13
            n_classes = 3
            dataset_info = f"wine: {X.shape[0]} примеров, {n_features} признаков, {n_classes} класса"

        else:
            data = load_breast_cancer()
            X = data.data
            y = data.target
            n_features = 30
            n_classes = 2
            dataset_info = f"breast_cancer: {X.shape[0]} примеров, {n_features} признаков, {n_classes} класса"

        y_onehot = np.eye(n_classes)[y]

        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            X, y_onehot, test_size=0.2, random_state=random_seed, stratify=y
        )

        X_train_norm, X_test_norm = normalize_features(X_train, X_test)

        if n_features <= 10:
            model = Sequential([
                Linear(n_features, 32),
                ReLU(),
                Linear(32, 16),
                ReLU(),
                Linear(16, n_classes)
            ])
        else:
            model = Sequential([
                Linear(n_features, 64),
                ReLU(),
                Linear(64, 32),
                ReLU(),
                Linear(32, n_classes)
            ])
        loss_fn = CrossEntropyLoss()
        metric = Accuracy()
        is_classification = True

        st.info(f"**Классификация:** {dataset_info}")

    if optimizer_name == "Adam":
        optimizer = Adam(lr=learning_rate, beta1=beta1, beta2=beta2, eps=eps)
        optimizer_params = f"β₁={beta1}, β₂={beta2}"
    elif optimizer_name == "SGD":
        optimizer = SGD(lr=learning_rate)
        optimizer_params = "без момента"
    else:
        optimizer = Momentum(lr=learning_rate, momentum=momentum)
        optimizer_params = f"momentum={momentum}"

    trainer = Trainer(
        loss_fn=loss_fn,
        optimizer=optimizer,
        metric=metric,
        verbose=False
    )

    status_text.text("Обучение началось...")

    history = trainer.fit(
        model=model,
        X_train=X_train_norm,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size
    )

    progress_bar.progress(100)
    status_text.text("Обучение завершено!")

    y_pred_test = model.forward(X_test_norm)
    test_loss = loss_fn.forward(y_pred_test, y_test)

    metric.reset()
    metric.update(y_pred_test, y_test)
    test_metric = metric.compute()

    st.markdown("---")
    st.subheader("Результаты обучения")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric(
            label="Финальный loss (train)",
            value=f"{history['loss'][-1]:.6f}",
            delta_color="normal"
        )

    with metric_col2:
        if is_classification:
            st.metric(
                label="Точность на тесте (accuracy)",
                value=f"{test_metric:.4f} ({test_metric * 100:.1f}%)",
                delta_color="normal"
            )
        else:
            st.metric(
                label="MSE на тесте",
                value=f"{test_metric:.6f}",
                delta_color="normal"
            )

    with metric_col3:
        st.metric(
            label="Оптимизатор",
            value=optimizer_name,
            delta_color="off"
        )

    st.markdown("---")
    st.subheader("Графическое представление")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history['loss'], label='Train loss', color='blue', linewidth=2)
        ax.set_xlabel('Эпоха')
        ax.set_ylabel('Loss')
        ax.set_title('Кривая обучения (loss)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        if history['metric']:
            ax.plot(history['metric'], label='Train metric', color='green', linewidth=2)
            if is_classification:
                random_guess = 1.0 / n_classes if 'n_classes' in locals() else 0.1
                ax.axhline(y=random_guess, color='red', linestyle='--',
                           label=f'Случайное угадывание ({random_guess * 100:.1f}%)')
                ax.set_ylabel('Accuracy')
            else:
                ax.set_ylabel('MSE')
            ax.set_xlabel('Эпоха')
            ax.set_title('Кривая обучения (метрика)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Метрика не выбрана', ha='center', va='center')
            ax.set_title('Кривая обучения (метрика)')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    if is_classification:
        all_logits = model.forward(X_test_norm)
        all_pred = np.argmax(all_logits, axis=1)
        all_true = np.argmax(y_test, axis=1)

        st.subheader("Матрица ошибок")

        cm = confusion_matrix(all_true, all_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Предсказанный класс')
        ax.set_ylabel('Истинный класс')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.subheader("Детальная статистика")

        class_accuracy = []
        for i in range(n_classes):
            mask = (all_true == i)
            if np.sum(mask) > 0:
                acc = np.sum(all_pred[mask] == i) / np.sum(mask)
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)

        class_names = [f"Класс {i}" for i in range(n_classes)]
        accuracy_df = pd.DataFrame({
            'Класс': class_names,
            'Точность': [f"{acc * 100:.1f}%" for acc in class_accuracy],
            'Количество примеров': [np.sum(all_true == i) for i in range(n_classes)]
        })
        st.dataframe(accuracy_df, use_container_width=True)

    else:
        st.subheader("Предсказания vs Истинные значения")

        X_test_orig = X_test
        y_pred_plot = model.forward(X_test_norm)

        X_range = np.linspace(-5, 5, 200).reshape(-1, 1)
        X_range_norm = (X_range - X_train.mean()) / X_train.std()
        y_range_pred = model.forward(X_range_norm)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(X_test_orig, y_test, alpha=0.5, label='Истинные значения (тест)', s=30, color='blue')
        ax.scatter(X_test_orig, y_pred_plot, alpha=0.5, label='Предсказания', marker='x', s=50, color='red')

        X_smooth = np.linspace(-5, 5, 200)
        y_true_smooth = a_coef * X_smooth + b_coef
        ax.plot(X_smooth, y_true_smooth, 'g--', linewidth=2, alpha=0.7,
                label=f'Истинная: y = {a_coef:.1f}x + {b_coef:.1f}')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Регрессия: предсказания модели (MSE = {test_metric:.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
