import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from nn.sequential import Sequential
from nn.layers import Linear
from nn.activations import ReLU
from nn.losses import CrossEntropyLoss
from nn.metrics import Accuracy
from nn.optimizers import Adam
from nn.trainer import Trainer
from nn.data import train_test_split, normalize_features, one_hot_conversion
from sklearn.metrics import confusion_matrix
import seaborn as sns


random_seed = 42
epochs = 20
batch_size = 64



def main(with_graphic=True):
    digits = load_digits()
    X = digits.data
    y = digits.target

    y_onehot = one_hot_conversion(10, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.2, random_state=random_seed, shuffle=True
    )

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)

    model = Sequential([
        Linear(64, 32),
        ReLU(),
        Linear(32, 10)
    ])

    loss = CrossEntropyLoss()
    optimizer = Adam()
    metric = Accuracy()
    trainer = Trainer(loss_fn=loss, optimizer=optimizer, metric=metric, verbose=True)

    history = trainer.fit(
        model=model,
        X_train=X_train_norm,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size,
    )

    logits = model.forward(X_test_norm)

    test_loss = loss.forward(logits, y_test)

    metric.reset()

    metric.update(logits, y_test)
    test_accuracy = metric.compute()

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Final training loss:     {history['loss'][-1]:.6f}")
    if history['metric']:
        print(f"Final training accuracy:   {history['metric'][-1]:.4f} ({history['metric'][-1] * 100:.2f}%)")
    print(f"Test loss:        {test_loss:.6f}")
    print(f"Test accuracy:       {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

    random_accuracy = 0.1
    if test_accuracy > random_accuracy:
        print(
            f"\nGood result: accuracy {test_accuracy * 100:.2f}% > {random_accuracy * 100:.0f}% (random guessing)")
    else:
        print(f"\nAccuracy below random guessing, needs improvement")

    print("=" * 50)

    if with_graphic:

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].plot(history['loss'], label='Train loss', color='blue', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('CrossEntropy Loss')
        axes[0].set_title('Learning Curve (loss)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if history['metric']:
            axes[1].plot(history['metric'], label='Train accuracy', color='green', linewidth=2)
            axes[1].axhline(y=random_accuracy, color='red', linestyle='--',
                            label=f'Random guess ({random_accuracy * 100:.0f}%)')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Learning Curve (accuracy)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)


        all_logits = model.forward(X_test_norm)
        all_pred_classes = np.argmax(all_logits, axis=1)
        all_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(all_true_classes, all_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], cbar=False)
        axes[2].set_xlabel('Predicted class')
        axes[2].set_ylabel('True class')
        axes[2].set_title('Confusion Matrix')

        fig.suptitle(
            f"Classification Model Analysis: epochs - {epochs}, batch size - {batch_size}",
            fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('../results/mnist_result.png', dpi=150)
        plt.show()

if __name__ == "__main__":
    main()