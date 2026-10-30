"""Train the from-scratch models and render reproducible showcase figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from simple_ml import loss_err, nn_epoch, parse_mnist, softmax_regression_epoch  # noqa: E402


COLORS = {
    "softmax": "#2563eb",
    "network": "#e11d48",
    "train": "#0891b2",
    "test": "#f97316",
}


def evaluate(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    loss, error = loss_err(logits, labels)
    return float(loss), float(error)


def train_softmax(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    theta = np.zeros(
        (X_train.shape[1], int(y_train.max()) + 1), dtype=np.float32
    )
    history = []

    for epoch in range(epochs + 1):
        train_loss, train_error = evaluate(X_train @ theta, y_train)
        test_loss, test_error = evaluate(X_test @ theta, y_test)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_error": train_error,
                "test_error": test_error,
            }
        )
        if epoch < epochs:
            softmax_regression_epoch(
                X_train,
                y_train,
                theta,
                lr=learning_rate,
                batch=batch_size,
            )

    return theta, history


def train_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    rng = np.random.RandomState(0)
    class_count = int(y_train.max()) + 1
    W1 = (
        rng.randn(X_train.shape[1], hidden_dim).astype(np.float32)
        / np.sqrt(hidden_dim)
    )
    W2 = (
        rng.randn(hidden_dim, class_count).astype(np.float32)
        / np.sqrt(class_count)
    )
    history = []

    for epoch in range(epochs + 1):
        train_logits = np.maximum(X_train @ W1, 0) @ W2
        test_logits = np.maximum(X_test @ W1, 0) @ W2
        train_loss, train_error = evaluate(train_logits, y_train)
        test_loss, test_error = evaluate(test_logits, y_test)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_error": train_error,
                "test_error": test_error,
            }
        )
        if epoch < epochs:
            nn_epoch(
                X_train,
                y_train,
                W1,
                W2,
                lr=learning_rate,
                batch=batch_size,
            )

    return W1, W2, history


def plot_convergence(
    softmax_history: list[dict[str, float]],
    network_history: list[dict[str, float]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for history, label, color in (
        (softmax_history, "Softmax regression", COLORS["softmax"]),
        (network_history, "Two-layer ReLU network", COLORS["network"]),
    ):
        epochs = [item["epoch"] for item in history]
        axes[0].plot(
            epochs,
            [item["test_loss"] for item in history],
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
            color=color,
        )
        axes[1].plot(
            epochs,
            [100 * item["test_error"] for item in history],
            marker="o",
            markersize=3,
            linewidth=2,
            label=label,
            color=color,
        )

    axes[0].set_title("Test cross-entropy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Test classification error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error (%)")

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
        axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Learning from raw MNIST bytes without an ML framework", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def plot_confusion(matrix: np.ndarray, output: Path) -> None:
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = matrix / np.maximum(row_totals, 1)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_title("Two-layer network: normalized confusion matrix")
    axis.set_xlabel("Predicted digit")
    axis.set_ylabel("True digit")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))

    for row in range(10):
        for column in range(10):
            value = normalized[row, column]
            if value >= 0.01:
                axis.text(
                    column,
                    row,
                    f"{100 * value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 else "#0f172a",
                )

    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of true class")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mistakes(
    images: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    output: Path,
    count: int = 15,
) -> None:
    mistake_indices = np.flatnonzero(predictions != labels)[:count]
    columns = 5
    rows = int(np.ceil(len(mistake_indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 2.25 * rows))
    axes = np.asarray(axes).reshape(-1)

    for axis, index in zip(axes, mistake_indices):
        axis.imshow(images[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"true {labels[index]}  /  predicted {predictions[index]}")
        axis.axis("off")

    for axis in axes[len(mistake_indices) :]:
        axis.axis("off")

    fig.suptitle("Representative mistakes reveal ambiguous handwriting", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=60_000)
    parser.add_argument("--test-samples", type=int, default=10_000)
    parser.add_argument("--softmax-epochs", type=int, default=10)
    parser.add_argument("--network-epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "assets" / "showcase")
    args = parser.parse_args()

    positive_fields = (
        "train_samples",
        "test_samples",
        "softmax_epochs",
        "network_epochs",
        "hidden_dim",
        "batch",
    )
    if any(getattr(args, field) <= 0 for field in positive_fields):
        parser.error("sample counts, epochs, hidden_dim, and batch must be positive")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    X_train, y_train = parse_mnist(
        ROOT / "data" / "train-images-idx3-ubyte.gz",
        ROOT / "data" / "train-labels-idx1-ubyte.gz",
    )
    X_test, y_test = parse_mnist(
        ROOT / "data" / "t10k-images-idx3-ubyte.gz",
        ROOT / "data" / "t10k-labels-idx1-ubyte.gz",
    )
    X_train = X_train[: args.train_samples]
    y_train = y_train[: args.train_samples]
    X_test = X_test[: args.test_samples]
    y_test = y_test[: args.test_samples]

    _, softmax_history = train_softmax(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.softmax_epochs,
        learning_rate=0.2,
        batch_size=args.batch,
    )
    W1, W2, network_history = train_network(
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_dim=args.hidden_dim,
        epochs=args.network_epochs,
        learning_rate=0.2,
        batch_size=args.batch,
    )

    network_predictions = (np.maximum(X_test @ W1, 0) @ W2).argmax(axis=1)
    matrix = confusion_matrix(y_test, network_predictions)

    plot_convergence(
        softmax_history,
        network_history,
        args.output / "convergence.png",
    )
    plot_confusion(matrix, args.output / "confusion_matrix.png")
    plot_mistakes(
        X_test,
        y_test,
        network_predictions,
        args.output / "mistakes.png",
    )

    payload = {
        "config": {
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "softmax_epochs": args.softmax_epochs,
            "network_epochs": args.network_epochs,
            "hidden_dim": args.hidden_dim,
            "batch": args.batch,
            "softmax_learning_rate": 0.2,
            "network_learning_rate": 0.2,
            "seed": 0,
        },
        "softmax": {
            "history": softmax_history,
            "final": softmax_history[-1],
        },
        "two_layer_network": {
            "history": network_history,
            "final": network_history[-1],
            "confusion_matrix": matrix.tolist(),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print("| Model | Test loss | Test error |")
    print("|---|---:|---:|")
    for name, result in (
        ("Softmax regression", softmax_history[-1]),
        ("Two-layer ReLU network", network_history[-1]),
    ):
        print(
            f"| {name} | {result['test_loss']:.5f} | "
            f"{100 * result['test_error']:.2f}% |"
        )
    print(f"\nWrote reproducible artifacts to {args.output}")


if __name__ == "__main__":
    main()
