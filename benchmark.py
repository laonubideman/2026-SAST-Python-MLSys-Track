"""Compare the NumPy, C++, and CUDA softmax-regression backends."""

import argparse
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simple_ml import loss_err, parse_mnist, softmax_regression_epoch  # noqa: E402


def available_backends():
    backends = [("NumPy", softmax_regression_epoch)]

    try:
        from simple_ml_ext import softmax_regression_epoch_cpp

        backends.append(("C++", softmax_regression_epoch_cpp))
    except ImportError:
        pass

    try:
        from simple_ml_cuda import (
            cuda_available,
            softmax_regression_epoch_cuda,
        )

        if cuda_available():
            backends.append(("CUDA", softmax_regression_epoch_cuda))
    except ImportError:
        pass

    return backends


def benchmark_backend(update, X, y, epochs, repeats, lr, batch):
    durations = []
    final_theta = None

    for _ in range(repeats):
        theta = np.zeros((X.shape[1], int(y.max()) + 1), dtype=np.float32)
        start = time.perf_counter()
        for _ in range(epochs):
            update(X, y, theta, lr=lr, batch=batch)
        durations.append(time.perf_counter() - start)
        final_theta = theta

    return median(durations), final_theta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.2)
    args = parser.parse_args()

    if args.samples <= 0 or args.epochs <= 0 or args.repeats <= 0:
        parser.error("samples, epochs, and repeats must be positive")
    if args.batch <= 0:
        parser.error("batch must be positive")

    X_train, y_train = parse_mnist(
        PROJECT_ROOT / "data/train-images-idx3-ubyte.gz",
        PROJECT_ROOT / "data/train-labels-idx1-ubyte.gz",
    )
    X_test, y_test = parse_mnist(
        PROJECT_ROOT / "data/t10k-images-idx3-ubyte.gz",
        PROJECT_ROOT / "data/t10k-labels-idx1-ubyte.gz",
    )
    sample_count = min(args.samples, X_train.shape[0])
    X_train = np.ascontiguousarray(X_train[:sample_count])
    y_train = np.ascontiguousarray(y_train[:sample_count])

    results = []
    reference_theta = None
    numpy_duration = None

    for name, update in available_backends():
        duration, theta = benchmark_backend(
            update,
            X_train,
            y_train,
            args.epochs,
            args.repeats,
            args.lr,
            args.batch,
        )
        train_loss, train_error = loss_err(X_train @ theta, y_train)
        test_loss, test_error = loss_err(X_test @ theta, y_test)

        if reference_theta is None:
            reference_theta = theta
            numpy_duration = duration
        max_difference = float(np.max(np.abs(theta - reference_theta)))
        results.append(
            (
                name,
                duration,
                numpy_duration / duration,
                train_loss,
                train_error,
                test_loss,
                test_error,
                max_difference,
            )
        )

    print(
        f"samples={sample_count}, epochs={args.epochs}, "
        f"batch={args.batch}, median of {args.repeats} runs"
    )
    print(
        "| Backend | Time (s) | vs NumPy | Train loss | Train err | "
        "Test loss | Test err | max |delta theta| |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        name, duration, speedup, tr_loss, tr_err, te_loss, te_err, difference = result
        print(
            f"| {name} | {duration:.4f} | {speedup:.2f}x | "
            f"{tr_loss:.5f} | {tr_err:.5f} | {te_loss:.5f} | "
            f"{te_err:.5f} | {difference:.3e} |"
        )

    if len(results) == 1:
        print("\nC++ is not built and no CUDA device is available; only NumPy ran.")


if __name__ == "__main__":
    main()
