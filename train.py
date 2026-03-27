"""MNIST digit classification — baseline model.

Trains a simple feedforward network on MNIST and reports validation loss.
Contributors: improve val_loss by changing architecture, hyperparameters,
training procedure, or anything else. The only rule is that train.py must
run end-to-end and write results to results.tsv.
"""

import csv
import os
import struct
import gzip
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data loading (downloads MNIST if not cached)
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")

URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def download(name: str, url: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
    return path


def load_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols) / 255.0


def load_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def one_hot(labels: np.ndarray, n_classes: int = 10) -> np.ndarray:
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1
    return out


# ---------------------------------------------------------------------------
# Model — two hidden layers, ReLU, softmax cross-entropy, momentum SGD
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(pred, target):
    return -np.mean(np.sum(target * np.log(pred + 1e-8), axis=1))


class MLP:
    def __init__(self, input_dim=784, hidden1=256, hidden2=128, output_dim=10):
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(output_dim)

        # Momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.out = softmax(self.z3)
        return self.out

    def backward(self, X, y_onehot, lr=0.01, momentum=0.9):
        m = X.shape[0]
        dz3 = (self.out - y_onehot) / m
        dW3 = self.a2.T @ dz3
        db3 = dz3.sum(axis=0)
        da2 = dz3 @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        for param, grad, vel in [
            ('W3', dW3, 'vW3'), ('b3', db3, 'vb3'),
            ('W2', dW2, 'vW2'), ('b2', db2, 'vb2'),
            ('W1', dW1, 'vW1'), ('b1', db1, 'vb1'),
        ]:
            v = getattr(self, vel)
            v[:] = momentum * v - lr * grad
            getattr(self, param).__iadd__(v)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    np.random.seed(42)

    # Load data
    X_train = load_images(download("train_images", URLS["train_images"]))
    y_train = load_labels(download("train_labels", URLS["train_labels"]))
    X_test = load_images(download("test_images", URLS["test_images"]))
    y_test = load_labels(download("test_labels", URLS["test_labels"]))

    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    # Hyperparameters
    epochs = 30
    batch_size = 128
    lr_init = 0.1
    hidden1 = 256
    hidden2 = 128

    model = MLP(hidden1=hidden1, hidden2=hidden2)

    results = []
    best_loss = float('inf')
    best_weights = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # LR decay
        lr = lr_init * (0.95 ** epoch)

        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_train, y_train_oh = X_train[idx], y_train_oh[idx]

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train_oh[i:i+batch_size]
            model.forward(X_batch)
            model.backward(X_batch, y_batch, lr=lr, momentum=0.9)

        # Evaluate
        val_pred = model.forward(X_test)
        val_loss = cross_entropy(val_pred, y_test_oh)
        val_acc = (val_pred.argmax(axis=1) == y_test).mean()

        print(f"Epoch {epoch+1:2d}/{epochs}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        results.append({"epoch": epoch + 1, "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 4)})

        # Early stopping with best model checkpoint
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.copy() for k, v in model.__dict__.items() if isinstance(v, np.ndarray)}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best weights and re-evaluate for final result
    if best_weights:
        for k, v in best_weights.items():
            setattr(model, k, v)
        val_pred = model.forward(X_test)
        final_loss = cross_entropy(val_pred, y_test_oh)
        final_acc = (val_pred.argmax(axis=1) == y_test).mean()
        results.append({"epoch": epoch + 2, "val_loss": round(final_loss, 6), "val_acc": round(final_acc, 4)})

    # Write results
    with open("results.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "val_loss", "val_acc"], delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFinal val_loss: {results[-1]['val_loss']}")
    print(f"Results written to results.tsv")
    return results[-1]["val_loss"]


if __name__ == "__main__":
    train()
