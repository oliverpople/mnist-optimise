# MNIST Optimise

Baseline MNIST digit classifier using a numpy-only MLP. No frameworks — just matrix math.

**Current val_loss: 0.0648** (single hidden layer, 128 units, 20 epochs)

## Run

```bash
pip install -r requirements.txt
python train.py
```

Results are written to `results.tsv`. The metric to optimise is `val_loss` (minimize).

## What can be improved

- Architecture (more layers, different activations, convolutions)
- Hyperparameters (learning rate, batch size, hidden dim, epochs)
- Training procedure (learning rate scheduling, momentum, weight decay)
- Data augmentation
- Regularisation (dropout, batch norm)

Only constraint: must use numpy only (no PyTorch/TensorFlow).
