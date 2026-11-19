# MNIST Learning Guide

Run from the project directory.

## Baseline

```bash
./ml_env/bin/python 2_train_mnist.py
```

This trains for five epochs with 128-neuron Dense layer, 20% dropout, learning rate `0.001`, and a 10% validation split.

## Experiment Without Guessing

Change one parameter at a time:

```bash
./ml_env/bin/python 2_train_mnist.py --epochs 3 --learning-rate 0.001
./ml_env/bin/python 2_train_mnist.py --epochs 10 --learning-rate 0.001
./ml_env/bin/python 2_train_mnist.py --epochs 10 --learning-rate 0.0001
./ml_env/bin/python 2_train_mnist.py --epochs 10 --dense-units 256 --dropout 0.3
```

### Parameter meanings

| Option | Effect |
|---|---|
| `--epochs` | Complete passes through the training images |
| `--batch-size` | Images used for each weight update |
| `--dense-units` | Hidden-layer capacity; larger can learn more detail but may overfit |
| `--dropout` | Randomly disables neurons during training to improve generalization |
| `--learning-rate` | Step size used by Adam; too high diverges, too low learns slowly |
| `--validation-split` | Fraction kept aside to estimate generalization |
| `--seed` | Makes random choices repeatable |

## Smart Investigation Process

1. Create a baseline and record test accuracy.
2. Change only one argument.
3. Repeat the same seed and record the new result.
4. Compare `training_history.csv`, `training_history.png`, and `evaluation.json`.
5. Do not conclude that a small number of runs proves improvement. Repeat important experiments.

## Overfitting Signals

Training accuracy rising while validation accuracy stops improving or falls indicates overfitting.

Try:

```bash
./ml_env/bin/python 2_train_mnist.py --epochs 20 --dropout 0.5
```

## Underfitting Signals

Training and validation accuracy remain low and improve slowly. Try more capacity or a longer run:

```bash
./ml_env/bin/python 2_train_mnist.py --epochs 10 --dense-units 256
```

## Loading the Trained Model

After a successful training run:

```bash
./ml_env/bin/python 2_train_mnist.py --skip-training
```

This loads `mnist_outputs/mnist_final.keras` and re-evaluates it.

## Predicting Your Own Image

Place a $28 \times 28$ or larger grayscale/custom-digit PNG named `custom_digit.png` in the project directory. The script will automatically invert very white images and resize them to $28 \times 28$.

Draw the digit using black ink on a white background for an MNIST-like image.

## Outputs

| File | Purpose |
|---|---|
| `mnist_outputs/mnist_final.keras` | Complete reusable Keras model |
| `mnist_outputs/mnist_best.keras` | Best validation checkpoint |
| `mnist_outputs/mnist_saved_model/1` | TensorFlow SavedModel for serving |
| `mnist_outputs/training_history.csv` | Raw epoch metrics |
| `mnist_outputs/training_history.png` | Accuracy/loss learning curves |
| `mnist_outputs/confusion_matrix.png` | Actual versus predicted digits |
| `mnist_outputs/prediction_examples.png` | 25 example predictions |
| `mnist_outputs/evaluation.json` | Test loss and accuracy |

## Run Tests

```bash
./ml_env/bin/python -m pytest -q test_mnist.py
```
