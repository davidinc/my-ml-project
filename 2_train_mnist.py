"""Train, evaluate, visualize, and export a TensorFlow MNIST classifier.

The script intentionally exposes common machine-learning controls through command-line
arguments. It also saves learning curves, predictions, a confusion matrix, Keras and
TensorFlow models, and test metrics for analysis.

Typical usage:
    ./ml_env/bin/python 2_train_mnist.py --epochs 5 --learning-rate 0.001

Run --help to see every available experiment control:
    ./ml_env/bin/python 2_train_mnist.py --help
"""

import os

# Suppress TensorFlow informational messages; 0 enables all, 2 enables warnings/errors only.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# All generated graphs, logs, metrics, and model files are placed here.
OUTPUT_DIR = Path("mnist_outputs")


def parse_arguments():
    """Read experiment settings supplied on the command line.

    Returns:
        argparse.Namespace: Parsed options used to control preprocessing, training,
        model architecture, reproducibility, and saved-model loading.

    Example:
        --epochs 10 --dense-units 256 --dropout 0.3 --learning-rate 0.0005
    """
    parser = argparse.ArgumentParser(description="Train and analyze a configurable MNIST classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of complete training passes")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of images processed before updating weights")
    parser.add_argument("--dense-units", type=int, default=128, help="Number of neurons in the hidden Dense layer")
    parser.add_argument("--dropout", type=float, default=0.2, help="Fraction of hidden neurons randomly disabled during training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam optimizer learning rate")
    parser.add_argument("--validation-split", type=float, default=0.1, help="Fraction of training data reserved for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible experiments")
    parser.add_argument("--skip-training", action="store_true", help="Load a saved model instead of training")
    return parser.parse_args()


def set_seed(seed):
    """Seed Python, NumPy, and TensorFlow random operations.

    Args:
        seed (int): Reproducibility value used by all random-number generators.

    A fixed seed makes initialization and shuffling repeatable. It does not
    guarantee identical results on every computer or every TensorFlow backend.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_prepare_data():
    """Download MNIST and normalize its grayscale pixel values.

    Returns:
        tuple: ((training_images, training_labels),
                (testing_images, testing_labels)).

    MNIST has 60,000 training images and 10,000 testing images. Each image is
    28 by 28 pixels and originally contains values from 0 to 255. Dividing by
    255 converts each pixel to the stable range 0.0 to 1.0.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", x_test.shape)
    print("Pixel range:", float(x_train.min()), float(x_train.max()))

    return (x_train, y_train), (x_test, y_test)


def build_model(dense_units, dropout, learning_rate):
    """Create, compile, and describe a dense neural-network classifier.

    Args:
        dense_units (int): Number of neurons in the hidden Dense layer.
        dropout (float): Fraction of hidden activations disabled during training.
        learning_rate (float): Step size used by the Adam optimizer.

    Returns:
        tf.keras.Model: Compiled Keras Sequential model for digits 0 through 9.
    """
    model = tf.keras.models.Sequential([
        # Defines the shape of one grayscale MNIST image.
        tf.keras.Input(shape=(28, 28), name="image_input"),

        # Converts the 28 x 28 matrix into 784 individual input values.
        tf.keras.layers.Flatten(name="flatten_pixels"),

        # Learns combinations of pixels. ReLU keeps positive activations and
        # turns negative activations into zero.
        tf.keras.layers.Dense(dense_units, activation="relu", name="hidden_layer"),

        # Randomly disables neurons only during training to reduce overfitting.
        tf.keras.layers.Dropout(dropout, name="dropout"),

        # Produces one probability for each digit 0 through 9.
        tf.keras.layers.Dense(10, activation="softmax", name="digit_probabilities"),
    ])

    model.compile(
        # Adam adapts the weight-update step during training.
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        # Integer labels 0-9 match this sparse cross-entropy loss.
        loss="sparse_categorical_crossentropy",

        # Accuracy is the percentage of correct predictions.
        metrics=["accuracy"],
    )

    model.summary()
    return model


def make_callbacks():
    """Create callbacks that monitor, preserve, and control training.

    Returns:
        list[tf.keras.callbacks.Callback]: Callbacks for early stopping, best
        model checkpointing, learning-rate reduction, and CSV logging.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return [
        # Stops if validation accuracy fails to improve for two epochs.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),

        # Saves only the model with the highest observed validation accuracy.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=OUTPUT_DIR / "mnist_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        # Halves the learning rate after no validation-loss improvement.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),

        # Writes epoch metrics for later plotting and analysis.
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR / "training_history.csv"),
    ]


def plot_history(history):
    """Plot training and validation accuracy and loss against epoch number.

    Args:
        history (tf.keras.callbacks.History): Result returned by model.fit.

    The image is written to mnist_outputs/training_history.png. A widening
    training/validation gap can indicate overfitting.
    """
    history_data = history.history

    # Two side-by-side plots provide a compact learning diagnostic.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_data["accuracy"], label="Training accuracy")
    axes[0].plot(history_data["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_data["loss"], label="Training loss")
    axes[1].plot(history_data["val_loss"], label="Validation loss")
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred):
    """Visualize which actual digits the model confuses with each prediction.

    Args:
        y_true (array-like): Actual integer labels from 0 through 9.
        y_pred (array-like): Predicted integer labels from 0 through 9.

    Row positions represent actual digits and column positions represent predicted
    digits. A strong diagonal and low off-diagonal values indicate good accuracy.
    """
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=10).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title("MNIST Confusion Matrix")
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("Actual Digit")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    threshold = matrix.max() / 2
    for row in range(10):
        for column in range(10):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_prediction_examples(model, x_test, y_test):
    """Display 25 test images with actual and predicted digit labels.

    Args:
        model (tf.keras.Model): Trained or loaded classifier.
        x_test (array-like): Unseen test images.
        y_test (array-like): Actual test labels.

    The title confidence is the probability assigned to the highest output neuron.
    This makes correct, incorrect, and uncertain predictions easy to inspect.
    """
    probabilities = model.predict(x_test[:25], verbose=0)

    # argmax chooses the class with the largest probability for every image.
    predictions = probabilities.argmax(axis=1)

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    for index, ax in enumerate(axes.flat):
        actual = int(y_test[index])
        predicted = int(predictions[index])
        confidence = float(probabilities[index, predicted])

        ax.imshow(x_test[index], cmap="gray")
        ax.set_title(f"Actual: {actual}\nPredicted: {predicted} ({confidence:.1%})")
        ax.axis("off")

    fig.suptitle("Sample Predictions", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_examples.png", dpi=150)
    plt.close(fig)


def predict_from_image(model, image_path):
    """Preprocess and classify an external 28-pixel-style digit image.

    Args:
        model (tf.keras.Model): Trained or loaded classifier.
        image_path (str or Path): Path to a PNG or another supported image.

    If custom_digit.png does not exist, prediction is skipped so normal training
    remains unaffected. Conventional white-background drawings are inverted because
    MNIST digits are bright pixels on a dark background.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        return

    image = tf.keras.preprocessing.image.load_img(
        str(image_path), color_mode="grayscale", target_size=(28, 28)
    )
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Convert a conventional white-background image to MNIST-style white digits.
    if float(image_array.mean()) > 127:
        image_array = 255.0 - image_array

    image_array = image_array.astype("float32") / 255.0

    # Add a batch dimension: (28, 28, 1) becomes (1, 28, 28, 1).
    # However, this model receives (batch, 28, 28), so only a batch dimension is needed.
    probabilities = model.predict(image_array[None, ..., 0], verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))

    print(f"Custom image prediction: {predicted_digit}")
    print("Probability for every digit:")
    for digit, probability in enumerate(probabilities):
        print(f"  {digit}: {probability:.2%}")


def evaluate_and_export(model, x_test, y_test):
    """Measure test performance and save every reusable model format and report.

    Args:
        model (tf.keras.Model): Trained or loaded classifier.
        x_test (array-like): Unseen test images.
        y_test (array-like): Actual test labels.

    Returns:
        dict: Test loss, test accuracy, and test-image count.
    """
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    probabilities = model.predict(x_test, verbose=0)
    predicted_labels = probabilities.argmax(axis=1)

    plot_confusion_matrix(y_test, predicted_labels)

    report = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_images": int(len(x_test)),
    }

    with open(OUTPUT_DIR / "evaluation.json", "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    model.save(OUTPUT_DIR / "mnist_final.keras")
    model.export(OUTPUT_DIR / "mnist_saved_model" / "1")

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print("Saved .keras model:", OUTPUT_DIR / "mnist_final.keras")
    print("Saved TensorFlow model:", OUTPUT_DIR / "mnist_saved_model" / "1")

    return report


def main():
    """Main entry point for training, evaluating, and using the MNIST classifier.

    It parses command-line arguments, sets the random seed, loads or prepares the data,
    builds or loads the model, trains the model if not skipped, evaluates the model,
    and exports the results and model.
    """
    arguments = parse_arguments()
    set_seed(arguments.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_and_prepare_data()
    keras_path = OUTPUT_DIR / "mnist_final.keras"

    if arguments.skip_training:
        if not keras_path.exists():
            raise FileNotFoundError(
                f"No saved model found at {keras_path}. Run training before --skip-training."
            )
        model = tf.keras.models.load_model(keras_path)
        print("Loaded existing model:", keras_path)
    else:
        model = build_model(
            dense_units=arguments.dense_units,
            dropout=arguments.dropout,
            learning_rate=arguments.learning_rate,
        )
        history = model.fit(
            x_train,
            y_train,
            epochs=arguments.epochs,
            batch_size=arguments.batch_size,
            validation_split=arguments.validation_split,
            callbacks=make_callbacks(),
            verbose=2,
        )
        plot_history(history)

    plot_prediction_examples(model, x_test, y_test)
    evaluate_and_export(model, x_test, y_test)

    custom_image = Path("custom_digit.png")
    predict_from_image(model, custom_image)


if __name__ == "__main__":
    main()