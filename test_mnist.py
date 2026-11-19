import numpy as np
import tensorflow as tf

import importlib.util

spec = importlib.util.spec_from_file_location("mnist_training", "2_train_mnist.py")
mnist_training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mnist_training)


def test_seed_changes_only_random_state_not_model_shape():
    mnist_training.set_seed(42)
    assert np.random.randint(0, 100) == np.random.RandomState(42).randint(0, 100)


def test_load_and_prepare_normalizes_images():
    (x_train, y_train), (x_test, y_test) = mnist_training.load_and_prepare_data()

    assert x_train.dtype == np.float32
    assert x_train.min() == 0.0
    assert x_train.max() == 1.0
    assert x_train.shape[1:] == (28, 28)
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


def test_model_has_expected_input_and_output_shapes():
    model = mnist_training.build_model(dense_units=32, dropout=0.2, learning_rate=0.001)

    assert model.input_shape == (None, 28, 28)
    assert model.output_shape == (None, 10)
    assert len(model.weights) == 4


def test_model_prediction_is_valid_probability_distribution():
    model = mnist_training.build_model(dense_units=32, dropout=0.0, learning_rate=0.001)
    images = np.zeros((2, 28, 28), dtype=np.float32)
    probabilities = model.predict(images, verbose=0)

    assert probabilities.shape == (2, 10)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
