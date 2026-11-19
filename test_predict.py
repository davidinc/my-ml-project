"""REST API test for the TensorFlow Serving MNIST model.

Examples:
    ./ml_env/bin/python test_predict.py
    ./ml_env/bin/python test_predict.py --index 25
    ./ml_env/bin/python test_predict.py --batch 20
    ./ml_env/bin/python test_predict.py --base-url http://localhost:8501
"""

import argparse
import json

import numpy as np
import requests
import tensorflow as tf


DEFAULT_BASE_URL = "http://localhost:8501"
DEFAULT_MODEL_NAME = "mnist"


def parse_arguments():
    """Parse REST API test configuration."""
    parser = argparse.ArgumentParser(description="Test the MNIST TensorFlow Serving REST API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="TensorFlow Serving host, for example http://localhost:8501")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Model name exposed by TensorFlow Serving")
    parser.add_argument("--index", type=int, help="Send one test image from this MNIST index")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of consecutive test images to send in one request")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="HTTP timeout in seconds")
    parser.add_argument("--json-output", help="Optional path for prediction results JSON")
    return parser.parse_args()


def model_urls(base_url, model_name):
    """Build status and prediction URLs for the configured model."""
    base_url = base_url.rstrip("/")
    return (
        f"{base_url}/v1/models/{model_name}",
        f"{base_url}/v1/models/{model_name}:predict",
    )


def check_model_status(status_url, timeout):
    """Raise a clear error unless the REST status endpoint reports a model is available."""
    try:
        response = requests.get(status_url, timeout=timeout)
        response.raise_for_status()
        body = response.json()
        statuses = body["model_version_status"]
    except requests.Timeout as error:
        raise RuntimeError(f"Timed out checking {status_url}") from error
    except requests.ConnectionError as error:
        raise RuntimeError(
            f"Cannot connect to TensorFlow Serving at {status_url}. "
            "Start Docker Desktop and the mnist-serving container."
        ) from error
    except (requests.RequestException, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid TensorFlow Serving status response from {status_url}") from error

    available = [item for item in statuses if item.get("state") == "AVAILABLE"]
    if not available:
        raise RuntimeError(f"The model is not available: {body}")
    return available


def send_prediction_request(predict_url, instances, timeout):
    """Send a REST Predict request and validate the JSON prediction response."""
    try:
        response = requests.post(
            predict_url,
            json={"instances": instances},
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        predictions = np.asarray(body["predictions"], dtype=np.float32)
    except requests.Timeout as error:
        raise RuntimeError(f"Prediction request timed out at {predict_url}") from error
    except requests.ConnectionError as error:
        raise RuntimeError(f"Cannot connect to the prediction endpoint: {predict_url}") from error
    except requests.HTTPError as error:
        detail = error.response.text if error.response is not None else str(error)
        raise RuntimeError(f"Prediction request failed with HTTP {error.response.status_code}: {detail}") from error
    except (requests.RequestException, KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("Invalid prediction response from TensorFlow Serving") from error

    expected_shape = (len(instances), 10)
    if predictions.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected prediction shape {predictions.shape}; expected {expected_shape}"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("Prediction response contains NaN or infinite values")
    return predictions


def main():
    """Run a single-image or batch REST API test and print the results."""
    arguments = parse_arguments()

    status_url, predict_url = model_urls(arguments.base_url, arguments.model_name)
    available_versions = check_model_status(status_url, arguments.timeout)

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if arguments.index is not None:
        start_index = arguments.index
        if arguments.batch != 1:
            raise ValueError("Use either --index for one image or --batch for consecutive images")
    else:
        start_index = 0

    batch_size = 1 if arguments.index is not None else arguments.batch
    end_index = start_index + batch_size
    if start_index < 0 or end_index > len(x_test):
        raise IndexError(
            f"Requested MNIST test range [{start_index}, {end_index}) is outside 0-{len(x_test)}"
        )

    images = x_test[start_index:end_index].astype("float32") / 255.0
    actual_labels = y_test[start_index:end_index].astype(np.int64)
    predictions = send_prediction_request(predict_url, images.tolist(), arguments.timeout)

    predicted_labels = predictions.argmax(axis=1)
    confidences = predictions[np.arange(len(predicted_labels)), predicted_labels]

    print("REST API:", predict_url)
    print("Available model versions:", ", ".join(str(item["version"]) for item in available_versions))

    results = []
    for offset, (actual, predicted, confidence) in enumerate(
        zip(actual_labels, predicted_labels, confidences)
    ):
        index = start_index + offset
        correct = int(actual) == int(predicted)
        print(
            f"Index {index}: Actual={int(actual)}, Predicted={int(predicted)}, "
            f"Confidence={float(confidence):.2%}, Correct={correct}"
        )
        results.append({
            "index": int(index),
            "actual": int(actual),
            "predicted": int(predicted),
            "confidence": float(confidence),
            "correct": bool(correct),
        })

    accuracy = float(np.mean(predicted_labels == actual_labels))
    print(f"Batch Accuracy: {accuracy:.2%}")

    if arguments.json_output:
        report = {
            "endpoint": predict_url,
            "available_versions": [item["version"] for item in available_versions],
            "batch_accuracy": accuracy,
            "results": results,
        }
        with open(arguments.json_output, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)
        print("JSON report:", arguments.json_output)


if __name__ == "__main__":
    main()