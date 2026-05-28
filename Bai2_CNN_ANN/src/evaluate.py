import os
import argparse
import numpy as np
import tensorflow as tf

from config import SAVED_MODEL_DIR, RESULTS_DIR
from data_loader import load_cifar10_data
from utils import make_dir, save_confusion_matrix, save_classification_report


MODEL_FILES = {
    "ann": "ann_model.keras",
    "cnn": "cnn_model.keras",
    "improved_cnn": "improved_cnn_model.keras",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["ann", "cnn", "improved_cnn"],
        required=True
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join(SAVED_MODEL_DIR, MODEL_FILES[model_name])
    result_dir = os.path.join(RESULTS_DIR, model_name)
    make_dir(result_dir)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    _, _, x_test, y_test_cat, y_test_raw = load_cifar10_data()

    model = tf.keras.models.load_model(model_path)

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"{model_name} Test Loss: {test_loss:.4f}")
    print(f"{model_name} Test Accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test_raw.flatten()

    save_confusion_matrix(
        y_true,
        y_pred,
        os.path.join(result_dir, "confusion_matrix.png")
    )

    save_classification_report(
        y_true,
        y_pred,
        os.path.join(result_dir, "classification_report.txt")
    )


if __name__ == "__main__":
    main()