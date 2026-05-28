import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


def load_cifar10_data():
    """
    Load CIFAR10 có sẵn trong TensorFlow.
    Ảnh có kích thước 32x32x3, gồm 10 lớp.
    """

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_label = y_train.flatten()
    y_test_label = y_test.flatten()

    y_train_cat = keras.utils.to_categorical(y_train_label, 10)
    y_test_cat = keras.utils.to_categorical(y_test_label, 10)

    return x_train, y_train_label, y_train_cat, x_test, y_test_label, y_test_cat


def convert_image_to_lstm_sequence(x):
    """
    Chuyển ảnh CIFAR10 từ 32x32x3 thành chuỗi:
    32 time steps, mỗi step có 32*3 = 96 features.
    """

    return x.reshape(x.shape[0], 32, 96)


def save_history_csv(history, filename):
    path = os.path.join(RESULT_DIR, filename)

    keys = list(history.history.keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)

        for i in range(len(history.history[keys[0]])):
            row = [i + 1]
            for key in keys:
                row.append(history.history[key][i])
            writer.writerow(row)

    print("Đã lưu history:", path)


def plot_history(history, model_name):
    """
    Vẽ accuracy và loss cho model.
    """

    # Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="validation")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, f"{model_name.lower()}_accuracy.png"), dpi=300)
    plt.close()

    # Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, f"{model_name.lower()}_loss.png"), dpi=300)
    plt.close()

    print(f"Đã lưu biểu đồ {model_name}.")


def save_model_result(model_name, loss, accuracy):
    path = os.path.join(RESULT_DIR, "model_results.csv")

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model", "loss", "accuracy"])

        writer.writerow([model_name, loss, accuracy])

    print("Đã lưu kết quả:", model_name)


def plot_sample_images(x, y, filename="cifar10_samples.png"):
    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i])
        plt.title(CLASS_NAMES[y[i]])
        plt.axis("off")

    plt.suptitle("Một số ảnh trong CIFAR10")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, filename), dpi=300)
    plt.close()