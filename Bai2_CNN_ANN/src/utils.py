import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report

from config import CLASS_NAMES


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_history(history, save_path):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_classification_report(y_true, y_pred, save_path):
    """
    Lưu classification report dưới 2 dạng:
    1. .txt để đọc văn bản
    2. .csv để Streamlit hiển thị dạng bảng đẹp
    """

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True
    )

    df_report = pd.DataFrame(report_dict).transpose()

    csv_path = save_path.replace(".txt", ".csv")
    df_report.to_csv(csv_path, encoding="utf-8")