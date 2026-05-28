import os
import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, EPOCHS, SAVED_MODEL_DIR, RESULTS_DIR
from data_loader import load_cifar10_data
from models import build_ann
from utils import make_dir, plot_history, save_confusion_matrix, save_classification_report


def main():
    model_name = "ann"
    result_dir = os.path.join(RESULTS_DIR, model_name)
    make_dir(result_dir)
    make_dir(SAVED_MODEL_DIR)

    x_train, y_train, x_test, y_test_cat, y_test_raw = load_cifar10_data()

    model = build_ann()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=callbacks
    )

    model_path = os.path.join(SAVED_MODEL_DIR, "ann_model.keras")
    model.save(model_path)

    test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"ANN Test Accuracy: {test_acc:.4f}")

    y_pred_prob = model.predict(x_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = y_test_raw.flatten()

    plot_history(history, os.path.join(result_dir, "accuracy_loss.png"))
    save_confusion_matrix(y_true, y_pred, os.path.join(result_dir, "confusion_matrix.png"))
    save_classification_report(y_true, y_pred, os.path.join(result_dir, "classification_report.txt"))

    with open(os.path.join(result_dir, "result.txt"), "w", encoding="utf-8") as f:
        f.write(f"ANN Test Loss: {test_loss:.4f}\n")
        f.write(f"ANN Test Accuracy: {test_acc:.4f}\n")


if __name__ == "__main__":
    main()