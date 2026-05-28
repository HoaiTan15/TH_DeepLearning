import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

from utils import (
    MODEL_DIR,
    load_cifar10_data,
    plot_history,
    save_history_csv,
    save_model_result,
    plot_sample_images
)


EPOCHS = 20
BATCH_SIZE = 64


def build_ann_model():
    """
    ANN xử lý ảnh CIFAR10 bằng cách Flatten ảnh 32x32x3 thành vector.
    """

    model = Sequential()

    model.add(Flatten(input_shape=(32, 32, 3)))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(128, activation="relu"))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    x_train, y_train_label, y_train_cat, x_test, y_test_label, y_test_cat = load_cifar10_data()

    plot_sample_images(x_train, y_train_label)

    model = build_ann_model()

    print("\n========== ANN MODEL SUMMARY ==========")
    model.summary()

    print("\n========== TRAIN ANN ==========")

    history = model.fit(
        x_train,
        y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )

    print("\n========== EVALUATE ANN ==========")

    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=1)

    print("ANN Test Loss:", loss)
    print("ANN Test Accuracy:", accuracy)

    model_path = os.path.join(MODEL_DIR, "ann_cifar10.keras")
    model.save(model_path)

    plot_history(history, "ANN")
    save_history_csv(history, "history_ann.csv")
    save_model_result("ANN", loss, accuracy)

    print("Đã lưu model ANN tại:", model_path)


if __name__ == "__main__":
    main()