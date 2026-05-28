import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

from utils import (
    MODEL_DIR,
    load_cifar10_data,
    convert_image_to_lstm_sequence,
    plot_history,
    save_history_csv,
    save_model_result
)


EPOCHS = 20
BATCH_SIZE = 64


def build_lstm_model():
    """
    LSTM xử lý ảnh CIFAR10 như chuỗi:
    - 32 time steps
    - mỗi time step có 96 features
    """

    model = Sequential()

    model.add(
        LSTM(
            128,
            input_shape=(32, 96),
            return_sequences=True
        )
    )
    model.add(Dropout(0.3))

    model.add(
        LSTM(
            64,
            return_sequences=False
        )
    )
    model.add(Dropout(0.3))

    model.add(Dense(64, activation="relu"))

    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    x_train, y_train_label, y_train_cat, x_test, y_test_label, y_test_cat = load_cifar10_data()

    x_train_lstm = convert_image_to_lstm_sequence(x_train)
    x_test_lstm = convert_image_to_lstm_sequence(x_test)

    model = build_lstm_model()

    print("\n========== LSTM MODEL SUMMARY ==========")
    model.summary()

    print("\n========== TRAIN LSTM ==========")

    history = model.fit(
        x_train_lstm,
        y_train_cat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )

    print("\n========== EVALUATE LSTM ==========")

    loss, accuracy = model.evaluate(x_test_lstm, y_test_cat, verbose=1)

    print("LSTM Test Loss:", loss)
    print("LSTM Test Accuracy:", accuracy)

    model_path = os.path.join(MODEL_DIR, "lstm_cifar10.keras")
    model.save(model_path)

    plot_history(history, "LSTM")
    save_history_csv(history, "history_lstm.csv")
    save_model_result("LSTM", loss, accuracy)

    print("Đã lưu model LSTM tại:", model_path)


if __name__ == "__main__":
    main()