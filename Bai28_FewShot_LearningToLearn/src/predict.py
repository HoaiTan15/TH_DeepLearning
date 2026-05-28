import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from utils import (
    MODEL_DIR,
    CLASS_NAMES,
    load_cifar10_data,
    convert_image_to_lstm_sequence
)


def load_models():
    """
    Load các model đã train trong thư mục saved_models.
    Lưu ý:
    fewshot_lstm_encoder.keras có Lambda layer nên cần safe_mode=False.
    """

    models = {}

    ann_path = os.path.join(MODEL_DIR, "ann_cifar10.keras")
    lstm_path = os.path.join(MODEL_DIR, "lstm_cifar10.keras")
    fewshot_path = os.path.join(MODEL_DIR, "fewshot_lstm_encoder.keras")

    if os.path.exists(ann_path):
        models["ann"] = keras.models.load_model(ann_path)
        print("Đã load ANN model:", ann_path)
    else:
        print("Chưa tìm thấy ANN model:", ann_path)

    if os.path.exists(lstm_path):
        models["lstm"] = keras.models.load_model(lstm_path)
        print("Đã load LSTM model:", lstm_path)
    else:
        print("Chưa tìm thấy LSTM model:", lstm_path)

    if os.path.exists(fewshot_path):
        models["fewshot"] = keras.models.load_model(
            fewshot_path,
            safe_mode=False
        )
        print("Đã load Few-shot model:", fewshot_path)
    else:
        print("Chưa tìm thấy Few-shot model:", fewshot_path)

    return models


def predict_ann(model, image):
    """
    Dự đoán 1 ảnh bằng ANN.
    Input image shape: 32x32x3
    """

    image_input = np.expand_dims(image, axis=0)

    pred = model.predict(image_input, verbose=0)[0]

    label_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return CLASS_NAMES[label_id], confidence


def predict_lstm(model, image):
    """
    Dự đoán 1 ảnh bằng LSTM.
    Ảnh 32x32x3 được chuyển thành chuỗi 32x96.
    """

    image_input = np.expand_dims(image, axis=0)
    image_input = image_input.reshape(1, 32, 96)

    pred = model.predict(image_input, verbose=0)[0]

    label_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return CLASS_NAMES[label_id], confidence


def build_fewshot_prototypes(encoder, k_shot=5):
    """
    Tạo prototype cho 10 lớp từ tập train CIFAR10.
    Mỗi lớp lấy k_shot ảnh đầu tiên.
    Prototype = trung bình embedding của các ảnh support.
    """

    x_train, y_train_label, y_train_cat, x_test, y_test_label, y_test_cat = load_cifar10_data()

    x_train_seq = convert_image_to_lstm_sequence(x_train)

    prototypes = []

    for class_id in range(10):
        class_images = x_train_seq[y_train_label == class_id][:k_shot]

        embeddings = encoder.predict(class_images, verbose=0)

        prototype = np.mean(embeddings, axis=0)

        prototypes.append(prototype)

    prototypes = np.array(prototypes)

    return prototypes


def predict_fewshot(encoder, image, prototypes):
    """
    Dự đoán 1 ảnh bằng Few-shot LSTM Encoder.
    Ảnh được đưa qua encoder để tạo embedding.
    Sau đó so khoảng cách với các prototype.
    """

    image_input = np.expand_dims(image, axis=0)
    image_input = image_input.reshape(1, 32, 96)

    embedding = encoder.predict(image_input, verbose=0)[0]

    distances = np.sum((prototypes - embedding) ** 2, axis=1)

    label_id = int(np.argmin(distances))

    confidence = float(1.0 / (1.0 + distances[label_id]))

    return CLASS_NAMES[label_id], confidence


def predict_test_index(index=0):
    """
    Dự đoán ảnh trong tập test CIFAR10 theo index.
    Web Flask sẽ gọi hàm này.
    """

    x_train, y_train_label, y_train_cat, x_test, y_test_label, y_test_cat = load_cifar10_data()

    if index < 0:
        index = 0

    if index >= len(x_test):
        index = len(x_test) - 1

    image = x_test[index]
    true_label = CLASS_NAMES[y_test_label[index]]

    models = load_models()

    result = {
        "true_label": true_label,
        "ann": None,
        "lstm": None,
        "fewshot": None
    }

    if "ann" in models:
        result["ann"] = predict_ann(models["ann"], image)

    if "lstm" in models:
        result["lstm"] = predict_lstm(models["lstm"], image)

    if "fewshot" in models:
        prototypes = build_fewshot_prototypes(models["fewshot"], k_shot=5)
        result["fewshot"] = predict_fewshot(
            models["fewshot"],
            image,
            prototypes
        )

    return image, result