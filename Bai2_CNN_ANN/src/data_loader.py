import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from config import NUM_CLASSES


def load_cifar10_data():
    """
    Load CIFAR-10 dataset.
    Normalize ảnh về [0, 1].
    One-hot label để dùng categorical_crossentropy.
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    return x_train, y_train_cat, x_test, y_test_cat, y_test