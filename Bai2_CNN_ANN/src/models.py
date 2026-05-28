import tensorflow as tf
from tensorflow.keras import layers, models

from config import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ann():
    model = models.Sequential(name="ANN_Baseline")

    model.add(layers.Input(shape=INPUT_SHAPE))
    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return compile_model(model)


def build_cnn():
    model = models.Sequential(name="CNN_Baseline")

    model.add(layers.Input(shape=INPUT_SHAPE))

    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return compile_model(model)


def build_improved_cnn():
    model = models.Sequential(name="Improved_CNN_ANN")

    model.add(layers.Input(shape=INPUT_SHAPE))

    model.add(layers.RandomFlip("horizontal"))
    model.add(layers.RandomRotation(0.1))
    model.add(layers.RandomZoom(0.1))

    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.40))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    return compile_model(model)