import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

from utils import (
    MODEL_DIR,
    RESULT_DIR,
    load_cifar10_data,
    convert_image_to_lstm_sequence,
    save_model_result
)


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Few-shot setting
N_WAY = 5
K_SHOT = 5
Q_QUERY = 10

# Nếu máy yếu thì giảm EPISODES xuống 300
EPISODES = 1000
EVAL_EPISODES = 100
LEARNING_RATE = 0.001


def build_fewshot_lstm_encoder():
    """
    Encoder dùng LSTM.
    Input: ảnh CIFAR10 dạng chuỗi 32 time steps, mỗi step 96 features.
    Output: embedding vector 64 chiều.

    Lưu ý:
    Không dùng Lambda layer để tránh lỗi khi load model trong Flask.
    """

    inputs = keras.Input(shape=(32, 96))

    x = layers.LSTM(
        128,
        return_sequences=True,
        name="lstm_encoder_1"
    )(inputs)
    x = layers.Dropout(0.3, name="dropout_1")(x)

    x = layers.LSTM(
        64,
        return_sequences=False,
        name="lstm_encoder_2"
    )(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    x = layers.Dense(
        64,
        activation="relu",
        name="embedding_dense"
    )(x)

    # Thay cho Lambda(lambda t: tf.math.l2_normalize(...))
    # Layer này save/load an toàn hơn trong Keras.
    outputs = layers.UnitNormalization(
        axis=1,
        name="embedding_normalization"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="fewshot_lstm_encoder"
    )

    return model


def group_data_by_class(x, y):
    """
    Gom dữ liệu theo từng class.
    """

    data_by_class = {}

    for class_id in range(10):
        data_by_class[class_id] = x[y == class_id]

    return data_by_class


def sample_episode(data_by_class, n_way, k_shot, q_query):
    """
    Sinh một episode cho Few-shot Learning.

    Mỗi episode gồm:
    - N_WAY lớp
    - Mỗi lớp có K_SHOT ảnh support
    - Mỗi lớp có Q_QUERY ảnh query
    """

    selected_classes = np.random.choice(
        list(data_by_class.keys()),
        size=n_way,
        replace=False
    )

    support_x = []
    query_x = []
    query_y = []

    for new_label, class_id in enumerate(selected_classes):
        class_data = data_by_class[class_id]

        selected_indices = np.random.choice(
            len(class_data),
            size=k_shot + q_query,
            replace=False
        )

        selected_data = class_data[selected_indices]

        support_samples = selected_data[:k_shot]
        query_samples = selected_data[k_shot:]

        support_x.append(support_samples)
        query_x.append(query_samples)
        query_y.extend([new_label] * q_query)

    support_x = np.concatenate(support_x, axis=0)
    query_x = np.concatenate(query_x, axis=0)
    query_y = np.array(query_y, dtype=np.int32)

    return support_x, query_x, query_y


def compute_prototypes(support_embeddings, n_way, k_shot):
    """
    Tính prototype cho từng lớp.
    Prototype = trung bình embedding của support samples trong cùng một lớp.
    """

    embedding_dim = tf.shape(support_embeddings)[-1]

    support_embeddings = tf.reshape(
        support_embeddings,
        shape=(n_way, k_shot, embedding_dim)
    )

    prototypes = tf.reduce_mean(support_embeddings, axis=1)

    return prototypes


def compute_logits(query_embeddings, prototypes):
    """
    Tính khoảng cách Euclidean từ query embedding đến prototype.
    Vì loss cần logits càng lớn càng tốt, ta dùng logits = -distance.
    """

    query_expand = tf.expand_dims(query_embeddings, axis=1)
    proto_expand = tf.expand_dims(prototypes, axis=0)

    distances = tf.reduce_sum(
        tf.square(query_expand - proto_expand),
        axis=2
    )

    logits = -distances

    return logits


def evaluate_fewshot(encoder, data_by_class, eval_episodes):
    """
    Đánh giá Few-shot model qua nhiều episode.
    """

    total_acc = []

    for _ in range(eval_episodes):
        support_x, query_x, query_y = sample_episode(
            data_by_class,
            N_WAY,
            K_SHOT,
            Q_QUERY
        )

        support_embeddings = encoder(support_x, training=False)
        query_embeddings = encoder(query_x, training=False)

        prototypes = compute_prototypes(
            support_embeddings,
            N_WAY,
            K_SHOT
        )

        logits = compute_logits(query_embeddings, prototypes)

        pred = tf.argmax(logits, axis=1)

        acc = tf.reduce_mean(
            tf.cast(pred == query_y, tf.float32)
        )

        total_acc.append(float(acc.numpy()))

    return np.mean(total_acc)


def main():
    x_train, y_train_label, y_train_cat, x_test, y_test_label, y_test_cat = load_cifar10_data()

    x_train_seq = convert_image_to_lstm_sequence(x_train)
    x_test_seq = convert_image_to_lstm_sequence(x_test)

    train_by_class = group_data_by_class(x_train_seq, y_train_label)
    test_by_class = group_data_by_class(x_test_seq, y_test_label)

    encoder = build_fewshot_lstm_encoder()

    print("\n========== FEW-SHOT LSTM ENCODER SUMMARY ==========")
    encoder.summary()

    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_losses = []
    train_accs = []

    print("\n========== TRAIN FEW-SHOT LEARNING TO LEARN ==========")

    for episode in range(1, EPISODES + 1):
        support_x, query_x, query_y = sample_episode(
            train_by_class,
            N_WAY,
            K_SHOT,
            Q_QUERY
        )

        with tf.GradientTape() as tape:
            support_embeddings = encoder(support_x, training=True)
            query_embeddings = encoder(query_x, training=True)

            prototypes = compute_prototypes(
                support_embeddings,
                N_WAY,
                K_SHOT
            )

            logits = compute_logits(query_embeddings, prototypes)

            loss = loss_fn(query_y, logits)

        gradients = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, encoder.trainable_variables)
        )

        pred = tf.argmax(logits, axis=1)

        acc = tf.reduce_mean(
            tf.cast(pred == query_y, tf.float32)
        )

        train_losses.append(float(loss.numpy()))
        train_accs.append(float(acc.numpy()))

        if episode % 50 == 0:
            eval_acc = evaluate_fewshot(
                encoder,
                test_by_class,
                eval_episodes=20
            )

            print(
                f"Episode {episode}/{EPISODES} | "
                f"Loss: {loss.numpy():.4f} | "
                f"Train Acc: {acc.numpy():.4f} | "
                f"Eval Acc: {eval_acc:.4f}"
            )

    final_acc = evaluate_fewshot(
        encoder,
        test_by_class,
        eval_episodes=EVAL_EPISODES
    )

    print("\n========== FEW-SHOT RESULT ==========")
    print("Few-shot Test Accuracy:", final_acc)

    model_path = os.path.join(
        MODEL_DIR,
        "fewshot_lstm_encoder.keras"
    )

    encoder.save(model_path)

    history_path = os.path.join(
        RESULT_DIR,
        "history_fewshot.csv"
    )

    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "loss", "accuracy"])

        for i in range(len(train_losses)):
            writer.writerow([
                i + 1,
                train_losses[i],
                train_accs[i]
            ])

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Few-shot train loss")
    plt.title("Few-shot Learning Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(RESULT_DIR, "fewshot_loss.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Few-shot train accuracy")
    plt.title("Few-shot Learning Accuracy")
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(RESULT_DIR, "fewshot_accuracy.png"),
        dpi=300
    )
    plt.close()

    save_model_result(
        "Few-shot LSTM Encoder",
        0.0,
        final_acc
    )

    print("Đã lưu Few-shot model tại:", model_path)
    print("Đã lưu history tại:", history_path)


if __name__ == "__main__":
    main()