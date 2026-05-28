import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from PIL import Image

from src.config import CLASS_NAMES, SAVED_MODEL_DIR, RESULTS_DIR


MODEL_OPTIONS = {
    "ANN": "ann_model.keras",
    "CNN": "cnn_model.keras",
    "Improved CNN": "improved_cnn_model.keras",
}

RESULT_FOLDER = {
    "ANN": "ann",
    "CNN": "cnn",
    "Improved CNN": "improved_cnn",
}


st.set_page_config(
    page_title="CIFAR-10 CNN ANN Demo",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_selected_model(model_file):
    model_path = os.path.join(SAVED_MODEL_DIR, model_file)

    if not os.path.exists(model_path):
        return None

    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_cifar10_test_data():
    (_, _), (x_test, y_test) = cifar10.load_data()

    x_test_norm = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()

    return x_test, x_test_norm, y_test


def predict_image(model, image_array):
    input_image = np.expand_dims(image_array, axis=0)
    pred = model.predict(input_image, verbose=0)

    class_id = int(np.argmax(pred))
    confidence = float(np.max(pred))

    return class_id, confidence, pred[0]


st.title("Đồ án 2: Phân loại ảnh CIFAR-10 bằng ANN, CNN và CNN cải tiến")

st.markdown("""
Ứng dụng demo sử dụng trực tiếp **tập test CIFAR-10** để dự đoán ảnh.

Chức năng chính:

- Chọn mô hình ANN / CNN / Improved CNN
- Chọn ảnh từ tập test CIFAR-10
- Hiển thị nhãn thật và nhãn dự đoán
- Xem biểu đồ Accuracy/Loss, Confusion Matrix, Classification Report
- Xem bộ siêu tham số của 3 mô hình
""")

model_label = st.sidebar.selectbox(
    "Chọn mô hình",
    list(MODEL_OPTIONS.keys())
)

model = load_selected_model(MODEL_OPTIONS[model_label])

if model is None:
    st.warning(
        f"Chưa tìm thấy model `{MODEL_OPTIONS[model_label]}`. "
        f"Hãy train model trước."
    )
else:
    st.success(f"Đã load model: {model_label}")


x_test_original, x_test_norm, y_test = load_cifar10_test_data()


tab1, tab2, tab3, tab4 = st.tabs([
    "Dự đoán ảnh CIFAR-10",
    "Kết quả huấn luyện",
    "Thông tin mô hình",
    "Siêu tham số"
])


with tab1:
    st.subheader("Dự đoán ảnh từ tập test CIFAR-10")

    st.markdown("""
    Dữ liệu demo được lấy từ **10.000 ảnh test của CIFAR-10**.  
    Mỗi ảnh đều có nhãn thật để so sánh với kết quả dự đoán của mô hình.
    """)

    col_select_1, col_select_2 = st.columns([2, 1])

    with col_select_1:
        image_index = st.slider(
            "Chọn chỉ số ảnh trong tập test",
            min_value=0,
            max_value=len(x_test_original) - 1,
            value=0,
            step=1
        )

    with col_select_2:
        if st.button("Chọn ảnh ngẫu nhiên"):
            image_index = int(np.random.randint(0, len(x_test_original)))
            st.session_state["random_image_index"] = image_index

    if "random_image_index" in st.session_state:
        image_index = st.session_state["random_image_index"]

    image_original = x_test_original[image_index]
    image_norm = x_test_norm[image_index]
    true_label_id = int(y_test[image_index])
    true_label_name = CLASS_NAMES[true_label_id]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Ảnh CIFAR-10 test")

        display_image = Image.fromarray(image_original)
        st.image(
            display_image.resize((256, 256)),
            caption=f"Ảnh test index = {image_index}",
            use_container_width=False
        )

        st.write(f"**Nhãn thật:** `{true_label_name}`")

    with col2:
        st.markdown("### Kết quả dự đoán")

        if model is not None:
            pred_label_id, confidence, probabilities = predict_image(
                model,
                image_norm
            )

            pred_label_name = CLASS_NAMES[pred_label_id]

            if pred_label_id == true_label_id:
                st.success(f"Dự đoán đúng: `{pred_label_name}`")
            else:
                st.error(f"Dự đoán sai: `{pred_label_name}`")

            st.write(f"**Nhãn dự đoán:** `{pred_label_name}`")
            st.write(f"**Độ tin cậy:** `{confidence:.4f}`")

            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": probabilities
            })

            prob_df = prob_df.sort_values(
                by="Probability",
                ascending=False
            )

            st.markdown("#### Xác suất dự đoán từng lớp")
            st.dataframe(
                prob_df.style.format({"Probability": "{:.4f}"}),
                use_container_width=True
            )

            st.bar_chart(prob_df.set_index("Class"))

        else:
            st.info("Chưa load được model để dự đoán.")


with tab2:
    st.subheader("Kết quả huấn luyện")

    folder = RESULT_FOLDER[model_label]
    result_dir = os.path.join(RESULTS_DIR, folder)

    acc_loss_path = os.path.join(result_dir, "accuracy_loss.png")
    cm_path = os.path.join(result_dir, "confusion_matrix.png")
    report_txt_path = os.path.join(result_dir, "classification_report.txt")
    report_csv_path = os.path.join(result_dir, "classification_report.csv")
    result_txt_path = os.path.join(result_dir, "result.txt")

    st.markdown(f"### Mô hình đang xem: `{model_label}`")

    if os.path.exists(result_txt_path):
        st.markdown("#### Kết quả tổng quan")
        with open(result_txt_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("Chưa có file result.txt. Hãy train model trước.")

    st.markdown("---")

    if os.path.exists(acc_loss_path):
        st.markdown("#### Biểu đồ Accuracy và Loss")
        st.image(
            acc_loss_path,
            caption="Accuracy và Loss",
            use_container_width=True
        )
    else:
        st.info("Chưa có biểu đồ Accuracy/Loss.")

    st.markdown("---")

    if os.path.exists(cm_path):
        st.markdown("#### Ma trận nhầm lẫn")
        st.image(
            cm_path,
            caption="Confusion Matrix",
            use_container_width=True
        )
    else:
        st.info("Chưa có Confusion Matrix.")

    st.markdown("---")

    st.markdown("#### Classification Report")

    if os.path.exists(report_csv_path):
        df_report = pd.read_csv(report_csv_path, index_col=0)

        display_columns = ["precision", "recall", "f1-score", "support"]
        df_report = df_report[display_columns]

        st.dataframe(
            df_report.style.format({
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }),
            use_container_width=True
        )

    elif os.path.exists(report_txt_path):
        with open(report_txt_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("Chưa có Classification Report.")


with tab3:
    st.subheader("Thông tin mô hình")

    if model is not None:
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        summary = "\n".join(string_list)

        st.code(summary, language="text")
    else:
        st.info("Chưa load được model.")


with tab4:
    st.subheader("Bộ siêu tham số sử dụng trong thực nghiệm")

    st.markdown("### 1. Siêu tham số chung")

    common_hyperparams = pd.DataFrame({
        "Nhóm": [
            "Dữ liệu",
            "Dữ liệu",
            "Dữ liệu",
            "Huấn luyện",
            "Huấn luyện",
            "Huấn luyện",
            "Huấn luyện",
            "Huấn luyện",
            "Đánh giá",
            "Đánh giá",
        ],
        "Siêu tham số": [
            "Dataset",
            "Kích thước ảnh đầu vào",
            "Số lớp phân loại",
            "Epochs",
            "Batch size",
            "Optimizer",
            "Learning rate",
            "Loss function",
            "Validation split",
            "Metric",
        ],
        "Giá trị": [
            "CIFAR-10",
            "32 × 32 × 3",
            "10",
            "30",
            "64",
            "Adam",
            "0.001",
            "Categorical Crossentropy",
            "0.2",
            "Accuracy",
        ],
    })

    st.dataframe(common_hyperparams, use_container_width=True)

    st.markdown("---")

    st.markdown("### 2. Siêu tham số theo từng mô hình")

    model_hyperparams = pd.DataFrame({
        "Mô hình": [
            "ANN",
            "ANN",
            "ANN",
            "ANN",
            "CNN",
            "CNN",
            "CNN",
            "CNN",
            "CNN",
            "Improved CNN",
            "Improved CNN",
            "Improved CNN",
            "Improved CNN",
            "Improved CNN",
            "Improved CNN",
            "Improved CNN",
        ],
        "Thành phần": [
            "Flatten",
            "Dense layers",
            "Dropout",
            "Output layer",
            "Conv2D filters",
            "Kernel size",
            "Pooling size",
            "Dense layer",
            "Dropout",
            "Data Augmentation",
            "Conv2D filters",
            "Batch Normalization",
            "Dropout",
            "GlobalAveragePooling2D",
            "Dense layers",
            "Output layer",
        ],
        "Giá trị": [
            "Có",
            "512, 256, 128",
            "0.3, 0.3",
            "Dense(10), Softmax",
            "32, 64, 128",
            "3 × 3",
            "2 × 2",
            "256",
            "0.4",
            "RandomFlip, RandomRotation(0.1), RandomZoom(0.1)",
            "32, 64, 128",
            "Có",
            "0.25, 0.30, 0.40, 0.50, 0.40",
            "Có",
            "512, 256",
            "Dense(10), Softmax",
        ],
    })

    st.dataframe(model_hyperparams, use_container_width=True)

    st.markdown("---")

    st.markdown("### 3. Siêu tham số Callback")

    callback_hyperparams = pd.DataFrame({
        "Callback": [
            "EarlyStopping",
            "EarlyStopping",
            "EarlyStopping",
            "ReduceLROnPlateau",
            "ReduceLROnPlateau",
            "ReduceLROnPlateau",
            "ReduceLROnPlateau",
        ],
        "Siêu tham số": [
            "monitor",
            "patience",
            "restore_best_weights",
            "monitor",
            "factor",
            "patience",
            "min_lr",
        ],
        "Giá trị": [
            "val_loss",
            "8",
            "True",
            "val_loss",
            "0.5",
            "3",
            "1e-6",
        ],
    })

    st.dataframe(callback_hyperparams, use_container_width=True)

    st.markdown("---")

    st.markdown("### 4. Ý nghĩa mô hình cải tiến")

    st.info(
        "Mô hình Improved CNN sử dụng CNN để trích xuất đặc trưng ảnh, "
        "sau đó dùng các lớp Dense của ANN để phân loại. "
        "So với CNN baseline, mô hình được bổ sung Data Augmentation, "
        "Batch Normalization, Dropout, GlobalAveragePooling2D, "
        "EarlyStopping và ReduceLROnPlateau nhằm tăng khả năng tổng quát hóa "
        "và giảm overfitting."
    )