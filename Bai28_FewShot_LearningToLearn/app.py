import os
import sys
import base64
import random
from io import BytesIO

from flask import Flask, request, render_template_string, send_from_directory
from PIL import Image
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from predict import predict_test_index
from utils import RESULT_DIR


app = Flask(__name__)


def image_to_base64(image_array):
    image_uint8 = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_uint8)

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_base64


HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Bài 28 - Few-shot Learning to Learn</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 1250px;
            margin: auto;
            background: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.12);
        }

        h1, h2, h3 {
            color: #111827;
        }

        .card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 18px;
            margin-top: 18px;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            background: white;
        }

        th, td {
            border: 1px solid #d1d5db;
            padding: 10px;
            text-align: left;
            vertical-align: top;
        }

        th {
            background: #e5e7eb;
        }

        .image-box {
            text-align: center;
            margin-top: 20px;
        }

        .image-box img {
            width: 220px;
            image-rendering: pixelated;
            border: 1px solid #ddd;
            border-radius: 8px;
        }

        input, button {
            padding: 10px;
            font-size: 16px;
            margin-top: 8px;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            margin-right: 8px;
        }

        button:hover {
            background: #1d4ed8;
        }

        .random-btn {
            background: #059669;
        }

        .random-btn:hover {
            background: #047857;
        }

        .note {
            color: #6b7280;
            font-size: 14px;
        }

        .warning {
            background: #fff7ed;
            color: #9a3412;
            border: 1px solid #fed7aa;
            padding: 12px;
            border-radius: 8px;
            margin-top: 12px;
        }

        .good {
            background: #ecfdf5;
            color: #065f46;
            border: 1px solid #a7f3d0;
            padding: 12px;
            border-radius: 8px;
            margin-top: 12px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
        }

        .plot-img {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: white;
        }

        .img-title {
            font-weight: bold;
            margin-top: 8px;
        }

        @media (max-width: 850px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
<div class="container">
    <h1>Bài 28 - Few-shot Learning to Learn</h1>

    <p>
        Đề tài xây dựng và so sánh 3 mô hình trên dataset CIFAR10:
        <b>ANN</b>, <b>LSTM</b>, và <b>Few-shot Learning to Learn bằng LSTM Encoder</b>.
    </p>

    <div class="card">
        <h2>Dự đoán ảnh CIFAR10 có sẵn</h2>

        <form method="get">
            <label>Nhập index ảnh test từ 0 đến 9999:</label><br>
            <input type="number" name="index" min="0" max="9999" value="{{ index }}">
            <button type="submit">Dự đoán</button>
            <button class="random-btn" type="submit" name="random" value="1">Lấy ảnh test ngẫu nhiên</button>
        </form>

        <p class="note">
            Web dùng trực tiếp tập test CIFAR10 có sẵn trong TensorFlow, không cần upload ảnh ngoài.
        </p>

        <div class="image-box">
            <img src="data:image/png;base64,{{ image_base64 }}">
            <h3>Index: {{ index }} | Nhãn thật: {{ result.true_label }}</h3>
        </div>

        <table>
            <tr>
                <th>Mô hình</th>
                <th>Dự đoán</th>
                <th>Độ tin cậy</th>
                <th>Nhận xét nhanh</th>
            </tr>

            <tr>
                <td>ANN</td>
                <td>{{ result.ann[0] if result.ann else "Chưa train ANN" }}</td>
                <td>{{ "%.4f"|format(result.ann[1]) if result.ann else "-" }}</td>
                <td>
                    ANN dùng Flatten nên dễ mất thông tin không gian của ảnh.
                    Kết quả thường thấp hơn LSTM/CNN.
                </td>
            </tr>

            <tr>
                <td>LSTM</td>
                <td>{{ result.lstm[0] if result.lstm else "Chưa train LSTM" }}</td>
                <td>{{ "%.4f"|format(result.lstm[1]) if result.lstm else "-" }}</td>
                <td>
                    LSTM xem ảnh như chuỗi 32 dòng, mỗi dòng có 96 đặc trưng.
                    Kết quả hiện tại ổn hơn ANN.
                </td>
            </tr>

            <tr>
                <td>Few-shot LSTM Encoder</td>
                <td>{{ result.fewshot[0] if result.fewshot else "Chưa train Few-shot" }}</td>
                <td>{{ "%.4f"|format(result.fewshot[1]) if result.fewshot else "-" }}</td>
                <td>
                    Few-shot dùng prototype classification.
                    Nếu accuracy quanh 0.2 với bài 5-way thì mô hình đang gần mức đoán ngẫu nhiên.
                </td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>Siêu tham số chi tiết</h2>

        <table>
            <tr>
                <th>Thông tin</th>
                <th>ANN</th>
                <th>LSTM</th>
                <th>Few-shot Learning to Learn</th>
            </tr>

            <tr>
                <td>Dataset</td>
                <td>CIFAR10</td>
                <td>CIFAR10</td>
                <td>CIFAR10</td>
            </tr>

            <tr>
                <td>Số lớp</td>
                <td>10 lớp</td>
                <td>10 lớp</td>
                <td>Episode 5-way, chọn 5 lớp mỗi episode</td>
            </tr>

            <tr>
                <td>Input ban đầu</td>
                <td>Ảnh 32 x 32 x 3</td>
                <td>Ảnh 32 x 32 x 3</td>
                <td>Ảnh 32 x 32 x 3</td>
            </tr>

            <tr>
                <td>Biểu diễn input</td>
                <td>Flatten ảnh thành vector 3072 chiều</td>
                <td>32 time steps x 96 features</td>
                <td>32 time steps x 96 features</td>
            </tr>

            <tr>
                <td>Tiền xử lý</td>
                <td>Chuẩn hóa pixel về [0, 1], one-hot label</td>
                <td>Chuẩn hóa pixel về [0, 1], reshape sequence, one-hot label</td>
                <td>Chuẩn hóa pixel về [0, 1], tạo episode N-way K-shot</td>
            </tr>

            <tr>
                <td>Kiến trúc</td>
                <td>
                    Flatten<br>
                    Dense 512 ReLU<br>
                    Dropout 0.3<br>
                    Dense 256 ReLU<br>
                    Dropout 0.3<br>
                    Dense 128 ReLU<br>
                    Dense 10 Softmax
                </td>
                <td>
                    LSTM 128, return_sequences=True<br>
                    Dropout 0.3<br>
                    LSTM 64<br>
                    Dropout 0.3<br>
                    Dense 64 ReLU<br>
                    Dense 10 Softmax
                </td>
                <td>
                    LSTM Encoder 128<br>
                    Dropout 0.3<br>
                    LSTM Encoder 64<br>
                    Dropout 0.3<br>
                    Dense 64 ReLU<br>
                    UnitNormalization<br>
                    Prototype Classification
                </td>
            </tr>

            <tr>
                <td>Epoch / Episode</td>
                <td>20 epochs</td>
                <td>20 epochs</td>
                <td>1000 episodes</td>
            </tr>

            <tr>
                <td>Batch size</td>
                <td>64</td>
                <td>64</td>
                <td>
                    Không dùng batch truyền thống.<br>
                    Mỗi episode gồm support set và query set.
                </td>
            </tr>

            <tr>
                <td>Few-shot setting</td>
                <td>Không áp dụng</td>
                <td>Không áp dụng</td>
                <td>
                    N-way = 5<br>
                    K-shot = 5<br>
                    Query mỗi lớp = 10
                </td>
            </tr>

            <tr>
                <td>Optimizer</td>
                <td>Adam</td>
                <td>Adam</td>
                <td>Adam</td>
            </tr>

            <tr>
                <td>Learning rate</td>
                <td>Mặc định Adam: 0.001</td>
                <td>Mặc định Adam: 0.001</td>
                <td>0.001</td>
            </tr>

            <tr>
                <td>Loss function</td>
                <td>Categorical Crossentropy</td>
                <td>Categorical Crossentropy</td>
                <td>Sparse Categorical Crossentropy from logits</td>
            </tr>

            <tr>
                <td>Metric</td>
                <td>Accuracy</td>
                <td>Accuracy</td>
                <td>Episode Accuracy</td>
            </tr>

            <tr>
                <td>File model</td>
                <td>saved_models/ann_cifar10.keras</td>
                <td>saved_models/lstm_cifar10.keras</td>
                <td>saved_models/fewshot_lstm_encoder.keras</td>
            </tr>
        </table>
    </div>

    <div class="card">
        <h2>Đánh giá biểu đồ hiện tại</h2>

        <div class="good">
            <b>ANN:</b> Biểu đồ accuracy tăng dần và loss giảm dần. Model đã học được nhưng accuracy còn thấp,
            phù hợp vì ANN flatten ảnh nên không giữ tốt đặc trưng không gian.
        </div>

        <div class="good">
            <b>LSTM:</b> Accuracy train tăng lên khoảng trên 0.65, validation khoảng gần 0.60.
            Loss train giảm tốt, validation loss giảm rồi đi ngang. Đây là kết quả ổn cho LSTM trên CIFAR10.
            Chưa cần train lại LSTM.
        </div>

        <div class="warning">
            <b>Few-shot:</b> Loss dao động quanh 1.61 và accuracy quanh 0.2.
            Với bài 5-way, đoán ngẫu nhiên cũng khoảng 0.2. Vì vậy Few-shot hiện tại chưa học tốt.
            Nếu cần báo cáo đẹp, nên train lại Few-shot hoặc cải tiến encoder.
        </div>
    </div>

    <div class="card">
        <h2>Biểu đồ kết quả</h2>

        <div class="grid">
            {% for img in result_images %}
            <div>
                <img class="plot-img" src="/results/{{ img }}">
                <p class="img-title">{{ img }}</p>
            </div>
            {% endfor %}
        </div>
    </div>
</div>
</body>
</html>
"""


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULT_DIR, filename)


@app.route("/")
def index():
    if request.args.get("random") == "1":
        index_value = random.randint(0, 9999)
    else:
        index_value = int(request.args.get("index", 0))

    if index_value < 0:
        index_value = 0

    if index_value > 9999:
        index_value = 9999

    image, result = predict_test_index(index_value)
    image_base64 = image_to_base64(image)

    result_images = []

    if os.path.exists(RESULT_DIR):
        for filename in os.listdir(RESULT_DIR):
            if filename.endswith(".png"):
                result_images.append(filename)

    result_images = sorted(result_images)

    return render_template_string(
        HTML,
        index=index_value,
        image_base64=image_base64,
        result=result,
        result_images=result_images
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)