ĐỒ ÁN 2: PHÂN LOẠI ẢNH CIFAR-10 BẰNG ANN, CNN VÀ CNN CẢI TIẾN

1. Mô tả đề tài
Dự án xây dựng 3 mô hình Deep Learning để phân loại ảnh trong bộ dữ liệu CIFAR-10:

- ANN baseline
- CNN baseline
- Improved CNN

Bộ dữ liệu CIFAR-10 gồm 10 lớp:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

Ứng dụng demo được xây dựng bằng Streamlit, sử dụng trực tiếp tập test CIFAR-10 để dự đoán ảnh và so sánh nhãn thật với nhãn dự đoán.


2. Cấu trúc thư mục

Bai2_CNN_ANN/
│
├── app.py
├── requirements.txt
├── README.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── data_loader.py
│   ├── train_ann.py
│   ├── train_cnn.py
│   ├── train_improved_cnn.py
│   ├── evaluate.py
│   └── utils.py
│
├── saved_models/
│   ├── ann_model.keras
│   ├── cnn_model.keras
│   └── improved_cnn_model.keras
│
├── results/
│   ├── ann/
│   ├── cnn/
│   └── improved_cnn/
│
└── report/


3. Yêu cầu môi trường

Khuyến nghị sử dụng:

- Python 3.10
- TensorFlow
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pillow


4. Cài đặt thư viện

Mở terminal tại thư mục Bai2_CNN_ANN và chạy:

pip install -r requirements.txt

Nếu chưa có requirements.txt, có thể cài trực tiếp:

pip install tensorflow streamlit pillow numpy pandas matplotlib seaborn scikit-learn


5. Huấn luyện mô hình

Huấn luyện ANN:

python src\train_ann.py

Huấn luyện CNN:

python src\train_cnn.py

Huấn luyện Improved CNN:

python src\train_improved_cnn.py


6. Đánh giá mô hình

Sau khi train xong, có thể chạy lại đánh giá để tạo confusion matrix và classification report:

python src\evaluate.py --model ann
python src\evaluate.py --model cnn
python src\evaluate.py --model improved_cnn


7. Chạy ứng dụng demo web

Chạy lệnh:

streamlit run app.py

Sau đó trình duyệt sẽ mở giao diện demo.

Ứng dụng gồm các tab:

- Dự đoán ảnh CIFAR-10
- Kết quả huấn luyện
- Thông tin mô hình
- Siêu tham số


8. Kết quả mô hình

Kết quả thực nghiệm tham khảo:

- ANN: khoảng 44%
- CNN: khoảng 75%
- Improved CNN: khoảng 80%

Improved CNN cho kết quả tốt nhất nhờ sử dụng thêm:

- Data Augmentation
- Batch Normalization
- Dropout
- GlobalAveragePooling2D
- EarlyStopping
- ReduceLROnPlateau


9. Ghi chú khi nộp bài

Không cần nộp thư mục môi trường ảo như:

- venv/
- .venv/
- __pycache__/

Chỉ cần nộp:

- source code
- saved_models/
- results/
- README.txt
- requirements.txt
- báo cáo Word/PDF nếu có


10. Tác giả

Sinh viên thực hiện: ................................
Môn học: Deep Learning
Đề tài: Phân loại ảnh CIFAR-10 bằng ANN, CNN và CNN cải tiến