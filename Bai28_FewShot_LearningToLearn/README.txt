BÀI 28 - FEW-SHOT LEARNING TO LEARN

1. MỤC TIÊU
Bài này xây dựng hệ thống học sâu trên dataset CIFAR10, gồm 3 mô hình:
- ANN
- LSTM
- Few-shot Learning to Learn bằng LSTM Encoder

Mục tiêu là so sánh cách học thông thường với cách học few-shot.
ANN và LSTM được huấn luyện theo kiểu phân lớp truyền thống.
Few-shot model được huấn luyện theo từng episode N-way K-shot.

2. DATASET
Dataset sử dụng: CIFAR10 có sẵn trong TensorFlow.

CIFAR10 gồm:
- 50,000 ảnh train
- 10,000 ảnh test
- Kích thước ảnh: 32 x 32 x 3
- 10 lớp:
  0 airplane
  1 automobile
  2 bird
  3 cat
  4 deer
  5 dog
  6 frog
  7 horse
  8 ship
  9 truck

3. CẤU TRÚC THƯ MỤC

Bai28_FewShot_LearningToLearn/
├── report/
├── results/
├── saved_models/
├── src/
│   ├── utils.py
│   ├── train_ann.py
│   ├── train_lstm.py
│   ├── train_fewshot.py
│   └── predict.py
├── app.py
├── README.txt
└── requirements.txt

4. CÀI ĐẶT THƯ VIỆN

Mở terminal trong thư mục Bai28_FewShot_LearningToLearn và chạy:

pip install -r requirements.txt

5. THỨ TỰ CHẠY CODE

Bước 1: Train ANN

python src/train_ann.py

Sau khi chạy xong sẽ tạo:
- saved_models/ann_cifar10.keras
- results/ann_accuracy.png
- results/ann_loss.png
- results/history_ann.csv
- results/model_results.csv

Bước 2: Train LSTM

python src/train_lstm.py

Sau khi chạy xong sẽ tạo:
- saved_models/lstm_cifar10.keras
- results/lstm_accuracy.png
- results/lstm_loss.png
- results/history_lstm.csv

Bước 3: Train Few-shot Learning to Learn

python src/train_fewshot.py

Sau khi chạy xong sẽ tạo:
- saved_models/fewshot_lstm_encoder.keras
- results/fewshot_accuracy.png
- results/fewshot_loss.png
- results/history_fewshot.csv

Bước 4: Chạy web demo Flask

python app.py

Sau đó mở trình duyệt tại:

http://127.0.0.1:5000

6. GIẢI THÍCH MÔ HÌNH

6.1. ANN
ANN nhận ảnh CIFAR10 đầu vào kích thước 32x32x3.
Ảnh được làm phẳng bằng Flatten thành vector.
Sau đó đi qua các tầng Dense để phân loại thành 10 lớp.

Kiến trúc:
Flatten
Dense 512 ReLU
Dropout 0.3
Dense 256 ReLU
Dropout 0.3
Dense 128 ReLU
Dense 10 Softmax

6.2. LSTM
LSTM thường dùng cho dữ liệu chuỗi.
Trong bài này, ảnh CIFAR10 được xem như chuỗi gồm 32 dòng ảnh.
Mỗi dòng ảnh có 32 pixel và 3 kênh màu nên có 96 đặc trưng.

Biểu diễn ảnh:
32 x 32 x 3 -> 32 time steps x 96 features

Kiến trúc:
LSTM 128 return_sequences=True
Dropout 0.3
LSTM 64
Dropout 0.3
Dense 64 ReLU
Dense 10 Softmax

6.3. Few-shot Learning to Learn
Few-shot model học theo từng episode.
Mỗi episode chọn N lớp, mỗi lớp có K ảnh support và một số ảnh query.

Cấu hình:
N-way = 5
K-shot = 5
Query mỗi lớp = 10

Quy trình:
- LSTM Encoder biến ảnh thành embedding vector.
- Với mỗi lớp, tính prototype bằng trung bình embedding của support set.
- Ảnh query được phân loại theo prototype gần nhất.
- Model học cách tạo embedding tốt để thích nghi với dữ liệu ít mẫu.

7. GỢI Ý KHI MÁY YẾU

Nếu train lâu, có thể giảm trong các file train:

EPOCHS = 20

thành:

EPOCHS = 5

Với few-shot, có thể giảm:

EPISODES = 1000

thành:

EPISODES = 300

8. FILE KẾT QUẢ

Thư mục results chứa:
- cifar10_samples.png
- ann_accuracy.png
- ann_loss.png
- lstm_accuracy.png
- lstm_loss.png
- fewshot_accuracy.png
- fewshot_loss.png
- history_ann.csv
- history_lstm.csv
- history_fewshot.csv
- model_results.csv

Thư mục saved_models chứa:
- ann_cifar10.keras
- lstm_cifar10.keras
- fewshot_lstm_encoder.keras

9. NHẬN XÉT BÁO CÁO

ANN là mô hình cơ bản, xử lý ảnh bằng cách flatten toàn bộ pixel thành vector nên dễ mất thông tin không gian.

LSTM xử lý ảnh như một chuỗi các dòng ảnh. Cách này giúp mô hình học quan hệ tuần tự giữa các dòng ảnh, tuy nhiên ảnh không phải dữ liệu chuỗi tự nhiên nên kết quả có thể không cao bằng CNN.

Few-shot Learning to Learn không chỉ học phân loại trực tiếp mà học cách tạo embedding để phân biệt lớp mới từ ít mẫu. Mô hình này phù hợp khi dữ liệu huấn luyện cho mỗi lớp bị hạn chế.

10. LỆNH CHẠY NHANH

pip install -r requirements.txt
python src/train_ann.py
python src/train_lstm.py
python src/train_fewshot.py
python app.py