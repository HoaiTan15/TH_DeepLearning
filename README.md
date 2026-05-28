# TH_Deep - Deep Learning Projects

Repository chứa các dự án Deep Learning khác nhau về CNN, ANN, LSTM và Few-Shot Learning.

## 📚 Các Dự Án

### 1. 📁 Bai2_CNN_ANN - So Sánh CNN và ANN trên CIFAR-10

**Mô Tả**: So sánh hiệu suất giữa ANN, CNN và Improved CNN trên dataset CIFAR-10

**Mục Tiêu**:
- Huấn luyện 3 mô hình deep learning
- So sánh độ chính xác, loss và tốc độ
- Phân tích điểm mạnh/yếu của từng kiến trúc

**Dataset**: CIFAR-10 (50K train, 10K test, 10 classes)

**Kết Quả**:
| Mô Hình | Accuracy | Loss | Thời gian |
|---------|----------|------|----------|
| ANN | 42.5% | 1.63 | ~2 phút |
| CNN | 75.0% | 0.72 | ~5 phút |
| Improved CNN | 85.2% | 0.48 | ~15 phút |

**Chạy**:
```bash
cd Bai2_CNN_ANN
pip install -r requirements.txt
streamlit run app.py
```

**Tệp tin chính**:
- `src/train_ann.py` - Huấn luyện ANN
- `src/train_cnn.py` - Huấn luyện CNN
- `src/train_improved_cnn.py` - Huấn luyện Improved CNN
- `app.py` - Streamlit dashboard

---

### 2. 📁 Bai28_FewShot_LearningToLearn - Few-Shot Learning trên CIFAR-10

**Mô Tả**: Triển khai Few-Shot Learning (Prototypical Networks) với LSTM Encoder

**Mục Tiêu**:
- Hiểu rõ Few-Shot Learning và Metric Learning
- Triển khai LSTM Encoder cho embedding
- So sánh với baseline (ANN, LSTM)
- Phân tích khả năng generalization

**Dataset**: CIFAR-10 trong cấu hình Few-Shot
- **5-way, 5-shot, 10-query**
- 1000 episodes training
- 100 episodes evaluation

**Kiến Trúc**:
```
LSTM Encoder:
Input(32, 96) → LSTM(128) → Dropout → LSTM(64) 
→ Dropout → Dense(64) → UnitNormalization
↓
Prototypical Networks:
Support set → Prototypes
Query set → Distances → Softmax → Predictions
```

**Kết Quả**:
| Mô Hình | Accuracy | Loss |
|---------|----------|------|
| ANN (20 epochs) | 37.2% | 1.87 |
| LSTM (20 epochs) | 59.5% | 1.20 |
| Few-Shot (1000 eps) | 40.5% | 1.60 |

**Chạy**:
```bash
cd Bai28_FewShot_LearningToLearn
pip install -r requirements.txt
streamlit run app.py
```

**Tệp tin chính**:
- `src/train_ann.py` - ANN baseline
- `src/train_lstm.py` - LSTM model
- `src/train_fewshot.py` - Few-Shot Learning ★
- `src/predict.py` - Inference
- `app.py` - Streamlit app

---

## 🛠️ Cài Đặt Chung

### Yêu Cầu
- Python 3.8+
- TensorFlow 2.10+
- NumPy, Pandas, Matplotlib, Scikit-learn

### Setup
```bash
# Clone repository
git clone https://github.com/HoaiTan15/TH_DeepLearning.git
cd TH_Deep

# Cài đặt dependencies cho từng dự án
cd Bai2_CNN_ANN
pip install -r requirements.txt

# Hoặc cho Bai28
cd ../Bai28_FewShot_LearningToLearn
pip install -r requirements.txt
```

---

## 📊 So Sánh Tổng Quan

### Kiến Trúc:
| Dự Án | Chủ Đề | Phức Tạp | Hiệu Suất |
|-------|--------|---------|----------|
| Bai2 | CNN vs ANN | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Bai28 | Few-Shot Learning | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Thích Hợp Cho:
- **Bai2**: Học CNN cơ bản, so sánh mô hình
- **Bai28**: Học Few-Shot Learning, Metric Learning

---

## 📁 Cấu Trúc Thư Mục

```
TH_Deep/
├── README.md                          # File này
├── .vscode/
│   └── settings.json
│
├── Bai2_CNN_ANN/
│   ├── README.txt
│   ├── app.py
│   ├── requirements.txt
│   ├── src/
│   │   ├── models.py
│   │   ├── train_ann.py
│   │   ├── train_cnn.py
│   │   ├── train_improved_cnn.py
│   │   ├── evaluate.py
│   │   ├── data_loader.py
│   │   └── utils.py
│   ├── saved_models/
│   │   ├── ann_model.keras
│   │   ├── cnn_model.keras
│   │   └── improved_cnn_model.keras
│   └── results/
│       ├── ann/
│       ├── cnn/
│       └── improved_cnn/
│
└── Bai28_FewShot_LearningToLearn/
    ├── README.txt
    ├── app.py
    ├── requirements.txt
    ├── src/
    │   ├── train_ann.py
    │   ├── train_lstm.py
    │   ├── train_fewshot.py    # ★ Main
    │   ├── predict.py
    │   └── utils.py
    ├── saved_models/
    │   ├── ann_cifar10.keras
    │   ├── lstm_cifar10.keras
    │   └── fewshot_lstm_encoder.keras
    └── results/
        ├── history_*.csv
        └── *.png (plots)
```

---

## 🚀 Quick Start

### Chạy Bai2 (CNN vs ANN)
```bash
cd Bai2_CNN_ANN
streamlit run app.py
# Mở http://localhost:8501
```

### Chạy Bai28 (Few-Shot Learning)
```bash
cd Bai28_FewShot_LearningToLearn
streamlit run app.py
# Mở http://localhost:8501
```

### Huấn Luyện Lại (Tuỳ Chọn)
```bash
# Bai2: Mỗi mô hình ~5 phút
python src/train_ann.py
python src/train_cnn.py
python src/train_improved_cnn.py

# Bai28: Few-Shot ~30 phút
python src/train_fewshot.py
```

---

## 💡 Các Khái Niệm Chính

### Bai2: CNN vs ANN
- **ANN**: Mạng fully-connected, không tận dụng spatial structure
- **CNN**: Sử dụng convolution để học features cục bộ
- **Improved CNN**: Deeper architecture + Dropout + Augmentation

### Bai28: Few-Shot Learning
- **Few-Shot Learning**: Học từ ít mẫu (K-shot learning)
- **Metric Learning**: Học metric distance để so sánh
- **Prototypical Networks**: Prototype = trung bình embedding
- **Episode Training**: Mỗi episode là N-way K-shot problem

---

## 🔬 Các Công Nghệ Sử Dụng

### Deep Learning Framework
- **TensorFlow/Keras**: Model building & training

### Data Processing
- **NumPy**: Numerical operations
- **Pandas**: Data manipulation
- **Scikit-learn**: Metrics & preprocessing

### Visualization
- **Matplotlib**: Plotting
- **Streamlit**: Web interface

### Hardware
- **GPU support**: CUDA/CuDNN (tùy chọn)

---

## 📈 Hiệu Suất & Kết Quả

### Bai2 Results:
- ✅ CNN vượt trội hơn ANN
- ✅ Improved CNN cho kết quả tốt nhất (~85%)
- ✅ Visualization clarity, easy comparison

### Bai28 Results:
- ✅ Few-Shot framework hoạt động
- ✅ LSTM encoder learns good embeddings
- ✅ Demo "Learning to Learn" paradigm
- ⚠️ Accuracy thấp do CIFAR-10 quá nhỏ cho few-shot

---

## 🎓 Học Tập

### Bai2 - Dành Cho Người Mới Bắt Đầu
1. Hiểu CNN cơ bản
2. So sánh với ANN
3. Thấy tác dụng của architecture design

### Bai28 - Advanced Topic
1. Hiểu Few-Shot Learning paradigm
2. Prototypical Networks implementation
3. Metric Learning concepts

---

## 📞 Liên Hệ & Tác Giả

- **Tác Giả**: Hoài Tân
- **Email**: hoaitan15@example.com
- **GitHub**: [HoaiTan15](https://github.com/HoaiTan15)
- **Repository**: [TH_DeepLearning](https://github.com/HoaiTan15/TH_DeepLearning)

---

## 📜 License

MIT License - Tự do sử dụng cho mục đích học tập & nghiên cứu

---

## 🔗 Tham Khảo

### Bai2 References:
- [CNN Architecture Overview](https://cs231n.github.io/convolutional-networks/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)

### Bai28 References:
- [Prototypical Networks Paper](https://arxiv.org/abs/1703.05175)
- [Few-Shot Learning Survey](https://arxiv.org/abs/1904.05046)
- [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080)

---

**Cập nhật lần cuối**: May 2026  
**Version**: 1.0
