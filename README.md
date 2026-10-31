# Hand-Washing Movement Recognition Based on Hand Skeleton Landmarks

Tái lập toàn diện nghiên cứu khoa học:
> **"Hand-Washing Movement Recognition Based on Hand Skeleton Landmarks"**  
> **Tác giả:** Maksims Ivanovs và Ilze Ivanova (University of Latvia)  
> **Công bố tại:** Baltic DB&IS 2026, CEUR-WS Vol. 4211, trang 55–65.

---

## 1. Giới thiệu bài toán & Mục tiêu

Nhận dạng hành vi rửa tay theo chuẩn của Tổ chức Y tế Thế giới (WHO) đóng vai trò quyết định trong việc giảm thiểu nhiễm khuẩn bệnh viện (HAIs). Các phương pháp truyền thống xử lý trực tiếp toàn bộ ảnh RGB qua mạng tích chập thường bị nhiễu do bọt xà phòng, thay đổi ánh sáng, màu da và bồn rửa.

Nghiên cứu này đề xuất chuyển đổi video RGB sang **chuỗi điểm khớp xương bàn tay (Skeleton Landmarks)** sử dụng **MediaPipe Hands**:
- Trích xuất 21 điểm khớp 3D cho mỗi bàn tay ($2 \times 21 \times 3 = 126$ tọa độ mỗi khung hình).
- Xử lý mất dấu (missing hand detection) bằng phương pháp nội suy tuyến tính theo thời gian (Linear Imputation).
- Chuẩn hóa không gian: Centering theo cổ tay (`norm_loc`) hoặc kết hợp chuẩn hóa tỷ lệ (`norm_zoom`).
- Xây dựng chuỗi cửa sổ trượt (sliding window) với độ dài $T \in \{15, 24, 30, 48\}$ khung hình và bước trượt $stride = 1$.
- Phân loại thành **7 lớp động tác** (Class 0: Chuyển động khác; Class 1–6: 6 bước rửa tay chuẩn WHO).
- So sánh 3 kiến trúc học sâu: **1D Temporal CNN**, **GRU**, và **LSTM**.

---

## 2. Kết quả công bố trong bài báo

| Kiến trúc mô hình | Test Accuracy (%) | Test Macro F1-score (%) | Cấu hình tối ưu trong bài báo |
| :--- | :---: | :---: | :--- |
| **LSTM** | 65,9% | 65,7% | $T=30$, Raw landmarks, Imputation=True, Hidden=64, 2 layers |
| **GRU** | 69,7% | 68,1% | $T=48$, Norm Zoom, Imputation=False, Hidden=128, 2 layers |
| **1D Temporal CNN** | **70,3%** | **70,5%** | **$T=48$, Norm Loc, Imputation=True, Filters=64, Kernel=3, 2 layers** |

---

## 3. Cấu trúc thư mục dự án

```text
handwashing_improve/
├── artifacts/                    # Manifests phân chia và báo cáo trung gian
├── checkpoints/                  # Trọng số checkpoint mô hình tốt nhất (best_model_temporal_cnn.pt)
├── configs/                      # File cấu hình YAML cho các thực nghiệm
├── data/                         # Thư mục dữ liệu (raw videos, npz landmarks, split metadata)
│   ├── raw/                      # Video METC & tệp JSON gán nhãn
│   ├── landmarks/                # Tọa độ điểm khớp (.npz) trích xuất bằng MediaPipe
│   ├── processed/                # Dữ liệu chuỗi sau chuẩn hóa
│   ├── split_info.json           # Phân chia 70% Train / 10% Val / 20% Test theo video
│   └── README.md                 # Hướng dẫn chi tiết tải & tiền xử lý dữ liệu
├── docs/                         # Tài liệu gốc & bài báo nghiên cứu (paper06.pdf)
├── models/                       # Kết quả thực nghiệm mở rộng & reproduction summary
├── results/                      # Báo cáo phân loại, ma trận nhầm lẫn và biểu đồ huấn luyện
├── scripts/                      # Các scripts hỗ trợ chuẩn bị pipeline và thực nghiệm
├── src/                          # Mã nguồn chính của dự án
│   ├── __init__.py
│   ├── download_dataset.py       # Tải dataset METC từ Zenodo (HTTP Range request & resume)
│   ├── split_data.py             # Chia tập dữ liệu 70/10/20 ở cấp độ video
│   ├── extract_landmarks.py      # Trích xuất MediaPipe Hands 3D landmarks
│   ├── batch_extract.py          # Trích xuất song song nhiều luồng (multiprocessing)
│   ├── features.py               # Tiền xử lý: Imputation, Chuẩn hóa vị trí/tỷ lệ, Bone vectors, Pairwise dists
│   ├── dataset.py                # PyTorch Dataset cửa sổ trượt (sliding window T, stride 1)
│   ├── models.py                 # Định nghĩa 1D Temporal CNN, GRU, LSTM theo chuẩn bài báo
│   ├── train.py                  # Vòng lặp huấn luyện Adam, lr=1e-3, batch=256, val macro F1 checkpointing
│   ├── evaluate.py               # Đánh giá trên tập kiểm thử (Test set), sinh classification report
│   ├── visualize.py              # Vẽ loss/accuracy/F1 curve và confusion matrix
│   └── pipeline.py               # Pipeline liên kết toàn bộ quy trình đầu-cuối
├── tests/
│   └── test_components.py        # Bộ unit test kiểm thử toàn diện các module
├── config.py                     # Cấu hình tập trung siêu tham số & đường dẫn
├── run_experiments.py            # CLI chạy thực nghiệm và so sánh kết quả
├── requirements.txt              # Danh sách thư viện phụ thuộc chuẩn hóa
└── README.md
```

---

## 4. Cài đặt & Môi trường

Khuyến nghị sử dụng Python 3.10 – 3.14:

```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo:
# Trên Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Trên Linux / macOS:
source venv/bin/activate

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

---

## 5. Hướng dẫn sử dụng

### 5.1 Tải bộ dữ liệu METC (DOI: 10.5281/zenodo.5808789)
```bash
# Tải tệp thống kê CSV và toàn bộ nhãn JSON của 212 video (nhẹ và nhanh)
python src/download_dataset.py --only-annotations

# Tải toàn bộ video archives (~2.02 GB) từ Zenodo
python src/download_dataset.py --download-all
```

### 5.2 Phân tách dữ liệu Train / Val / Test (70% / 10% / 20%)
```bash
python src/split_data.py
```

### 5.3 Trích xuất Landmarks hàng loạt bằng MediaPipe
```bash
python src/batch_extract.py --workers 4
```

### 5.4 Chạy thử nghiệm mô hình tốt nhất (1D Temporal CNN)
```bash
# Chạy mô hình 1D Temporal CNN (cấu hình tối ưu: T=48, norm_loc, imputation)
python run_experiments.py --model temporal_cnn --epochs 100
```

### 5.5 Chạy so sánh toàn bộ 3 kiến trúc (1D CNN vs GRU vs LSTM)
```bash
python run_experiments.py --model all --epochs 100
```

### 5.6 Chạy kiểm tra nhanh (Quick test với số epoch nhỏ hoặc tập con video)
```bash
python run_experiments.py --model temporal_cnn --epochs 5 --max-videos 10
```

---

## 6. Chạy Kiểm Thử (Unit Tests)

Dự án bao gồm bộ test kiểm tra tự động kiến trúc mô hình, tính đúng đắn của phép nội suy imputation, chuẩn hóa không gian và cửa sổ trượt:

```bash
pytest tests/ -v
```

---

## 7. Giấy phép (License)

Dự án được phân phối theo giấy phép [Apache 2.0](LICENSE).
