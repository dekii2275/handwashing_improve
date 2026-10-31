# Dataset Directory

Thư mục chứa dữ liệu của bài toán **Nhận dạng động tác rửa tay (Hand-Washing Recognition)**.

## Hướng dẫn tải và chuẩn bị dữ liệu

Bộ dữ liệu METC (DOI: `10.5281/zenodo.5808789`):
1. **Tải tệp metadata & toàn bộ nhãn JSON (nhẹ, nhanh):**
   ```bash
   python src/download_dataset.py --only-annotations
   ```
2. **Tải toàn bộ video raw từ Zenodo (~2.02 GB):**
   ```bash
   python src/download_dataset.py --download-all
   ```
3. **Phân chia tập dữ liệu (70% Train / 10% Val / 20% Test):**
   ```bash
   python src/split_data.py
   ```
4. **Trích xuất MediaPipe Hand Skeleton Landmarks:**
   ```bash
   python src/batch_extract.py --workers 4
   ```
