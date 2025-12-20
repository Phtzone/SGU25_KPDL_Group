# 🚀 Credit Card Customer Segmentation - Deployment Guide

Web application để phân khúc khách hàng thẻ tín dụng sử dụng K-Means Clustering.

## 📁 Cấu trúc thư mục

```
Deploy/
├── segmentation_model.py          # Module chứa preprocessing + KMeans
├── streamlit_app/
│   └── app.py                     # Streamlit web application
├── .streamlit/
│   └── config.toml                # Streamlit theme configuration
├── model_artifacts/
│   └── credit_segmentation_k4.joblib  # Model artifact (generated)
├── requirements.txt               # Python dependencies
└── README.md                      # Hướng dẫn này
```

## 🔧 Yêu cầu hệ thống

- Python 3.9+
- Các thư viện: xem `requirements.txt`

## 📦 Bước 1: Export Model Artifact

Chạy các cell cuối trong notebook `final project.ipynb` để:

1. Import `SegmentationModel`
2. Train model trên dữ liệu đầy đủ
3. Gắn `cluster_names` (persona names)
4. Lưu vào `Deploy/model_artifacts/credit_segmentation_k4.joblib`

```python
# Cell trong notebook
from segmentation_model import SegmentationModel

k = 4
model = SegmentationModel(k=k, random_state=42, n_init=50)
model.fit(df)
model.cluster_names = cluster_names  # từ auto-naming
model.save("Deploy/model_artifacts/credit_segmentation_k4.joblib")
```

## 🏃 Bước 2: Chạy local

### Windows (PowerShell)

```powershell
# Di chuyển vào thư mục Deploy
cd "d:\Temp Github\2611\SGU25_KPDL_Group\final project\Deploy"

# Tạo virtual environment (nếu chưa có)
python -m venv .venv

# Kích hoạt venv
.\.venv\Scripts\activate

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy Streamlit app
streamlit run streamlit_app\app.py
```

### Linux/macOS

```bash
cd "/path/to/final project/Deploy"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

App sẽ mở tại: **http://localhost:8501**

## ☁️ Bước 3: Deploy lên Streamlit Community Cloud

### Chuẩn bị

1. Push code lên GitHub repository
2. Đảm bảo có các file:
   - `final project/Deploy/segmentation_model.py`
   - `final project/Deploy/streamlit_app/app.py`
   - `final project/Deploy/model_artifacts/credit_segmentation_k4.joblib`
   - **`requirements.txt`** (ở repo root - quan trọng!)

### Deploy

1. Truy cập [share.streamlit.io](https://share.streamlit.io)
2. Đăng nhập bằng GitHub
3. Click **New app**
4. Chọn:
   - **Repository**: `<your-username>/SGU25_KPDL_Group`
   - **Branch**: `main` (hoặc branch của bạn)
   - **Main file path**: `final project/Deploy/streamlit_app/app.py`
5. Click **Deploy!**

Streamlit Cloud sẽ tự động:
- Cài đặt dependencies từ `requirements.txt`
- Chạy app tại public URL
- Auto-redeploy khi có commit mới

## 🎯 Cách sử dụng App

1. **Upload CSV**: Click "Chọn file CSV" và upload dữ liệu khách hàng
   - Format giống `CC GENERAL.csv` (các cột như `BALANCE`, `PURCHASES`, `CREDIT_LIMIT`,...)

2. **Xem kết quả**:
   - Biểu đồ phân phối cluster
   - Bảng kết quả với cột `Cluster` và `Persona`
   - Thống kê chi tiết

3. **Download**: Click "Download CSV" để tải kết quả với cluster assignments

4. **Chiến lược Marketing**: Xem gợi ý chiến lược cho từng persona

## 🔍 Tính năng

- ✅ **Upload CSV** và tự động preprocessing
- ✅ **Predict cluster** với model đã train
- ✅ **Hiển thị persona names** thay vì chỉ số cluster
- ✅ **Visualization**: Biểu đồ phân phối, statistics
- ✅ **Download kết quả** dạng CSV
- ✅ **Marketing strategies** cho từng persona
- ✅ **Responsive UI** với Streamlit

## 🛠️ Troubleshooting

### Lỗi: "Không tìm thấy file artifact"

```python
# Trong notebook, chạy lại cell export model
model.save("Deploy/model_artifacts/credit_segmentation_k4.joblib")
```

### Lỗi: "ModuleNotFoundError: No module named 'segmentation_model'"

```python
# Đảm bảo file segmentation_model.py nằm trong Deploy/
# Và app.py đã thêm sys.path.insert
```

### Lỗi: "Predict lỗi" khi upload CSV

- Kiểm tra CSV có đủ các cột cần thiết
- Đảm bảo format giống dataset train (`CC GENERAL.csv`)
- Các cột thiếu sẽ được auto-fill với 0, nhưng chất lượng prediction giảm

## 📊 Model Info

- **Algorithm**: K-Means Clustering
- **K clusters**: 4 (hoặc giá trị trong `chosen_k_for_marketing`)
- **Preprocessing**:
  - KNN imputation cho `MINIMUM_PAYMENTS`
  - Winsorization (1%-99% quantiles)
  - Feature engineering (ratios, shares)
  - Log1p transform
  - Feature selection
  - Standard scaling
- **Features**: ~10-12 features sau preprocessing

## 📚 Tài liệu tham khảo

- [Streamlit Documentation](https://docs.streamlit.io)
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Streamlit Community Cloud](https://docs.streamlit.io/streamlit-community-cloud)

## 👥 Team

Xem thông tin team trong notebook `final project.ipynb`

## 📝 License

Educational project - SGU 2025

---

**Last updated**: December 20, 2025
