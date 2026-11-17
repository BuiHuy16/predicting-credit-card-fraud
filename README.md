# Predicting Credit Card Fraud

Dự án bài tập lớn môn Học máy nhằm xây dựng, đánh giá và so sánh các mô hình
Machine Learning/Deep Learning cho bài toán phân loại giao dịch thẻ tín dụng.
Đầu ra của mô hình là xác suất `is_fraud`; kết quả dự đoán cuối cùng được đóng
gói thành `submission.csv` theo định dạng của Kaggle Playground Series S6E11.

Đây mới là giai đoạn khởi tạo cấu trúc repository. Repository **chưa** chứa mã
EDA, preprocessing, huấn luyện, mô hình, kết quả thực nghiệm hoặc kết quả giả.

## Dữ liệu

Các tệp dữ liệu hiện có được giữ nguyên tại:

- `datasets/raw/train.csv`
- `datasets/raw/test.csv`
- `datasets/raw/sample_submission.csv`


## Cấu trúc repository

```text
.
├── datasets/
│   ├── raw/                  # Các CSV gốc hiện có, không chỉnh sửa trực tiếp
│   └── processed/            # Dữ liệu đã xử lý sẽ được sinh ở giai đoạn sau (nếu cần)
├── notebooks/                # Notebook khảo sát và thử nghiệm có kiểm soát
├── src/
│   ├── data/                 # Đọc dữ liệu và preprocessing (sẽ triển khai sau)
│   ├── models/               # Định nghĩa mô hình ML/DL (sẽ triển khai sau)
│   ├── training/             # Quy trình huấn luyện (sẽ triển khai sau)
│   ├── evaluation/           # Metric và đánh giá (sẽ triển khai sau)
│   └── prediction/           # Suy luận và tạo submission (sẽ triển khai sau)
├── configs/                  # Cấu hình chạy, không lưu bí mật
├── experiments/              # Ghi chép/metadata cho các lần chạy
├── outputs/
│   ├── models/               # Model và checkpoint sinh tự động
│   ├── figures/              # Biểu đồ sinh tự động
│   └── submissions/          # Các submission.csv sinh tự động
├── tests/                    # Kiểm thử tự động
├── docs/                     # Tài liệu dự án
├── README.md                 # Tổng quan và hướng dẫn sử dụng dự án
├── requirements.txt          # Dependency nền tảng
└── .gitignore                # Quy tắc loại trừ tệp cục bộ/sinh tự động
```

## Thiết lập môi trường

Khuyến nghị Python 3.10 trở lên. Từ thư mục gốc của repository:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Trên macOS/Linux, kích hoạt môi trường bằng:

```bash
source .venv/bin/activate
```

Lệnh cài đặt chỉ là hướng dẫn; giai đoạn khởi tạo này không tự động cài bất kỳ
thư viện nào.

## Quy trình tổng quát

1. **EDA** — kiểm tra schema, chất lượng dữ liệu, phân phối biến và mức mất cân
   bằng của nhãn mà không làm thay đổi dữ liệu gốc.
2. **Preprocessing** — xây dựng quy trình biến đổi có thể tái lập, fit chỉ trên
   tập huấn luyện và tránh rò rỉ dữ liệu.
3. **Training** — huấn luyện các mô hình ML/DL theo cấu hình và seed được ghi
   nhận.
4. **Evaluation** — đánh giá trên chiến lược validation phù hợp bằng metric của
   cuộc thi và các metric bổ trợ đã được nhóm thống nhất.
5. **Prediction** — dự đoán xác suất `is_fraud` trên tập test bằng mô hình đã
   chọn.
6. **Submission** — kiểm tra schema, thứ tự dòng và lưu tệp tại
   `outputs/submissions/submission.csv`.

## Nguyên tắc làm việc

- Không chỉnh sửa dữ liệu gốc; mọi sản phẩm trung gian phải được tái tạo bằng mã.
- Không commit dữ liệu lớn, checkpoint, hình vẽ hoặc submission sinh tự động.
- Không lưu khóa API, token, thông tin đăng nhập hoặc cấu hình bí mật trong Git.
- Mọi quyết định về thuật toán, validation và metric phải được đưa ra sau bước
  khảo sát dữ liệu và được ghi nhận trong tài liệu/thực nghiệm.

