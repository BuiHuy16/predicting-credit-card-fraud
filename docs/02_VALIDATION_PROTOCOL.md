# Validation Protocol

**Trạng thái:** giao thức mặc định đã được chốt cho các thực nghiệm ML/DL trong project.  
**Phiên bản:** 1  
**Ngày chốt:** 22/11/2026

## 1. Mục đích

Mọi mô hình ML và DL phải được so sánh trên cùng một tập validation để kết quả có thể đối chiếu công bằng và tái lập. Tập `test.csv` chỉ dùng cho dự đoán cuối cùng theo định dạng submission; không dùng để chọn mô hình hoặc tham số.

## 2. Quyết định chia dữ liệu

- Chiến lược: **stratified holdout**, chia ngẫu nhiên có giữ tỷ lệ nhãn.
- Tỷ lệ: **80% train / 20% validation** trên `datasets/raw/train.csv`.
- Seed: **42**.
- Đơn vị chia: từng giao dịch; `id` chỉ là khóa để xác định hàng, không phải feature.
- Target: `is_fraud`.
- Split ID: `stratified-holdout-v1-seed42-val20-34abc885ee94`.
- Danh sách ID validation cố định: [`experiments/validation/validation_ids_seed42.csv`](../experiments/validation/validation_ids_seed42.csv).

EDA hiện có không xác định được timestamp giao dịch hoặc customer/group ID đáng tin cậy để dùng time split hay group split. `id` tuần tự không được xem là timestamp. Các kiểm tra đặc trưng cho thấy phân phối train/test rất gần nhau; vì vậy stratified random holdout là lựa chọn mặc định phù hợp với thông tin hiện có.

Validation có **139.992** giao dịch: 137.654 non-fraud và 2.338 fraud (fraud rate **1,6701%**). Training có **559.968** giao dịch: 550.617 non-fraud và 9.351 fraud (fraud rate **1,6699%**). Tổng hai phần đúng bằng 699.960 dòng train gốc; hai phần rời nhau.

### Tạo lại và nạp split

Artifact CSV liệt kê ID validation theo thứ tự tăng dần. ID trong train còn lại là phần training. Dùng đúng danh sách này; không tự gọi một hàm chia ngẫu nhiên mới cho từng mô hình.

```python
import pandas as pd

train = pd.read_csv("datasets/raw/train.csv")
validation_ids = pd.read_csv(
    "experiments/validation/validation_ids_seed42.csv"
)["id"]

is_validation = train["id"].isin(validation_ids)
train_part = train.loc[~is_validation].copy()
validation_part = train.loc[is_validation].copy()

X_train = train_part.drop(columns=["id", "is_fraud"])
y_train = train_part["is_fraud"]
X_validation = validation_part.drop(columns=["id", "is_fraud"])
y_validation = validation_part["is_fraud"]
```

Trước khi chạy mô hình, xác nhận số dòng và số fraud khớp với các con số ở trên. Giữ ID bên ngoài ma trận đặc trưng để ghép dự đoán và kiểm tra thứ tự hàng.

### Quy tắc sinh artifact

Artifact được tạo bằng phép xếp hạng băm SHA-256 có seed và nhãn riêng cho từng hàng, sau đó chọn đúng 20% ID trong từng lớp (làm tròn đến số nguyên gần nhất). Danh sách cuối được sắp xếp theo ID. Quy tắc này không phụ thuộc thứ tự lấy mẫu của thư viện ML; CSV đã lưu là nguồn sự thật cho mọi lần chạy.

SHA-256 của chuỗi ID validation đã sắp xếp, mỗi ID một dòng và có newline cuối: `34abc885ee94d882d7d8fe2b14eedbb0eb2ff2e7b4c925247369f2b5dc0ad652`.

## 3. Feature contract

- Target là `is_fraud`; giá trị hợp lệ `{0, 1}`.
- Loại `id` khỏi feature đầu vào mặc định. Dùng ID chỉ để nhận diện/ghép hàng.
- Dùng 24 feature hiện có: 13 numerical, 5 categorical string và 6 boolean.
- Không loại hoặc sửa feature/outlier chỉ dựa trên EDA đơn biến.
- `merchant_risk_score` và `is_ai_generated_scam_attempt` cần được xác minh về thời điểm tạo/khả dụng trước khi triển khai thực tế. Hiện chưa có đủ bằng chứng để tuyên bố chúng là leakage hoặc loại chúng khỏi baseline.

## 4. Chống leakage trong preprocessing

1. Tách train/validation bằng artifact ID trước mọi bước học tham số.
2. Fit imputer, scaler, encoder, feature selection và các phép biến đổi có thống kê chỉ trên `X_train`.
3. Chỉ transform `X_validation` và `X_test` bằng các transformer đã fit trên training partition.
4. Nếu dùng oversampling/undersampling/class weighting có thống kê học được, chỉ áp dụng trong training partition. Không resample validation/test.
5. Nếu dùng target encoding, phải thực hiện out-of-fold trong training partition; không tính encoding trên toàn bộ dữ liệu trước split.
6. Các nhóm category chưa thấy phải được pipeline xử lý an toàn khi transform.

ML và DL có thể dùng cách preprocessing khác nhau, nhưng phải giữ cùng hàng `train_part` và `validation_part`, cùng target, cùng thứ tự ID khi báo cáo prediction.

Code dùng chung nằm trong `src/data/loading.py`, `src/data/splitting.py` và `src/data/preprocessing.py`. `make_preprocessor(scale_numeric=True)` tạo nhánh impute + scale cho biến số, impute + one-hot cho biến chuỗi và impute cho biến boolean; dùng được làm đầu vào cho Logistic Regression hoặc MLP. Với mô hình cây, dùng `scale_numeric=False` để bỏ scale số học. Nếu MLP cần tensor dày `float32`, chuyển đầu ra sau `fit_transform`/`transform` sang dense `float32` trong pipeline của mô hình; không fit lại transformer trên validation. Categorical one-hot xử lý category chưa thấy bằng `handle_unknown="ignore"`.

## 5. Metric và quy tắc chọn mô hình

- **Metric chính:** ROC-AUC tính trên xác suất/score cho lớp fraud.
- **Metric bổ sung:** average precision (AP), báo cáo trong project dưới tên PR-AUC để phản ánh lớp fraud hiếm. Dùng cùng cách tính `average_precision_score` giữa mọi run.
- Precision, recall và F1 ở threshold **0,50** chỉ là số liệu tham khảo, phải ghi rõ threshold. Nếu chọn một threshold vận hành khác, ghi cách chọn và chỉ dùng validation; không dùng test hoặc leaderboard.
- Không dùng accuracy làm metric chính.
- Chọn mô hình theo ROC-AUC validation; dùng PR-AUC và nhu cầu vận hành làm thông tin bổ trợ. Không chọn mô hình theo kết quả test hoặc Public Leaderboard.
- Mọi run phải ghi `split_id`, model, seed huấn luyện riêng, feature list, preprocessing, tham số, ROC-AUC, AP/PR-AUC, threshold nếu có và đường dẫn artifact.

## 6. Hạn chế và xem xét lại

Đây là một holdout đơn, phù hợp để các thành viên có baseline chung và chạy nhanh; kết quả có thể phụ thuộc cách chia. Sau khi có mô hình ứng viên, có thể kiểm tra độ ổn định bằng cross-validation stratified trong phần training, nhưng validation holdout này vẫn là mốc so sánh chung cho báo cáo. Không thay split giữa chừng; nếu có bằng chứng mới về nhóm/thời gian hoặc feature leakage, tạo phiên bản protocol mới, split ID mới và chạy lại các mô hình cần so sánh.
