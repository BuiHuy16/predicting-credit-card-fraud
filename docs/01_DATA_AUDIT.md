# Initial Data Audit — Kaggle Playground Series S6E11

**Ngày kiểm tra:** 19/11/2026
**Phạm vi:** kiểm tra toàn bộ `train.csv`, `test.csv` và `sample_submission.csv`; không lấy mẫu, không làm sạch dữ liệu và không thay đổi file gốc.

## 1. Tổng quan dataset

Ba file CSV trong `datasets/raw/` đều đọc thành công bằng Python 3.12.7, pandas 3.0.3 và NumPy 2.4.5. Quá trình kiểm tra dùng `pandas.read_csv(..., low_memory=False)` và nạp toàn bộ từng file vào bộ nhớ. Không cần đọc theo chunk.

Dataset dùng cho bài toán phân loại nhị phân với target `is_fraud`. Sau khi tách riêng `id` và target, train/test có:

- 13 feature numerical (`int64` hoặc `float64`, không tính boolean).
- 5 feature categorical dạng chuỗi.
- 6 feature binary dạng `bool`, được xếp vào nhóm categorical/binary khi thiết kế preprocessing.
- `id` là định danh và không được mặc định xem là feature huấn luyện.

Checksum SHA-256 tại thời điểm audit:

| File | SHA-256 |
|---|---|
| `train.csv` | `87EAA10B8D17A7E4F657B8DA1C66242431338BDD8B6AD45861521AC0C0E9FBD5` |
| `test.csv` | `10F516959020357B565ABAAA4FD25BAAE97FC2F536D0805695BB7CC15C1660F0` |
| `sample_submission.csv` | `13FC6ED8B73DE5329884EC490F2A0B750CF29A7CDA772B78CFA0CF486F6F35FA` |

## 2. Shape, kích thước file và khả năng đọc

| File | Số dòng | Số cột | Kích thước | Thời gian đọc | Bộ nhớ DataFrame | Kết quả đọc |
|---|---:|---:|---:|---:|---:|---|
| `train.csv` | 699.960 | 26 | 98.533.176 byte (93,969 MiB) | 1,183 giây | 273,526 MiB | Thành công, toàn bộ dữ liệu |
| `test.csv` | 299.984 | 25 | 41.068.069 byte (39,166 MiB) | 0,467 giây | 114,928 MiB | Thành công, toàn bộ dữ liệu |
| `sample_submission.csv` | 299.984 | 2 | 8.399.564 byte (8,010 MiB) | 0,057 giây | 4,578 MiB | Thành công, toàn bộ dữ liệu |

Thời gian đọc và mức sử dụng bộ nhớ là số đo tại môi trường audit hiện tại, không phải yêu cầu tài nguyên cố định cho mọi máy.

Phân bố dtype vật lý:

| File | `int64` | `float64` | `bool` | `str` | Numerical feature | Categorical/binary feature | ID | Target |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `train.csv` | 8 | 7 | 6 | 5 | 13 | 11 | 1 | 1 |
| `test.csv` | 8 | 6 | 6 | 5 | 13 | 11 | 1 | 0 |
| `sample_submission.csv` | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 1 cột xác suất mẫu |

## 3. Schema, dtype và số giá trị duy nhất

### 3.1. Train và test

| Cột | Vai trò sơ bộ | Train dtype | Test dtype | Train `nunique` | Test `nunique` |
|---|---|---|---|---:|---:|
| `id` | Định danh | `int64` | `int64` | 699.960 | 299.984 |
| `amount_usd` | Numerical | `float64` | `float64` | 13.387 | 10.510 |
| `hours_since_last_txn` | Numerical | `float64` | `float64` | 3.633 | 3.331 |
| `txn_count_last_24h` | Numerical | `int64` | `int64` | 12 | 11 |
| `distance_from_home_km` | Numerical | `float64` | `float64` | 4.365 | 4.108 |
| `card_age_months` | Numerical | `int64` | `int64` | 49 | 49 |
| `customer_age` | Numerical | `int64` | `int64` | 64 | 64 |
| `account_balance_usd` | Numerical | `float64` | `float64` | 43.317 | 31.803 |
| `cvv_retry_count` | Numerical | `int64` | `int64` | 4 | 4 |
| `velocity_score` | Numerical | `float64` | `float64` | 666 | 644 |
| `time_of_day_hour` | Numerical rời rạc | `int64` | `int64` | 24 | 24 |
| `day_of_week` | Numerical rời rạc/cyclic | `int64` | `int64` | 7 | 7 |
| `merchant_risk_score` | Numerical | `float64` | `float64` | 950 | 917 |
| `prior_disputes` | Numerical | `int64` | `int64` | 5 | 5 |
| `merchant_category` | Categorical | `str` | `str` | 12 | 12 |
| `card_type` | Categorical | `str` | `str` | 5 | 5 |
| `auth_method` | Categorical | `str` | `str` | 5 | 5 |
| `channel` | Categorical | `str` | `str` | 5 | 5 |
| `device_type` | Categorical | `str` | `str` | 8 | 8 |
| `is_foreign_transaction` | Binary | `bool` | `bool` | 2 | 2 |
| `is_new_merchant` | Binary | `bool` | `bool` | 2 | 2 |
| `used_vpn` | Binary | `bool` | `bool` | 2 | 2 |
| `ip_country_mismatch` | Binary | `bool` | `bool` | 2 | 2 |
| `billing_shipping_mismatch` | Binary | `bool` | `bool` | 2 | 2 |
| `is_ai_generated_scam_attempt` | Binary | `bool` | `bool` | 2 | 2 |
| `is_fraud` | Target | `float64` | Không có | 2 | — |

Không có tên cột bị lặp trong train hoặc test. Năm cột chuỗi có cùng tập category giữa train và test, không có category chỉ xuất hiện ở một tập và không phát hiện giá trị có khoảng trắng ở đầu/cuối. Sáu cột boolean đều chứa cả `False` và `True` trong cả hai tập.

### 3.2. Sample submission

| Cột | Vai trò | Dtype | `nunique` |
|---|---|---|---:|
| `id` | Định danh | `int64` | 299.984 |
| `is_fraud` | Xác suất mẫu | `float64` | 1 |

`sample_submission.csv` không có tên cột bị lặp. `is_fraud` là cột hằng trong file mẫu với giá trị `0.0166995256871821`.

## 4. Missing values

### 4.1. Train và test

| Cột | Train missing | Train missing (%) | Test missing | Test missing (%) |
|---|---:|---:|---:|---:|
| `id` | 0 | 0,000000% | 0 | 0,000000% |
| `amount_usd` | 0 | 0,000000% | 0 | 0,000000% |
| `hours_since_last_txn` | 0 | 0,000000% | 0 | 0,000000% |
| `txn_count_last_24h` | 0 | 0,000000% | 0 | 0,000000% |
| `distance_from_home_km` | 0 | 0,000000% | 0 | 0,000000% |
| `card_age_months` | 0 | 0,000000% | 0 | 0,000000% |
| `customer_age` | 0 | 0,000000% | 0 | 0,000000% |
| `account_balance_usd` | 0 | 0,000000% | 0 | 0,000000% |
| `cvv_retry_count` | 0 | 0,000000% | 0 | 0,000000% |
| `velocity_score` | 0 | 0,000000% | 0 | 0,000000% |
| `time_of_day_hour` | 0 | 0,000000% | 0 | 0,000000% |
| `day_of_week` | 0 | 0,000000% | 0 | 0,000000% |
| `merchant_risk_score` | 0 | 0,000000% | 0 | 0,000000% |
| `prior_disputes` | 0 | 0,000000% | 0 | 0,000000% |
| `merchant_category` | 0 | 0,000000% | 0 | 0,000000% |
| `card_type` | 0 | 0,000000% | 0 | 0,000000% |
| `auth_method` | 0 | 0,000000% | 0 | 0,000000% |
| `channel` | 0 | 0,000000% | 0 | 0,000000% |
| `device_type` | 0 | 0,000000% | 0 | 0,000000% |
| `is_foreign_transaction` | 0 | 0,000000% | 0 | 0,000000% |
| `is_new_merchant` | 0 | 0,000000% | 0 | 0,000000% |
| `used_vpn` | 0 | 0,000000% | 0 | 0,000000% |
| `ip_country_mismatch` | 0 | 0,000000% | 0 | 0,000000% |
| `billing_shipping_mismatch` | 0 | 0,000000% | 0 | 0,000000% |
| `is_ai_generated_scam_attempt` | 0 | 0,000000% | 0 | 0,000000% |
| `is_fraud` | 0 | 0,000000% | — | — |

### 4.2. Sample submission

| Cột | Missing | Missing (%) |
|---|---:|---:|
| `id` | 0 | 0,000000% |
| `is_fraud` | 0 | 0,000000% |

Không phát hiện missing value trong bất kỳ cột nào của ba file. Không phát hiện `inf` hoặc `-inf` trong các cột numerical.

## 5. Duplicates và kiểm tra ID

| File | Bản ghi trùng hoàn toàn | Dòng thuộc nhóm trùng | ID thiếu | ID trùng | ID duy nhất | Khoảng ID | ID tăng dần/liên tục |
|---|---:|---:|---:|---:|---:|---|---|
| `train.csv` | 0 | 0 | 0 | 0 | 699.960 | 0–699.959 | Có/Có |
| `test.csv` | 0 | 0 | 0 | 0 | 299.984 | 699.960–999.943 | Có/Có |
| `sample_submission.csv` | 0 | 0 | 0 | 0 | 299.984 | 699.960–999.943 | Có/Có |

Không có ID giao nhau giữa train và test. ID trong test và sample submission trùng khớp hoàn toàn cả giá trị lẫn thứ tự. Việc ID liên tục và tăng dần chỉ được ghi nhận như thuộc tính dữ liệu; audit này không kết luận ID mang tín hiệu dự đoán và không xem ID là feature huấn luyện.

## 6. Phân phối target `is_fraud`

Target tồn tại trong `train.csv`, có dtype `float64` và chỉ chứa hai giá trị hợp lệ `0.0` và `1.0`.

| Lớp | Ý nghĩa | Số mẫu | Tỷ lệ |
|---:|---|---:|---:|
| 0 | Không gian lận | 688.271 | 98,330047% |
| 1 | Gian lận | 11.689 | 1,669953% |
| **Tổng** |  | **699.960** | **100,000000%** |

- Target missing: **0**.
- Giá trị ngoài `{0, 1}`: **0**.
- Tỷ lệ majority/minority: **58,8819:1**.
- Tỷ lệ minority/majority: **0,01698314**.

Nhãn mất cân bằng mạnh. Các bước validation và đánh giá sau này cần giữ tỷ lệ lớp phù hợp và ưu tiên ROC-AUC theo roadmap; PR-AUC cũng nên được theo dõi để phản ánh chất lượng trên lớp fraud hiếm. Audit này chưa tạo train/validation split và chưa chọn kỹ thuật xử lý mất cân bằng.

## 7. So sánh train và test

Sau khi loại `is_fraud` khỏi train:

- Train và test có cùng 25 cột, cùng tên và cùng thứ tự.
- Dtype của toàn bộ 25 cột tương ứng khớp chính xác.
- Không có cột thiếu trong test và không có cột dư trong test.
- `test.csv` không chứa target `is_fraud`.
- Năm cột chuỗi có cùng tập category giữa train và test.
- Sáu cột boolean đều dùng cùng hai giá trị `False`/`True`.
- Không có tên cột trùng lặp.

Schema hiện tương thích để dùng chung pipeline preprocessing. Điều này chưa chứng minh phân phối train/test giống nhau; kiểm tra drift, range, outlier và tần suất category thuộc phạm vi EDA tiếp theo.

## 8. Kiểm tra sample submission

| Kiểm tra | Kết quả |
|---|---|
| Cột đúng `id`, `is_fraud` và đúng thứ tự | Đạt |
| Số dòng bằng test (299.984) | Đạt |
| Dtype `id` khớp test (`int64`) | Đạt |
| ID khớp test cả giá trị và thứ tự | Đạt |
| ID thiếu hoặc trùng | Không có |
| `is_fraud` là `float64` | Đạt |
| `is_fraud` thiếu hoặc ngoài `[0, 1]` | Không có |

Giá trị mẫu `is_fraud` là hằng số `0.0166995256871821`, bằng tỷ lệ fraud trong train trong sai số số thực. Đây là placeholder hợp lệ của file mẫu, không phải kết quả mô hình. Khi tạo submission thật, cột này phải được thay bằng xác suất dự đoán theo đúng thứ tự ID.

## 9. Những vấn đề phát hiện được

1. **Mất cân bằng nhãn mạnh:** fraud chỉ chiếm 1,669953%, tương ứng khoảng 1 mẫu fraud trên 58,88 mẫu non-fraud.
2. **Target lưu dưới dạng `float64`:** giá trị thực tế vẫn hợp lệ và chỉ gồm `0.0`/`1.0`; pipeline cần xử lý nhất quán mà không tự ý thay đổi file raw.
3. **`id` là định danh tuần tự:** ID duy nhất, liên tục, tăng dần và tách biệt giữa train/test. Không mặc định dùng ID làm feature nếu chưa có bằng chứng hợp lệ.
4. **Sample submission chứa dự đoán hằng:** đây là placeholder bằng prevalence của train, không được nhầm với submission từ mô hình.

Không phát hiện missing value, duplicate toàn dòng, duplicate/missing ID, giá trị target bất hợp lệ, giá trị vô hạn, cột hằng trong train/test, category không đồng bộ hoặc khoảng trắng thừa trong category chuỗi.

## 10. Kết luận và đề xuất công việc tiếp theo

Dữ liệu **sẵn sàng cho bước EDA** về mặt khả năng đọc, schema, tính đầy đủ, tính duy nhất của ID và sự tương thích train/test. Dữ liệu **chưa nên chuyển trực tiếp sang huấn luyện** trước khi hoàn thành EDA và chốt validation protocol, chủ yếu do target mất cân bằng mạnh và chưa kiểm tra drift/leakage.

Các vấn đề cần nghiên cứu tiếp trong giai đoạn EDA:

- Phân phối, độ lệch và outlier của các feature numerical.
- Tần suất category và tỷ lệ fraud theo category/binary feature.
- Khác biệt phân phối giữa train và test, kể cả frequency drift dù tập category giống nhau.
- Quan hệ của `time_of_day_hour`, `day_of_week` và các biến có thể mang cấu trúc thời gian/nhóm để chọn chiến lược validation phù hợp.
- Tương quan và mức độ trùng lặp thông tin giữa các feature.
- Nguy cơ leakage hoặc proxy target, đặc biệt với các biến có tên gợi ý trực tiếp về rủi ro/gian lận; kết luận chỉ được đưa ra sau phân tích thực tế.
- Cách stratify và metric phụ phù hợp với tỷ lệ fraud thấp; không thực hiện split trong báo cáo audit này.

Audit này không thực hiện biểu đồ, cleaning, encoding, scaling, feature engineering, split, huấn luyện, tuning hoặc tạo submission.
