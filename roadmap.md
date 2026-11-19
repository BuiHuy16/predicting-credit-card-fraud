# ROADMAP — PREDICTING CREDIT CARD FRAUD

> **Bài tập lớn môn Học máy | Kaggle Playground Series S6E11 | Kế hoạch rút gọn 2 tuần**  
> **Phiên bản:** 2.0 — 19/11/2026  
> **Hạn hoàn thành và nộp:** **30/11/2026**  
> **Thời gian thực hiện còn lại:** **19/11–30/11/2026 (12 ngày lịch)**  
> **Nhóm:** 3 thành viên  
> **Phạm vi:** Machine Learning (ML) + Deep Learning (DL), **không sử dụng LLM trong giải pháp**

**Quy ước:** Trong tài liệu này, ngày **30/11** là hạn chốt toàn bộ sản phẩm của nhóm. Không lập kế hoạch triển khai, báo cáo hay nghiệm thu sau ngày này. Đây là lịch nén từ mục tiêu “2 tuần”, do thời điểm bắt đầu thực tế là 19/11. Nên hoàn thành submission Kaggle sớm hơn hạn ít nhất 1 ngày.

---

## 1. Mục tiêu và sản phẩm cuối cùng

### 1.1. Mục tiêu

Xây dựng và so sánh mô hình ML/DL để **dự đoán xác suất một giao dịch thẻ tín dụng có gian lận** (`is_fraud`) trên dữ liệu Kaggle Playground Series S6E11. Hệ thống dự đoán cho các dòng trong `test.csv` và xuất file CSV đúng định dạng để Kaggle chấm **ROC-AUC**.

### 1.2. Sản phẩm cần có trước 30/11

| Sản phẩm | Vị trí dự kiến | Tiêu chí đạt |
|---|---|---|
| Khảo sát dữ liệu và EDA | `docs/01_DATA_AUDIT.md`, `notebooks/01_eda.ipynb` | Có số liệu thật, biểu đồ và nhận xét |
| Giao thức validation, tiền xử lý | `docs/02_VALIDATION_PROTOCOL.md`, `src/data/` | Tái lập được; không data leakage |
| Mô hình ML | `src/models/`, `src/training/`, `experiments/ml/` | Baseline + ít nhất 1 ML mạnh hơn được đánh giá |
| Mô hình DL | `src/models/`, `src/training/`, `experiments/dl/` | Có MLP/PyTorch và kết quả validation thực |
| So sánh ML/DL | `docs/03_MODEL_COMPARISON.md`, `experiments/comparison/` | Dùng chung tập validation và ROC-AUC |
| File dự đoán | `outputs/submissions/submission.csv` | Đúng cột, đúng ID, xác suất trong [0,1] |
| Minh chứng nộp Kaggle | `docs/04_KAGGLE_SUBMISSION.md` | Ghi lại thời gian, file/model và điểm nếu được chấm |
| Báo cáo, slide, hướng dẫn chạy | `docs/` và `README.md` | Có thể trình bày, giải thích và chạy lại |
| Kiểm thử tối thiểu | `tests/` | Kiểm tra schema, split, prediction và submission |

**Lưu ý:** Đây là mục tiêu nội bộ đề xuất; nếu giảng viên có rubric cụ thể thì ưu tiên rubric.

---

## 2. Phân công công việc

| Người phụ trách | Vai trò công khai | Trách nhiệm chính | Bàn giao tối thiểu |
|---|---|---|---|
| **Người 1** | **Data & EDA** | Data audit, EDA, kiểm tra chất lượng dữ liệu, xác định split/validation và chuẩn bị dữ liệu dùng chung | Báo cáo dữ liệu; notebook EDA; quy ước split và preprocessing |
| **Người 2** | **ML** | Logistic Regression baseline; Random Forest/XGBoost hoặc mô hình cây phù hợp; tuning vừa đủ; phân tích kết quả ML | Script ML chạy được; xác suất validation/test; bảng metric và config |
| **Người 3** | **DL & Integration** | MLP bằng PyTorch; huấn luyện/tuning DL; kết nối pipeline dự đoán, so sánh kết quả và xuất submission | Script DL; learning curves; bảng so sánh; `submission.csv` |
| **Cả nhóm** | **Phối hợp** | Chốt giao thức đánh giá, kiểm tra chéo, viết phần báo cáo tương ứng, rà soát demo và nộp | Bản chạy cuối + báo cáo/slide nhất quán |

Không sử dụng chức danh quản lý trong tài liệu; bảng trên chỉ mô tả trách nhiệm chuyên môn và công việc phối hợp.

### Ranh giới trách nhiệm kỹ thuật

- **Người 1** đảm bảo **cùng tập dòng train/validation**; không phải ép mọi mô hình dùng y hệt các phép biến đổi.
- **Người 2 và Người 3** tự chịu trách nhiệm về việc fit tiền xử lý phù hợp cho mô hình của mình **chỉ trên phần train**; sử dụng thông tin validation/test đúng cách.
- **Người 3** tích hợp đầu ra dự đoán; **Người 1** đối chiếu schema; **Người 2** xác nhận lại việc nạp model ML và suy luận.
- Mỗi người tự ghi kết quả và viết phần báo cáo của mình. **Cả nhóm** chịu trách nhiệm kiểm tra tính nhất quán của bảng so sánh.

---

## 3. Hiện trạng repository và cách tổ chức

### 3.1. Cấu trúc repository hiện tại

```text
predicting-credit-card-fraud/
├── configs/                       # Cấu hình chạy; hiện có tệp giữ chỗ
│   └── .gitkeep
├── datasets/
│   ├── raw/                       # Ba CSV gốc, chỉ đọc
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   └── processed/                 # Vị trí dữ liệu xử lý sinh ra nếu cần
│       └── .gitkeep
├── docs/                          # Tài liệu dự án; hiện có tệp giữ chỗ
│   └── .gitkeep
├── experiments/
│   ├── ml/                        # Kết quả và metadata thực nghiệm ML
│   │   └── .gitkeep
│   ├── dl/                        # Kết quả và metadata thực nghiệm DL
│   │   └── .gitkeep
│   ├── comparison/                # Kết quả so sánh ML/DL
│   │   └── .gitkeep
│   └── .gitkeep
├── notebooks/                     # Notebook khảo sát và thử nghiệm
│   └── .gitkeep
├── outputs/
│   ├── models/                    # Model và checkpoint sinh tự động
│   │   └── .gitkeep
│   ├── figures/                   # Biểu đồ sinh tự động
│   │   └── .gitkeep
│   ├── logs/                      # Log chạy sinh tự động
│   │   └── .gitkeep
│   └── submissions/               # File dự đoán gửi Kaggle
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── training/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   └── prediction/
│       └── __init__.py
├── tests/                         # Kiểm thử tự động; hiện có tệp giữ chỗ
│   └── .gitkeep
├── .gitignore
├── README.md
├── requirements.txt
└── roadmap.md                     # Tài liệu này, đặt tại thư mục gốc
```

Các tệp `.gitkeep` chỉ giữ cấu trúc thư mục rỗng trong Git. Các package trong `src/` hiện mới có `__init__.py`; chưa có triển khai EDA, preprocessing, mô hình, huấn luyện, đánh giá, dự đoán hay kết quả thực nghiệm. Không di chuyển hoặc chỉnh sửa ba CSV gốc trong `datasets/raw/`.

### 3.2. Những file và thành phần sẽ triển khai sau

- **Cấu hình:** bổ sung `configs/data.yaml`, `configs/ml.yaml` và `configs/dl.yaml` khi pipeline thực tế cần cấu hình tách riêng; không lưu bí mật trong các file này.
- **Khảo sát dữ liệu:** xây dựng `notebooks/01_eda.ipynb` và ghi kết quả vào `docs/01_DATA_AUDIT.md` dựa trên số liệu thực.
- **Validation và preprocessing:** triển khai các module cần thiết trong `src/data/` và mô tả giao thức tại `docs/02_VALIDATION_PROTOCOL.md`.
- **Mô hình và huấn luyện:** bổ sung module ML/DL trong `src/models/` cùng quy trình chạy trong `src/training/`; không tạo file khung rỗng hàng loạt.
- **Đánh giá và dự đoán:** triển khai metric/so sánh trong `src/evaluation/`, inference và kiểm tra submission trong `src/prediction/`, kèm kiểm thử tương ứng trong `tests/`.
- **Theo dõi thực nghiệm:** ghi cấu hình, metric và kết quả thực trong `experiments/ml/`, `experiments/dl/` và `experiments/comparison/`; tổng hợp so sánh tại `docs/03_MODEL_COMPARISON.md`.
- **Artifact sinh tự động:** lưu checkpoint tại `outputs/models/`, biểu đồ tại `outputs/figures/`, log tại `outputs/logs/` và file dự đoán tại `outputs/submissions/`. Các artifact này tuân theo `.gitignore`; minh chứng nộp Kaggle được ghi tại `docs/04_KAGGLE_SUBMISSION.md`.
- **CLI tùy chọn:** chỉ tạo `scripts/` (ví dụ `run_ml.py`, `run_dl.py`, `make_submission.py`) khi cần lệnh chạy lại rõ ràng và `src/` chưa cung cấp entry point phù hợp.

Dữ liệu gốc, checkpoint lớn, cache, log, output sinh tự động và bí mật không được commit lên Git. Hướng dẫn đường dẫn dữ liệu và cách chạy sẽ tiếp tục được cập nhật trong `README.md` khi các thành phần tương ứng được triển khai.

---

## 4. Nguyên tắc kỹ thuật bắt buộc

1. **Bài toán nhị phân, đầu ra là xác suất:** mỗi `id` trong `test.csv` cần `is_fraud` nằm trong `[0, 1]`, không chỉ nhãn cứng 0/1.
2. **Chống leakage:** split trước khi fit imputation, scaling, encoding có thống kê học từ dữ liệu, feature selection hoặc oversampling. Nếu dùng resampling thì chỉ trên train fold.
3. **Cùng validation:** cố định seed, danh sách chỉ số dòng/tập split và quy tắc đánh giá; ML/DL cùng tập hàng validation.
4. **Kiểm tra đặc trưng nhạy cảm về thời gian/nhóm:** nếu audit phát hiện giao dịch có quan hệ theo thời gian hoặc nhóm, cân nhắc split phù hợp thay vì mặc định random stratified.
5. **Không xem `test.csv` là tập validation:** không có nhãn công khai và không được dùng điểm Public Leaderboard thay thế đánh giá nội bộ.
6. **Metric chính ROC-AUC**; bổ sung PR-AUC, precision, recall, F1 (các metric theo ngưỡng cần ghi rõ ngưỡng).
7. **Tái lập:** lưu seed, phiên bản thư viện, cấu hình, số fold, thời gian, metric và đường dẫn artifact.
8. **Không cần web/backend, cloud production, LLM:** tập trung chất lượng mô hình, phân tích và báo cáo.

---

## 5. Lộ trình theo ngày: 19–30/11/2026

### Giai đoạn A — Chốt dữ liệu và baseline (19–22/11)

| Ngày | Người 1 — Data & EDA | Người 2 — ML | Người 3 — DL & Integration | Mốc cần đạt |
|---|---|---|---|---|
| **19/11** | Kiểm tra `train`, `test`, `sample_submission`; thống kê shape, dtype, missing, duplicates, target | Chuẩn bị môi trường và bộ khung baseline; kiểm tra dependency | Chuẩn bị PyTorch/training skeleton; xác định giao diện dữ liệu/prediction | Dataset đọc được, thống nhất file/metric/đầu ra |
| **20/11** | EDA phân bố nhãn, numeric/categorical, outliers; kiểm tra nguy cơ leakage | Chạy baseline Logistic Regression trên split sơ bộ nếu hợp lệ | Smoke test MLP trên dữ liệu nhỏ để kiểm tra code, chưa kết luận chất lượng | Có bản audit và baseline đầu tiên |
| **21/11** | Chốt validation protocol (split, seed, feature, transform); xuất chỉ số split/cấu hình | Chạy lại baseline theo split chính thức; thử mô hình cây đầu tiên | Huấn luyện MLP cơ bản trên split chính thức, lưu train/val loss | **Gate A:** cùng validation, không leakage |
| **22/11** | Hoàn thiện notebook EDA và dữ liệu đầu vào ổn định | Huấn luyện Random Forest hoặc XGBoost; lưu metric | Thử dropout/early stopping/pos_weight nếu phù hợp; ghi training curves | Có ≥2 ML thử nghiệm và ≥1 DL chạy end-to-end |

**Gate A — Phải đạt trước 23/11:** Các file gốc không bị đổi; mô hình đọc được dữ liệu; có split chung, seed, baseline và ít nhất một kết quả DL hợp lệ. Nếu chưa đạt, dừng nâng cao, ưu tiên sửa pipeline.

### Giai đoạn B — Cải thiện và nộp thử (23–26/11)

| Ngày | Người 1 — Data & EDA | Người 2 — ML | Người 3 — DL & Integration | Mốc cần đạt |
|---|---|---|---|---|
| **23/11** | Phân tích sai khác train/test, missing, cardinality; đề xuất feature cải tiến hợp lý | Tuning nhỏ cho XGBoost/Random Forest hoặc thuật toán tốt nhất | Tuning MLP: width/depth, learning rate, batch size; theo dõi overfit | Có log thực nghiệm và baseline so sánh |
| **24/11** | Kiểm tra leakage, sự nhất quán giữa các mô hình, nguồn feature | So sánh ít nhất 2 mô hình ML; xuất OOF/validation probability | So sánh biến thể MLP; chọn checkpoint dựa trên validation | Bảng ROC-AUC/PR-AUC nội bộ phiên bản đầu |
| **25/11** | Đối chiếu thứ tự ID, schema train/test/sample | Đóng gói inference ML; dự đoán test | Ghép pipeline dự đoán, tạo và kiểm thử submission thử | **Có `submission.csv` hợp lệ lần đầu** |
| **26/11** | Kiểm tra kết quả validation, hỗ trợ xác nhận không leakage | Khắc phục lỗi inference nếu có; phân tích ML | **Nộp thử Kaggle**, ghi điểm/mốc submission; đối chiếu với CV | **Gate B:** ít nhất 1 submission được Kaggle chấp nhận |

**Gate B — Phải đạt trước 27/11:** Có script tái lập; ML và DL đều có chỉ số validation; submission đầu tiên được chấp nhận hoặc đã ghi rõ trở ngại truy cập Kaggle và phương án xử lý.

### Giai đoạn C — Chốt kết quả và nộp (27–30/11)

| Ngày | Người 1 — Data & EDA | Người 2 — ML | Người 3 — DL & Integration | Mốc cần đạt |
|---|---|---|---|---|
| **27/11** | Viết Data/EDA/Preprocessing/Validation vào tài liệu | Viết phương pháp ML + bảng tuning + error analysis | Viết phương pháp DL, training curves và kết quả; tổng hợp so sánh | Dự thảo báo cáo và bảng kết quả đầy đủ |
| **28/11** | Kiểm tra/hoàn thiện biểu đồ, nguồn số liệu, tính tái lập | Chốt cấu hình ML; hỗ trợ so sánh và demo | Chọn mô hình tốt nhất bằng validation; nếu có lợi ích rõ, thử blending đơn giản | **Gate C:** freeze pipeline, chọn ứng viên cuối |
| **29/11** | Rà soát dữ liệu, README, các file cho nộp | Test chạy lại ML từ đầu và inference | **Nộp submission cuối lên Kaggle**; lưu bằng chứng, khóa model | Submission cuối được gửi trước hạn nội bộ |
| **30/11** | Rà soát báo cáo, checklist và minh chứng | Rà soát code + kết quả ML + slide | Rà soát DL, tích hợp, `submission.csv`, demo | **Bàn giao toàn bộ project trước 30/11** |

**Gate cuối — 30/11:** Repo đọc được; quy trình chạy lại có hướng dẫn; có ML/DL thực nghiệm thật; có báo cáo, slide theo yêu cầu; có CSV hợp lệ và minh chứng nộp; không còn blocker P0.

### Trường hợp tiến độ trễ

- Nếu đến **21/11** chưa chốt được split → không triển khai tuning, ưu tiên validation/preprocessing.
- Nếu đến **24/11** mô hình DL chưa chạy ổn → dùng một MLP nhỏ với early stopping, không thêm kiến trúc mới.
- Nếu đến **25/11** chưa tạo được submission → dùng mô hình ML baseline tốt nhất để xuất CSV và nộp trước, sau đó cải thiện.
- Nếu đến **27/11** báo cáo chưa bắt đầu → dừng feature engineering nâng cao, chuyển sang tổng hợp và rà soát.
- Nếu đến **29/11** không còn thời gian → nộp mô hình ổn định đã được kiểm chứng, không thay đổi pipeline phút cuối.

---

## 6. Mô hình dự kiến và thứ tự ưu tiên

| Ưu tiên | Mô hình | Người thực hiện | Ghi chú |
|---|---|---|---|
| **P0** | Logistic Regression | Người 2 | Baseline nhanh, dễ đối chiếu |
| **P0** | XGBoost **hoặc** Random Forest | Người 2 | Ít nhất một mô hình phi tuyến có đánh giá thực |
| **P0** | MLP (PyTorch) | Người 3 | Deep Learning bắt buộc trong phạm vi nhóm |
| **P1** | Mô hình ML thứ ba (RF/XGBoost/LightGBM nếu phù hợp) | Người 2 | Chỉ nếu đủ thời gian |
| **P1** | Tuning MLP, early stopping, class weighting | Người 3 | Dựa trên bằng chứng validation |
| **P2** | Blending/ensemble hai mô hình | Người 3, Người 2 | Chỉ khi validation cải thiện có ý nghĩa |

Không cam kết một thuật toán luôn tốt hơn thuật toán khác. Không chọn chỉ dựa vào độ phức tạp kiến trúc hoặc một kết quả Public Leaderboard.

**Gợi ý MLP ban đầu:** đầu vào sau preprocessing → Dense(128)+ReLU → Dense(64)+ReLU → Dense(1) xuất **logit**; dùng `BCEWithLogitsLoss` khi train, sigmoid chỉ khi xuất xác suất. Cấu hình chỉ là điểm xuất phát, không phải kiến trúc tối ưu đã kiểm chứng.

---

## 7. Thiết kế thí nghiệm và bảng kết quả

### 7.1. Quy ước thí nghiệm

- Seed cho split phải thống nhất, ví dụ `42`; seed huấn luyện ghi riêng theo từng run.
- Nếu nhãn mất cân bằng, kiểm tra stratification và lựa chọn metric phụ hợp lý.
- Mỗi run ghi: `run_id`, `model`, `split_id`, `seed`, `features`, `preprocessing`, `params`, `roc_auc`, `pr_auc`, `train_seconds`, `artifact_path`.
- Điểm từ Kaggle ghi ở cột riêng `public_lb_auc`; **không trộn** với validation ROC-AUC.
- Có thể dùng 1 validation split để chạy nhanh giai đoạn đầu; nếu đủ thời gian kiểm tra tính ổn định bằng nhiều seed/fold cho mô hình tốt nhất.

### 7.2. Mẫu bảng so sánh (điền số đo thực)

| Model | Split | Val ROC-AUC | Val PR-AUC | Train time | Public LB | Ghi chú |
|---|---|---:|---:|---:|---:|---|
| Logistic Regression | TBD | — | — | — | — | Baseline |
| XGBoost/Random Forest | TBD | — | — | — | — | ML |
| MLP | TBD | — | — | — | — | DL |

**Không điền chỉ số giả** vào báo cáo hoặc trình bày như kết quả thực nghiệm.

---

## 8. Kiểm tra `submission.csv`

File phải bám theo `sample_submission.csv`, gồm `id` và xác suất `is_fraud` tương ứng với từng dòng trong `test.csv`.

**Checklist tối thiểu:**

- [ ] Đúng tên cột và thứ tự cột theo `sample_submission.csv`.
- [ ] Số dòng bằng số dòng `test.csv`.
- [ ] `id` khớp đúng thứ tự/giá trị của `test.csv`.
- [ ] `is_fraud` là số hữu hạn, không NaN/inf, thuộc `[0,1]`.
- [ ] Dùng preprocessing/model đã fit trên train; không dùng nhãn test ẩn.
- [ ] Lưu tên mô hình, commit (nếu có), seed và file config tạo submission.
- [ ] Nộp thử trước 29/11, lưu link hoặc hình minh chứng từ Kaggle.

**Mốc an toàn:** submission đầu tiên **26/11**, submission chốt **29/11**, ngày **30/11** chỉ dành cho xử lý sự cố và nộp toàn bộ. Đối chiếu thời điểm đóng Kaggle trên trang Timeline để tránh nhầm múi giờ.

---

## 9. Tiêu chí nghiệm thu nhanh

### Data & EDA — Người 1

- [ ] Dữ liệu gốc được bảo toàn; có kiểm tra schema, missing, duplicate, target balance.
- [ ] Có EDA và nhận xét dựa trên số liệu thật, không chỉ hình vẽ.
- [ ] Có giao thức split/validation cố định và kiểm soát leakage.
- [ ] Tiền xử lý có thể chạy lại bằng code.

### ML — Người 2

- [ ] Có Logistic Regression baseline.
- [ ] Có tối thiểu một mô hình ML phi tuyến chạy thật.
- [ ] Có xác suất validation/test, ROC-AUC và log cấu hình.
- [ ] Có giải thích kết quả và giới hạn mô hình.

### DL & Integration — Người 3

- [ ] Có MLP/PyTorch, training loop, validation và checkpoint phù hợp.
- [ ] Có learning curves, ROC-AUC và nhận xét overfitting/underfitting.
- [ ] Có bảng so sánh ML/DL trên chung validation.
- [ ] Có pipeline inference và `submission.csv` hợp lệ.
- [ ] Có minh chứng Kaggle và hướng dẫn tái lập.

### Cả nhóm

- [ ] Chạy lại được baseline và mô hình cuối theo README.
- [ ] Không còn lỗi blocker trong mã nguồn, dữ liệu hoặc định dạng submission.
- [ ] Mọi kết quả báo cáo có log/file nguồn đối chiếu được.
- [ ] Hoàn tất **trước hoặc trong ngày 30/11/2026**.

---

## 10. Rủi ro và phương án xử lý

| Rủi ro | Phòng ngừa / ứng phó |
|---|---|
| Dữ liệu lớn, máy yếu | Dùng sample cho smoke test; đọc tối ưu kiểu dữ liệu; Kaggle Notebook/Colab nếu cần |
| Data leakage | Fit mọi transform trên train fold; code review phần split và resampling |
| DL không hơn ML | Vẫn báo cáo kết quả trung thực; chọn mô hình tốt theo validation, không ép DL thắng |
| Bị chậm do phần dữ liệu | Người 1 bàn giao split ổn định sớm; Người 2/3 làm skeleton song song |
| Tuning tốn quá nhiều thời gian | Giới hạn số thử nghiệm; ưu tiên 1–2 mô hình tốt nhất |
| Kaggle không nhận submission | Kiểm tra đúng file mẫu, số dòng, ID, xác suất; nộp thử từ 26/11 |
| Public LB và validation lệch | Kiểm tra generalization/split; không overfit leaderboard |
| Báo cáo bị dồn cuối kỳ | Mỗi người cập nhật kết quả và phần viết sau mỗi mốc, hoàn thiện từ 27/11 |

---

## 11. Quy tắc làm việc với Git và tài liệu

- Tách nhánh theo công việc (`data-eda`, `ml`, `dl-integration`) nếu phù hợp; pull/review trước khi merge.
- `datasets/raw/` chỉ đọc; `datasets/processed/` lưu output sinh ra nếu thật sự cần và không commit dữ liệu nặng.
- `outputs/models/`, `outputs/logs/`, `outputs/submissions/` cần quy ước Git rõ ràng; giữ file submission cuối như artifact nộp riêng nếu `.gitignore` đang chặn.
- README phải mô tả setup, đường dẫn dữ liệu, lệnh EDA, lệnh training/evaluation và tạo CSV.
- Tài liệu `roadmap.md` nằm **ngay ở thư mục gốc**, không chuyển vào `docs/`.
- Không tự động xóa, ghi đè CSV gốc hoặc chạy `git commit`/`push` khi chưa thống nhất.

---

## 12. Nguồn tham khảo và giới hạn

- Cuộc thi: https://www.kaggle.com/competitions/playground-series-s6e11
- Dữ liệu: https://www.kaggle.com/competitions/playground-series-s6e11/data
- Đánh giá: https://www.kaggle.com/competitions/playground-series-s6e11/overview/evaluation
- Leaderboard: https://www.kaggle.com/competitions/playground-series-s6e11/leaderboard

**Chưa được xác minh từ repo ở thời điểm viết:** số dòng/cột, tỷ lệ gian lận, missing/duplicates, chất lượng dữ liệu, điểm mô hình và yêu cầu báo cáo chính thức từ giảng viên. Các con số thực nghiệm sẽ được bổ sung sau khi chạy thật; không được suy đoán.

**Chốt kế hoạch:** 19–22/11 chốt dữ liệu + baseline; 23–26/11 cải thiện + nộp thử; 27–30/11 viết báo cáo, chốt submission và bàn giao toàn bộ. **Deadline 30/11/2026.**
