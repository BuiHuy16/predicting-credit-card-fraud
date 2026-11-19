# Hướng dẫn triển khai Machine Learning

**Người thực hiện:** Người 2 - Machine Learning

**Thời hạn hoàn thành:** 30/11/2026; baseline theo split chính thức chậm nhất 21/11/2026

**Nguồn phạm vi:** `roadmap.md`, `docs/01_DATA_AUDIT.md` và giao thức validation sẽ được tạo tại `docs/02_VALIDATION_PROTOCOL.md`

Tài liệu này hướng dẫn xây dựng baseline Logistic Regression và ít nhất một mô hình phi tuyến là XGBoost hoặc Random Forest. Mọi kết quả chính thức phải dùng cùng hàng validation với DL, lưu xác suất thay vì chỉ nhãn cứng, và có đủ metadata để Người 3 tích hợp hoặc chạy lại. Không thay đổi danh sách mô hình hay mở rộng phạm vi ngoài roadmap nếu chưa thống nhất với nhóm.

## 1. Mục tiêu

- Xây dựng Logistic Regression làm baseline P0 dễ kiểm tra.
- Xây dựng XGBoost hoặc Random Forest làm mô hình ML phi tuyến P0; chọn theo dependency/tài nguyên thực tế.
- Chỉ thử mô hình ML thứ ba P1 khi hai mô hình bắt buộc, inference và bàn giao đã ổn định.
- Đánh giá công bằng bằng ROC-AUC chính và các metric phụ trên split chung.
- Lưu preprocessing, model, config, metric, thời gian và prediction probabilities có thể tái sử dụng.
- Bàn giao prediction validation/test và hướng dẫn nạp model cho Người 3.

## 2. Phạm vi trách nhiệm

Người 2 chịu trách nhiệm:

- Kiểm tra đầu vào từ Người 1 và từ chối chạy chính thức nếu thiếu `split_id`/protocol.
- Triển khai preprocessing riêng cho Logistic Regression và mô hình cây, fit trên training partition.
- Huấn luyện, đánh giá, tuning giới hạn và phân tích overfitting.
- Lưu model/transformer theo cách có thể load lại để inference mà không refit trên validation/test.
- Xuất validation probabilities đúng thứ tự/ID; hỗ trợ xuất test probabilities cho tích hợp.
- Viết kiểm thử tối thiểu cho pipeline ML, metric, serialization và prediction schema.
- Ghi phương pháp/kết quả ML thực vào artifact thí nghiệm và phần báo cáo tương ứng.

Ngoài phạm vi:

- Tự thay split, seed dùng chung, metric chính hoặc feature contract mà không phối hợp.
- Dùng Kaggle test/Public Leaderboard để tuning hoặc chọn hyperparameter.
- Khẳng định mô hình ML tốt hơn DL trước khi so sánh trên cùng protocol.
- Chạy grid search không giới hạn hoặc thêm nhiều thuật toán khi deliverable P0 chưa hoàn thành.

## 3. Trạng thái hiện tại

- Repository hiện chỉ có `src/models/__init__.py`, `src/training/__init__.py` và các thư mục giữ chỗ; chưa có model/pipeline/training script.
- `requirements.txt` có scikit-learn nhưng chưa có `xgboost` hoặc `lightgbm`. Không giả định XGBoost đã cài.
- Initial Data Audit xác nhận 699.960 dòng train, 24 feature đầu vào sau khi loại ID/target, không missing, target fraud 1,669953% và schema train/test tương thích.
- `id` mặc định là định danh, không phải feature.
- EDA chuyên sâu, split chính thức, preprocessing và `docs/02_VALIDATION_PROTOCOL.md` chưa hoàn thành.
- Chưa có ROC-AUC, model artifact, hyperparameter tốt nhất hoặc kết quả Kaggle thực.

Do đó, mọi tên module và artifact trong guide là đường dẫn cần tạo khi triển khai, không phải bằng chứng chúng đã tồn tại.

## 4. Điều kiện đầu vào và dependencies

### 4.1. Điều kiện trước run chính thức

Phải nhận từ Người 1:

- Feature contract: ID, target, numerical, categorical, binary, feature loại bỏ.
- Training/validation indices hoặc quy tắc tái tạo chính xác.
- `split_id`, seed chung, counts và tỷ lệ lớp từng partition.
- `docs/02_VALIDATION_PROTOCOL.md` với metric, threshold báo cáo và leakage guard.
- Giao diện preprocessing hoặc quy tắc bắt buộc cho từng họ mô hình.

Nếu các đầu vào trên chưa có, chỉ được dựng code/smoke test trên mẫu nhỏ. Không đưa metric từ split tạm vào bảng so sánh chính thức.

### 4.2. Dependency thuật toán

- Logistic Regression và Random Forest dùng scikit-learn đã có trong `requirements.txt`.
- XGBoost chỉ được chọn nếu dependency đã được nhóm bổ sung hợp lệ và môi trường chạy đáp ứng; guide này không cài hoặc sửa dependency.
- Nếu không thể dùng XGBoost đúng hạn, triển khai Random Forest theo roadmap thay vì tự chọn thư viện ngoài phạm vi.
- Joblib có thể dùng qua scikit-learn để serialize pipeline; artifact `.joblib` bị `.gitignore` loại khỏi Git, vì vậy phải lưu metadata/đường dẫn riêng.

### 4.3. Dependency giữa thành viên

- Người 1 sở hữu split/contract; Người 2 sở hữu pipeline ML đã fit.
- Người 3 cần validation/test probabilities và hàm load/inference ổn định.
- Mọi thay đổi feature phải được phản ánh ở config/run metadata và thông báo cho Người 1/3.

## 5. Công nghệ, thuật toán và phương pháp

### 5.1. Mô hình theo ưu tiên roadmap

| Ưu tiên | Mô hình | Trạng thái yêu cầu |
|---|---|---|
| P0 | Logistic Regression | Bắt buộc, baseline |
| P0 | XGBoost hoặc Random Forest | Bắt buộc ít nhất một mô hình phi tuyến |
| P1 | Mô hình ML thứ ba | Tùy chọn khi còn thời gian |

### 5.2. Preprocessing

- Logistic Regression: one-hot categorical, biểu diễn binary rõ ràng, scaling numerical. Dùng pipeline duy nhất để tránh fit encoder/scaler ngoài split.
- Random Forest: one-hot hoặc encoding được kiểm chứng; scaling thường không cần nhưng phải thống nhất input schema.
- XGBoost: chọn one-hot/sparse hoặc categorical support theo phiên bản thực tế; ghi rõ quyết định, không trộn hai cách trong cùng run.
- Unknown category phải được xử lý khi inference; mọi imputer/encoder/scaler/selector fit trên training partition.
- Không dùng oversampling trước split. Nếu thử resampling, chỉ áp dụng trong training fold bằng pipeline phù hợp và ghi dependency/phương pháp.

### 5.3. Đánh giá

- ROC-AUC là metric chính, tính từ probability/decision score trên validation.
- PR-AUC nên dùng average precision theo định nghĩa đã chốt trong protocol; không đổi cách tính giữa model.
- Precision, recall và F1 ghi kèm threshold. Báo cáo threshold 0,5 nếu protocol yêu cầu và có thể thêm threshold được chọn chỉ từ training/inner validation.
- Lưu confusion matrix chỉ như metric theo threshold, không thay thế ROC-AUC/PR-AUC.
- So sánh train và validation ROC-AUC/PR-AUC để nhận diện overfitting; không chọn model chỉ vì train score cao.

## 6. Các bước triển khai chi tiết

### 6.1. Xác nhận input và khởi tạo run

1. Nạp cấu hình dữ liệu và validation; xác nhận `split_id` đúng bản bàn giao.
2. Đọc `train.csv` qua module dữ liệu dùng chung, tách ID/target trước preprocessing.
3. Kiểm tra indices train/validation disjoint, cover đúng tập train và giữ cả hai lớp.
4. Ghi `run_id`, model name, thời gian bắt đầu, seed chung, seed model, phiên bản Python/thư viện và git commit nếu có.
5. Lưu config bất biến cho mỗi run; không sửa config đã dùng rồi gắn lại cùng `run_id`.

Quy ước `run_id` đề xuất: `YYYYMMDD-HHMM_<model>_<split_id>_<short-config-hash>`.

### 6.2. Xây dựng Logistic Regression baseline

1. Dùng `ColumnTransformer` theo feature contract.
2. Numerical: imputer dự phòng + scaler; categorical: imputer dự phòng + `OneHotEncoder(handle_unknown="ignore")`; binary: ánh xạ/encode ổn định.
3. Ghép preprocessing và `LogisticRegression` trong một scikit-learn `Pipeline`.
4. Chọn solver phù hợp sparse input và kích thước dữ liệu; đặt `max_iter` đủ, lưu cảnh báo hội tụ.
5. Bắt đầu với cấu hình đơn giản. Có thể so sánh `class_weight=None` và `class_weight="balanced"` nếu protocol cho phép; không giả định weighting chắc chắn tốt hơn.
6. Fit chỉ trên training partition, dự đoán probability cho train và validation.
7. Nếu solver không hội tụ, xử lý scaling/solver/tolerance/max_iter có kiểm soát; không bỏ qua cảnh báo rồi báo kết quả như ổn định.
8. Lưu pipeline hoàn chỉnh để inference có đúng encoder/scaler.

### 6.3. Xây dựng mô hình phi tuyến

#### Phương án A Random Forest

- Dùng khi cần bám dependency hiện có và có đủ RAM/thời gian.
- Khởi đầu với số cây giới hạn, `n_jobs` phù hợp máy, seed cố định.
- Kiểm soát `max_depth`, `min_samples_leaf`, `max_features` và class weighting.
- Theo dõi train-vs-validation gap; cây quá sâu có thể overfit.
- Xem xét sparse/dense memory trước khi one-hot; không tự chuyển ma trận lớn sang dense.

#### Phương án B XGBoost

- Chỉ dùng khi dependency/version đã được nhóm quản lý.
- Dùng objective binary probability và evaluation metric phù hợp; prediction phải là xác suất.
- Giới hạn `n_estimators`, depth/leaves, learning rate, subsampling và column sampling.
- Nếu dùng early stopping, evaluation set là validation theo protocol; không tiếp tục tuning vô hạn trên cùng validation.
- Ghi rõ xử lý class imbalance (`scale_pos_weight` hoặc không) và chứng minh bằng validation.

Chỉ một trong hai phương án là P0 bắt buộc. Phương án còn lại có thể trở thành mô hình thứ ba P1 nếu còn thời gian.

### 6.4. Hyperparameter tuning có giới hạn

1. Chốt ngân sách trước khi chạy: số cấu hình tối đa, thời gian tối đa và tài nguyên CPU/RAM.
2. Dùng coarse search nhỏ hoặc randomized search; ưu tiên tham số có tác động lớn.
3. Mỗi candidate phải dùng cùng split/fold theo protocol và cùng cách tính metric.
4. Nếu dùng cross-validation nội bộ, preprocessing/resampling phải nằm trong pipeline của từng fold.
5. Không tune threshold cùng dữ liệu rồi báo precision/recall như ước lượng độc lập. Ghi rõ mục đích threshold.
6. Dừng tuning khi cải thiện không ổn định hoặc đe dọa deadline; giữ baseline mạnh và tái lập được.

Ngân sách là cấu hình của run, không hard-code trong guide. Roadmap ưu tiên hoàn tất integration hơn việc tìm kiếm rộng.

### 6.5. Đánh giá và phân tích overfitting

Với mỗi run:

- Tính train/validation ROC-AUC và PR-AUC.
- Tính precision, recall, F1 tại threshold đã quy định; lưu threshold.
- Ghi thời gian fit, prediction và peak memory nếu có công cụ đáng tin cậy.
- Lưu ROC/PR curve data hoặc figure được sinh tự động.
- So sánh train-val gap và learning behavior nếu model hỗ trợ.
- Kiểm tra calibration chỉ như phân tích bổ sung; không thay ROC-AUC bằng calibration metric.
- Không báo số liệu nếu run lỗi, split khác hoặc output không qua schema check.

### 6.6. Feature importance và diễn giải

- Logistic Regression: coefficient chỉ diễn giải sau khi xác nhận encoding/scaling và mapping tên feature; tránh xem magnitude là quan hệ nhân quả.
- Random Forest/XGBoost: impurity/gain importance có thiên lệch; có thể bổ sung permutation importance trên validation với chi phí giới hạn.
- Không dùng validation để lặp feature selection quá nhiều.
- Ghi top feature cùng phương pháp, split và caveat; đối chiếu với nguy cơ leakage của Người 1.
- Nếu feature đáng ngờ đứng đầu, dừng kết luận và yêu cầu kiểm tra nguồn/thời điểm khả dụng.

### 6.7. Lưu và tái sử dụng model

1. Lưu pipeline gồm preprocessing + model tại `outputs/models/` với `run_id` trong tên.
2. Lưu metadata/config/checksum artifact trong `experiments/ml/`; không dựa vào tên file model để suy ra toàn bộ config.
3. Tạo hàm load model và predict probability; không refit khi inference.
4. Sau load, chạy cùng một sample cố định và xác nhận probability khớp trong tolerance.
5. Ghi phiên bản thư viện vì pickle/joblib không đảm bảo tương thích mọi phiên bản.
6. Artifact model bị Git ignore; hướng dẫn nơi chia sẻ/nạp lại phải rõ trong bàn giao.

### 6.8. Chuẩn hóa prediction outputs

Validation prediction nội bộ nên có tối thiểu:

| Cột | Ý nghĩa |
|---|---|
| `id` | ID gốc của hàng validation |
| `y_true` | Nhãn thật |
| `is_fraud` | Xác suất model trong `[0,1]` |
| `run_id` | Run tạo prediction |
| `model` | Tên model |
| `split_id` | Split dùng chung |

Test prediction bàn giao cho integration phải có `id,is_fraud`, đúng số dòng/thứ tự test, không missing/inf và xác suất hữu hạn trong `[0,1]`. Metadata `run_id`, `split_id`, artifact/config path được bàn giao kèm nhưng không thêm vào submission cuối.

### 6.9. Chọn ứng viên ML bàn giao

- Chỉ so sánh các run qua đủ kiểm tra và cùng protocol.
- Chọn ứng viên dựa trên validation ROC-AUC chính, PR-AUC, độ ổn định, overfit gap, khả năng inference và tài nguyên.
- Chênh lệch nhỏ phải được báo trung thực; không khẳng định ưu thế nếu chưa kiểm tra ổn định.
- Giữ Logistic baseline trong bảng kể cả khi không được chọn.
- Người 3 chọn model cuối toàn dự án sau khi so sánh ML/DL; Người 2 chỉ đề xuất ứng viên ML.

## 7. Đường dẫn file cần tạo hoặc chỉnh sửa

Đây là cấu trúc triển khai đề xuất; chỉ tạo file có nội dung chạy được:

| Đường dẫn | Hành động | Nội dung |
|---|---|---|
| `configs/ml.yaml` | Tạo | Model, seed, split/config path, tuning budget |
| `src/models/ml.py` | Tạo | Factory Logistic và tree/boosting model |
| `src/training/train_ml.py` | Tạo | Fit/evaluate/save theo run config |
| `src/evaluation/metrics.py` | Tạo hoặc dùng chung | ROC-AUC, PR-AUC và threshold metrics |
| `src/prediction/predict_ml.py` | Tạo | Load pipeline và xuất probabilities |
| `experiments/ml/` | Ghi artifact nhẹ | Config snapshot, metrics, prediction manifest |
| `outputs/models/` | Ghi artifact sinh ra | Pipeline/model đã fit, bị Git ignore |
| `outputs/logs/` | Ghi artifact sinh ra | Log có `run_id`, bị Git ignore |
| `tests/test_ml_pipeline.py` | Tạo | Fit/predict smoke test và schema |
| `tests/test_metrics.py` | Tạo | Metric từ probability, edge cases |
| `tests/test_model_serialization.py` | Tạo | Save/load prediction equivalence |

Nếu chọn XGBoost, thay đổi dependency chỉ được thực hiện như một quyết định riêng của nhóm; guide này không sửa `requirements.txt`.

## 8. Kiểm thử và phương pháp đánh giá

### 8.1. Unit/smoke tests

- Pipeline fit trên sample train nhỏ và tạo probability đúng shape.
- `predict_proba`/score được chuẩn hóa thành xác suất lớp fraud, không lấy nhầm cột lớp 0.
- Probability hữu hạn, trong `[0,1]`; số hàng và ID không đổi.
- Metric trả lỗi rõ với target một lớp hoặc input sai độ dài.
- Unknown category ở inference không làm pipeline vỡ.
- Save/load tạo prediction tương đương trong tolerance.

### 8.2. Validation checks

- `split_id` khớp protocol và prediction của DL để so sánh.
- Không giao nhau train/validation, không có ID test trong validation.
- Preprocessor fit train-only; không target encode/resample toàn bộ dữ liệu.
- ROC-AUC/PR-AUC tính từ probabilities, không từ nhãn threshold.
- Precision/recall/F1 kèm threshold.
- Run metadata có seed, params, preprocessing, feature list, thời gian và artifact path.

### 8.3. Integration checks

- Model load trên môi trường sạch theo dependency đã ghi.
- Test probabilities khớp test ID và có đúng 299.984 dòng theo audit.
- Người 3 có thể gọi inference mà không cần notebook state hoặc refit.

## 9. Đầu ra và sản phẩm bàn giao

- Logistic Regression baseline chạy được theo split chính thức.
- Ít nhất một XGBoost hoặc Random Forest chạy được và được đánh giá thật.
- Config/metadata/metrics cho từng run; không điền metric giả.
- Pipeline/model artifact cùng hàm load/inference.
- Validation probabilities chuẩn hóa cho mọi model được đưa vào so sánh.
- Test probabilities của ứng viên ML được chọn, kèm manifest và schema check.
- Bảng kết quả ML gồm ROC-AUC, PR-AUC, precision, recall, F1, threshold, train time và ghi chú overfitting.
- Phần mô tả phương pháp/giới hạn ML để tích hợp vào báo cáo.

## 10. Tiêu chí nghiệm thu Definition of Done

- [ ] Input/split/feature contract được kiểm tra trước run chính thức.
- [ ] Logistic Regression P0 hoàn thành và hội tụ hoặc cảnh báo được xử lý rõ.
- [ ] Ít nhất một mô hình phi tuyến P0 hoàn thành: XGBoost hoặc Random Forest.
- [ ] Mọi model dùng cùng `split_id` với DL cho bảng so sánh chính.
- [ ] Preprocessing nằm trong pipeline và fit train-only.
- [ ] ROC-AUC chính, PR-AUC và threshold metrics được tính đúng từ validation thật.
- [ ] Seed, hyperparameters, feature/preprocessing, phiên bản, thời gian và artifact path được lưu.
- [ ] Model save/load được kiểm thử; inference không refit.
- [ ] Validation/test probabilities qua schema/range/order checks.
- [ ] Người 3 xác nhận nạp được model hoặc prediction output ML.
- [ ] Không tự mở rộng model list, không metric giả và không dùng leaderboard để tuning.

## 11. Deadline và checklist công việc

| Hạn | Công việc | Bàn giao/Gate |
|---|---|---|
| 19/11 | Chuẩn bị code/config, kiểm tra dependency; chỉ smoke test nếu chưa có split | Skeleton có thể chạy |
| 20/11 | Baseline trên split tạm nếu hợp lệ, không báo là kết quả chính | Phát hiện lỗi sớm |
| 21/11 | Chạy lại Logistic theo split chính thức | Gate A: baseline + probabilities |
| 22/11 | Huấn luyện Random Forest hoặc XGBoost | Có ít nhất 2 ML thử nghiệm |
| 23-24/11 | Tuning nhỏ, so sánh và xuất validation probabilities | Bảng metric ML phiên bản đầu |
| 25/11 | Đóng gói inference, dự đoán test ứng viên ML | Hỗ trợ submission thử |
| 26/11 | Sửa lỗi inference, phân tích ML | Hỗ trợ Gate B |
| 27/11 | Viết phương pháp, tuning và error/importance analysis | Nội dung báo cáo ML |
| 28/11 | Chốt cấu hình ML và bàn giao ứng viên | Pipeline freeze |
| 29/11 | Test chạy lại từ đầu và inference | Sẵn sàng submission cuối |
| 30/11 | Rà soát code, kết quả và slide | Bàn giao trước hạn |

Nếu đến 25/11 chưa có submission, ưu tiên model ML ổn định tốt nhất để xuất CSV thay vì tiếp tục tuning.

## 12. Rủi ro và phương án xử lý

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| XGBoost chưa có dependency | Import lỗi/khác phiên bản | Dùng Random Forest P0; không tự cài ngoài kế hoạch |
| One-hot tốn RAM | Dense matrix lớn, OOM | Giữ sparse; kiểm tra estimator support; giảm song song |
| Logistic không hội tụ | ConvergenceWarning | Kiểm tra scaling/solver/max_iter; lưu log và chạy lại |
| Overfitting mô hình cây | Train AUC cao, val thấp | Giới hạn depth/leaves, min samples, subsampling; dừng tuning |
| Imbalance làm metric ngưỡng kém | ROC-AUC ổn nhưng recall/precision xấu | So sánh weighting có kiểm chứng; theo dõi PR-AUC; ghi threshold |
| Leakage qua tuning/preprocessing | Metric tăng bất thường | Pipeline trong fold; code review; vô hiệu hóa run lỗi |
| Model artifact không chia sẻ được | `.gitignore` chặn `.joblib` | Bàn giao đường dẫn/kho lưu thống nhất và config tái tạo |
| Inference sai cột xác suất | Xác suất đảo lớp | Kiểm tra `classes_` và test case lớp fraud |
| Tuning lấn deadline | Nhiều run chưa tổng hợp | Giới hạn ngân sách; đóng model P0, ưu tiên inference/report |

## 13. Phối hợp và bàn giao giữa các thành viên

### Nhận từ Data và EDA

Người 2 xác nhận bằng văn bản/run manifest rằng đã nhận đúng feature groups, indices, `split_id`, seed, metric và leakage guard. Nếu cần thay preprocessing cho estimator, chỉ thay phần thuật toán sở hữu, không thay hàng validation.

### Bàn giao cho DL và Integration

Người 2 cung cấp:

- Validation probabilities với ID, `y_true`, `run_id`, model và `split_id`.
- Test probabilities `id,is_fraud` cho ứng viên ML.
- Pipeline/model artifact hoặc lệnh tái tạo; config, version và checksum nếu có.
- Hàm/lệnh load-inference và kết quả test serialization.
- Bảng metric, threshold, thời gian, overfitting và giới hạn.

Người 3 kiểm tra schema và so sánh trên cùng validation; không chỉnh xác suất ML mà không tạo run mới có metadata.

### Kiểm tra chéo cuối

- Người 1 xác nhận feature/split và không leakage.
- Người 2 xác nhận model ML được load đúng và prediction không bị đảo lớp/thứ tự.
- Người 3 xác nhận integration/submission dùng đúng run đã chọn.
- Cả nhóm thống nhất model cuối dựa trên kết quả thực nghiệm, không dựa trên vai trò hoặc giả định ML/DL tốt hơn.
