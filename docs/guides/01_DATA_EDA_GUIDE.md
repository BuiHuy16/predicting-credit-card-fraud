# Hướng dẫn triển khai Data và EDA

**Người thực hiện:** Người 1 - Data & EDA

**Thời hạn hoàn thành:** 30/11/2026; bàn giao split và giao thức validation chậm nhất 21/11/2026

**Nguồn phạm vi:** `roadmap.md` và `docs/01_DATA_AUDIT.md`

Tài liệu này hướng dẫn Người 1 hoàn thiện EDA, chốt giao thức validation và xây dựng lớp dữ liệu dùng chung cho cả ML và DL. Kết quả quan trọng nhất không chỉ là notebook có biểu đồ, mà là một hợp đồng dữ liệu tái lập được: cùng hàng train/validation, cùng định nghĩa feature/target, mọi phép biến đổi học tham số chỉ fit trên train, và đầu ra đủ rõ để Người 2 và Người 3 triển khai song song.

## 1. Mục tiêu

- Hoàn thiện `notebooks/01_eda.ipynb` bằng số liệu và biểu đồ tính từ dữ liệu thật.
- Xác định nguy cơ drift, outlier, quan hệ feature-target và data leakage trước khi huấn luyện.
- Chốt một giao thức validation dùng chung, ghi tại `docs/02_VALIDATION_PROTOCOL.md` và cung cấp cùng tập chỉ số dòng cho ML/DL.
- Xây dựng preprocessing có thể fit lại, không leakage, có nhánh chung và nhánh phù hợp từng họ thuật toán.
- Bàn giao schema, split, feature contract và kiểm thử để Người 2/3 có thể chạy độc lập.

Không chạy lại toàn bộ Initial Data Audit nếu checksum và schema không đổi. Chỉ kiểm tra nhanh điều kiện đầu vào, rồi tập trung vào các câu hỏi chưa được audit giải đáp.

## 2. Phạm vi trách nhiệm

Người 1 chịu trách nhiệm:

- EDA chuyên sâu cho numerical, categorical và binary features.
- Phân tích target, tỷ lệ fraud theo feature, outlier, tương quan và train/test drift.
- Điều tra các feature có nguy cơ là proxy của target hoặc chứa thông tin không có tại thời điểm dự đoán.
- Đề xuất và chốt split với cả nhóm; xuất chỉ số train/validation ổn định.
- Xây dựng phần đọc dữ liệu, kiểm tra schema và preprocessing dùng chung trong `src/data/`.
- Định nghĩa các biến đổi riêng mà Người 2/3 phải tự fit trong pipeline của mô hình.
- Viết kiểm thử dữ liệu/split/preprocessing và tài liệu hóa quyết định.

Ngoài phạm vi:

- Huấn luyện/tuning mô hình ML hoặc MLP để kết luận mô hình tốt nhất.
- Dùng `test.csv` như tập validation hoặc suy đoán nhãn test.
- Tự động loại bỏ giao dịch chỉ vì chúng là outlier; outlier có thể chính là tín hiệu fraud.
- Thay đổi ba CSV trong `datasets/raw/`.

## 3. Trạng thái hiện tại

Đã xác nhận trong Initial Data Audit ngày 19/11/2026:

- `train.csv`: 699.960 dòng, 26 cột; `test.csv`: 299.984 dòng, 25 cột; sample submission có 299.984 dòng.
- Sau khi bỏ `id` và target, dữ liệu có 13 numerical, 5 categorical dạng chuỗi và 6 binary dạng boolean.
- `train.csv` không có missing, duplicate toàn dòng, duplicate ID, giá trị vô hạn hoặc target ngoài `{0, 1}`.
- Target `is_fraud` có 11.689 mẫu fraud, tương ứng 1,669953%; tỷ lệ majority/minority khoảng 58,8819:1.
- Train và test có cùng 25 cột đầu vào, cùng thứ tự và dtype. Năm categorical có cùng tập category; kết luận này chưa thay thế kiểm tra drift phân phối.
- `id` là định danh duy nhất, tuần tự và tách biệt giữa train/test; mặc định loại khỏi feature cho đến khi có bằng chứng hợp lệ.
- ID của test và sample submission khớp hoàn toàn cả giá trị lẫn thứ tự.

Chưa hoàn thành:

- EDA phân phối, outlier, fraud rate theo feature, tương quan, drift và leakage.
- Split/seed chính thức; `docs/02_VALIDATION_PROTOCOL.md` chưa tồn tại.
- Module dữ liệu, preprocessing, kiểm thử và artifact chỉ số split.
- Bất kỳ model, metric validation hay kết quả Kaggle nào.

## 4. Điều kiện đầu vào và dependencies

### 4.1. Đầu vào bắt buộc

| Đầu vào | Cách dùng | Điều kiện chấp nhận |
|---|---|---|
| `datasets/raw/train.csv` | EDA và tạo split | Chỉ đọc; có `id`, 24 feature và `is_fraud` |
| `datasets/raw/test.csv` | Kiểm tra schema/drift | Chỉ đọc; không dùng để đánh giá có giám sát |
| `datasets/raw/sample_submission.csv` | Chuẩn schema submission | Chỉ đọc; không xem giá trị hằng là dự đoán model |
| `docs/01_DATA_AUDIT.md` | Nguồn số liệu audit đã xác nhận | Dùng lại thay vì lặp toàn bộ audit |
| `roadmap.md` | Phạm vi, vai trò, deadline, metric | Không mở rộng phạm vi nếu chưa thống nhất |

### 4.2. Kiểm tra đầu vào tối thiểu

Trước EDA, xác nhận file tồn tại, tên cột/shape khớp audit và target chỉ có 0/1. Nếu checksum khác các giá trị trong audit, dừng và điều tra thay đổi trước khi tiếp tục. Không ghi đè hoặc chuẩn hóa trực tiếp file raw.

### 4.3. Dependencies giữa thành viên

- Người 2/3 có thể dựng skeleton song song, nhưng chỉ chạy kết quả chính thức sau khi nhận `split_id`, chỉ số split và quy tắc preprocessing.
- Người 2/3 phải phản hồi nếu pipeline mô hình cần sparse/dense tensor, scaling hoặc encoding khác; Người 1 giữ hợp đồng schema, không ép mọi model dùng cùng ma trận biến đổi.
- Người 3 cần schema ID và prediction contract để tích hợp; Người 1 đối chiếu lại submission cuối.

## 5. Công nghệ, thuật toán và phương pháp

- Python, pandas và NumPy để đọc/kiểm tra dữ liệu.
- JupyterLab cho `notebooks/01_eda.ipynb`.
- Matplotlib và seaborn cho biểu đồ; lưu hình cần dùng lại tại `outputs/figures/` khi pipeline thực tế được triển khai.
- scikit-learn cho split, `ColumnTransformer`, `Pipeline`, imputation dự phòng, encoding và scaling.
- PyYAML nếu cấu hình dữ liệu được lưu tại `configs/data.yaml`.
- pytest cho kiểm thử schema, split và preprocessing.

Phương pháp validation mặc định để xem xét là holdout stratified với seed cố định. Đây không phải kết luận tự động: nếu EDA phát hiện cấu trúc thời gian hoặc nhóm mà random split làm rò rỉ/đánh giá quá lạc quan, phải chuyển sang chiến lược theo thời gian/nhóm và ghi lý do. Seed dùng chung đề xuất là `42`; seed chỉ được xem là chính thức khi ghi trong `docs/02_VALIDATION_PROTOCOL.md`.

Metric chính là ROC-AUC; theo dõi thêm PR-AUC. Precision, recall và F1 phải ghi rõ threshold. Accuracy không được dùng làm chỉ báo chính với target mất cân bằng mạnh.

## 6. Các bước triển khai chi tiết

### 6.1. Tái xác nhận dữ liệu và dựng notebook

1. Tạo `notebooks/01_eda.ipynb` với các phần: mục tiêu, tải dữ liệu, kiểm tra nhanh, univariate, target relationship, outlier, correlation, drift, leakage, validation recommendation và kết luận.
2. Đọc ba CSV bằng đường dẫn tương đối từ repository root; không hard-code đường dẫn máy cá nhân.
3. Đặt hằng số `TARGET = "is_fraud"`, `ID_COLUMN = "id"`; tạo danh sách feature theo audit nhưng kiểm tra lại từ dtype/schema trong code.
4. Ghi phiên bản Python/thư viện, thời điểm chạy và seed. Kernel phải chạy lại từ đầu không phụ thuộc state ẩn.
5. Dùng số liệu audit cho phần tổng quan; chỉ tính lại thống kê cần cho EDA hoặc kiểm tra bất biến.

### 6.2. Phân tích numerical features

Với từng numerical feature:

- Báo cáo min, quantile, median, mean, max, độ lệch và số giá trị duy nhất.
- Vẽ histogram/density hoặc ECDF; dùng log scale bổ sung khi phân phối lệch, không thay thế biểu đồ gốc mà không giải thích.
- So sánh phân phối giữa `is_fraud=0` và `is_fraud=1` bằng cùng bin hoặc quantile bin.
- Tính fraud rate và số mẫu theo bin; luôn hiển thị denominator để tránh kết luận từ nhóm quá nhỏ.
- Với `time_of_day_hour` và `day_of_week`, xem như biến rời rạc/cyclic; cân nhắc biểu diễn sin/cos như thử nghiệm sau, không mặc định là cải thiện.
- Đánh dấu range train/test khác biệt và giá trị hiếm để đưa vào kiểm tra drift.

Không điền số liệu thủ công vào notebook. Mọi bảng/biểu đồ phải sinh từ DataFrame đã đọc.

### 6.3. Phân tích categorical và binary features

Với 5 categorical và 6 binary features:

- Lập bảng count, frequency, fraud count và fraud rate cho từng giá trị.
- Với nhóm ít mẫu, hiển thị cả count và khoảng bất định/ghi chú thận trọng; không kết luận chỉ từ tỷ lệ cao.
- So sánh frequency giữa train/test, kể cả khi tập category giống nhau.
- Kiểm tra interaction có lý do nghiệp vụ, ưu tiên các cặp như channel-device/auth và các cờ mismatch; tránh tạo tổ hợp hàng loạt không kiểm soát.
- Xác nhận boolean được ánh xạ nhất quán, không phụ thuộc vào thứ tự category.

### 6.4. Phân tích outlier

1. Dùng boxplot, quantile/IQR và percentile để nhận diện, không dùng một quy tắc xóa chung.
2. Đối chiếu fraud rate trong vùng đuôi so với vùng trung tâm và kiểm tra sample size.
3. Phân biệt lỗi dữ liệu bất khả thi với giao dịch hiếm nhưng hợp lệ. Chỉ đề xuất lọc nếu có bằng chứng lỗi rõ ràng.
4. Nếu thử clipping/winsorization/log transform, triển khai như biến thể pipeline fit trên train và so sánh validation; không sửa raw data.
5. Ghi riêng quyết định cho từng feature, gồm lý do giữ nguyên, transform hay cần kiểm tra thêm.

### 6.5. Tương quan và data leakage

- Tính Pearson/Spearman cho numerical khi phù hợp; với binary/categorical dùng thống kê liên hệ phù hợp và tránh diễn giải correlation như quan hệ nhân quả.
- Tìm feature trùng lặp, gần như xác định lẫn nhau hoặc proxy target.
- Với các tên như `merchant_risk_score` và `is_ai_generated_scam_attempt`, điều tra cách feature được tạo và thời điểm khả dụng. Tên gợi ý không đủ để kết luận leakage.
- Kiểm tra xem bất kỳ feature nào được tính sau khi giao dịch đã được xác nhận fraud hay sử dụng nhãn trong quy trình tạo.
- Không dùng target encoding toàn bộ dữ liệu. Nếu thử encoding dựa trên target, phải tính trong train fold và áp dụng out-of-fold đúng cách.
- Ghi kết luận và phần chưa xác minh; loại feature chỉ khi có căn cứ và có phiên bản thí nghiệm đối chứng.

### 6.6. Kiểm tra train/test drift

1. So sánh quantile/range cho numerical và frequency cho categorical/binary.
2. Dùng chỉ số drift như PSI, KS hoặc Jensen-Shannon khi phù hợp, nhưng kèm biểu đồ và sample size; không dùng một ngưỡng máy móc làm kết luận duy nhất.
3. Có thể huấn luyện bộ phân loại train-vs-test chỉ như công cụ chẩn đoán. Nếu dùng, loại `id`, dùng validation riêng và không biến kết quả thành metric model fraud.
4. Liệt kê feature drift đáng chú ý, hướng xử lý và rủi ro đối với validation.
5. Không dùng nhãn giả cho test và không tối ưu trực tiếp theo Public Leaderboard.

### 6.7. Chốt validation protocol

Thực hiện theo thứ tự:

1. Kiểm tra có khóa nhóm/thời gian thực sự hay không. `id`, `time_of_day_hour` và `day_of_week` không tự động tạo thành timestamp hoặc customer group.
2. Nếu không có cấu trúc nhóm/thời gian đủ căn cứ, dùng stratified holdout với tỷ lệ validation được nhóm chốt, seed `42`, và giữ nguyên tỷ lệ fraud.
3. Nếu có cấu trúc hợp lệ, dùng group/time split; tài liệu hóa cột, quy tắc sắp xếp, gap (nếu có) và lý do.
4. Xuất indices hoặc ID của train/validation thành artifact tái lập. Artifact sinh ra không được chứa feature/target biến đổi và phải tuân theo quy tắc Git của nhóm.
5. Tạo `split_id` ổn định, ví dụ từ tên chiến lược, seed và hash danh sách validation ID. Mọi run ML/DL phải ghi cùng `split_id`.
6. Ghi metric, cách tính PR-AUC, threshold báo cáo, quy tắc chọn model, cách fit transform và các kiểm tra leakage tại `docs/02_VALIDATION_PROTOCOL.md`.
7. Cả ba thành viên xác nhận giao thức trước khi coi kết quả là chính thức.

Không dùng Kaggle test có nhãn giả, sample submission hay điểm leaderboard để chọn split.

### 6.8. Xây dựng preprocessing

Phần chung:

- Hàm đọc dữ liệu và kiểm tra schema/target/ID.
- Tách `X`, `y`, `id`; loại `id` khỏi feature mặc định.
- Chọn cột theo tên/dtype đã kiểm tra, không dựa vào vị trí cột mơ hồ.
- Imputer là bước dự phòng cho dữ liệu suy luận tương lai dù audit hiện không có missing; mọi thống kê imputation fit trên train.
- Unknown category phải được xử lý an toàn khi inference.
- Đầu ra giữ thứ tự dòng và cho phép đối chiếu ID.

Nhánh theo thuật toán:

| Họ mô hình | Biến đổi đề xuất | Chủ sở hữu fit cuối |
|---|---|---|
| Logistic Regression | One-hot categorical, binary rõ ràng, scaling numerical | Người 2 trong pipeline ML |
| Random Forest/XGBoost | Encoding tương thích, thường không cần scaling; xác minh hỗ trợ sparse/category | Người 2 |
| MLP PyTorch | Encoding nhất quán, numerical scaling, tensor `float32`; cân nhắc sparse/dense theo bộ nhớ | Người 3 |

Người 1 cung cấp factory/contract và quy tắc; Người 2/3 fit transformer trong pipeline/run của mình chỉ trên training partition. Không fit một transformer toàn bộ train rồi chia validation.

### 6.9. Tạo hợp đồng bàn giao

Hợp đồng phải ghi:

- Tên/kiểu cột ID, target, numerical, categorical và binary.
- Danh sách feature được dùng/loại và lý do.
- `split_id`, seed, phương pháp split, số dòng và tỷ lệ lớp từng phần.
- Cách tái tạo hoặc nạp indices split.
- Giao diện transformer và dạng đầu ra: sparse/dense, dtype, thứ tự feature, xử lý unknown.
- Prediction validation phải giữ ID, `y_true`, probability, `run_id`, `model` và `split_id`.
- Prediction test dùng đúng ID theo thứ tự test; file submission cuối chỉ có `id,is_fraud`.

## 7. Đường dẫn file cần tạo hoặc chỉnh sửa

Các file dưới đây là đầu ra dự kiến khi triển khai, chưa được xem là đã tồn tại:

| Đường dẫn | Hành động | Nội dung |
|---|---|---|
| `notebooks/01_eda.ipynb` | Tạo | EDA chạy lại từ đầu, biểu đồ và kết luận |
| `docs/02_VALIDATION_PROTOCOL.md` | Tạo | Split, seed, metric, leakage guard và hợp đồng đánh giá |
| `configs/data.yaml` | Tạo khi pipeline cần | Đường dẫn tương đối, seed, split strategy, feature groups |
| `src/data/loading.py` | Tạo | Đọc CSV và kiểm tra schema |
| `src/data/splitting.py` | Tạo | Tạo/nạp split ổn định và `split_id` |
| `src/data/preprocessing.py` | Tạo | Factory preprocessing/feature contract |
| `tests/test_data_schema.py` | Tạo | Kiểm tra schema, target, ID, train/test compatibility |
| `tests/test_split.py` | Tạo | Disjoint, coverage, reproducibility, class presence |
| `tests/test_preprocessing.py` | Tạo | Fit train-only, shape/dtype, unknown category, finite output |

Không tạo module khung rỗng hàng loạt. Tên file có thể điều chỉnh khi triển khai nếu vẫn nằm đúng package và được cập nhật nhất quán trong validation protocol/README sau này.

## 8. Kiểm thử và phương pháp đánh giá

### 8.1. Kiểm thử dữ liệu và split

- Raw file chỉ được mở đọc; checksum không đổi so với audit.
- Train/validation disjoint, hợp lại đúng toàn bộ tập train và không có ID test.
- Chạy lại cùng cấu hình tạo đúng indices và `split_id`.
- Mỗi partition có cả hai lớp; tỷ lệ lớp phù hợp chiến lược đã chọn.
- Target/ID không xuất hiện trong ma trận feature.

### 8.2. Kiểm thử preprocessing

- Gọi `fit` chỉ với training partition; validation/test chỉ `transform`.
- Số dòng và thứ tự ID không đổi sau transform.
- Output không NaN/inf, dtype và sparse/dense đúng contract.
- Category chưa thấy không làm pipeline lỗi.
- Feature order ổn định khi save/load transformer.

### 8.3. Đánh giá EDA

- Mọi nhận xét định lượng truy được về cell/bảng sinh từ dữ liệu.
- Biểu đồ có tiêu đề, trục, đơn vị, legend và sample size khi cần.
- Fraud rate đi kèm count; không diễn giải nhóm rất nhỏ như bằng chứng chắc chắn.
- Drift/leakage có kết luận, bằng chứng và giới hạn rõ ràng.

## 9. Đầu ra và sản phẩm bàn giao

- Notebook EDA đã Restart Kernel and Run All thành công.
- `docs/02_VALIDATION_PROTOCOL.md` ở trạng thái được cả nhóm chấp nhận, không mô tả như hoàn thành trước khi thực sự tạo và rà soát.
- Code đọc dữ liệu/split/preprocessing cùng kiểm thử tương ứng.
- Artifact hoặc quy tắc tái tạo split có `split_id`.
- Bản tóm tắt feature contract và điểm drift/leakage cần Người 2/3 lưu ý.
- Bảng bàn giao ghi phiên bản cấu hình, seed, shape/dtype đầu ra và đường dẫn artifact.

## 10. Tiêu chí nghiệm thu Definition of Done

- [ ] Không sửa ba CSV raw; checksum khớp audit hoặc khác biệt đã được dừng để điều tra.
- [ ] `notebooks/01_eda.ipynb` chạy sạch từ đầu và dùng số liệu thật.
- [ ] Đã phân tích numerical, categorical, binary, fraud rate, outlier, correlation, drift và leakage.
- [ ] Không tự động xóa outlier và không dùng `id` làm feature khi chưa có bằng chứng.
- [ ] Validation strategy, tỷ lệ split, seed, `split_id`, metric và threshold được ghi rõ.
- [ ] ML và DL nhận đúng cùng hàng validation.
- [ ] Mọi transform học tham số fit trên training partition tương ứng.
- [ ] Preprocessing contract phân biệt nhánh Logistic/tree/MLP.
- [ ] Kiểm thử schema, split và preprocessing đạt.
- [ ] Người 2 và Người 3 xác nhận có thể nạp split và tạo input đúng contract.

## 11. Deadline và checklist công việc

| Hạn | Công việc | Bàn giao/Gate |
|---|---|---|
| 19/11 | Đọc audit, kiểm tra nhanh input, dựng notebook | Dataset đọc được, phạm vi EDA rõ |
| 20/11 | Univariate, target relationship, outlier, leakage sơ bộ | Nhận xét dựa trên số liệu thật |
| 21/11 | Chốt split, seed, metric và contract | Gate A: `docs/02_VALIDATION_PROTOCOL.md` + split dùng chung |
| 22/11 | Hoàn thiện notebook và preprocessing đầu vào ổn định | Người 2/3 nhận bàn giao |
| 23-24/11 | Drift, kiểm tra chéo leakage và hỗ trợ pipeline | Phản hồi cho các run chính thức |
| 25-26/11 | Đối chiếu ID/schema submission và validation | Hỗ trợ Gate B |
| 27-28/11 | Viết phần Data/EDA/Validation, hoàn thiện biểu đồ | Nội dung báo cáo có nguồn |
| 29/11 | Rà soát README, khả năng chạy lại, submission cuối | Không thay đổi split phút cuối |
| 30/11 | Kiểm tra checklist và minh chứng | Bàn giao toàn bộ trước hạn |

Nếu split chưa chốt vào 21/11, tạm dừng tuning; ưu tiên sửa validation/preprocessing theo roadmap.

## 12. Rủi ro và phương án xử lý

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Leakage qua preprocessing | Metric cao bất thường, transformer fit trước split | Code review call order; test fit train-only; chạy lại toàn bộ run |
| Random split không phù hợp | Phát hiện nhóm/thời gian hoặc proxy ID | Chuyển group/time split và vô hiệu hóa kết quả cũ |
| Outlier bị loại nhầm | Fraud tập trung ở vùng đuôi | Giữ raw; thử transform trong pipeline và so sánh validation |
| Target mất cân bằng | Accuracy cao nhưng bắt fraud kém | Stratification; ROC-AUC chính, PR-AUC và recall/precision phụ |
| Ma trận one-hot quá lớn | RAM tăng, dense conversion lỗi | Giữ sparse cho ML; kiểm soát encoding và batch/dtype cho DL |
| Drift train/test | Feature distribution khác rõ | Ghi rủi ro; thử feature/transform có căn cứ, không tune theo test label |
| Bàn giao split mơ hồ | ML/DL dùng hàng validation khác | Bắt buộc `split_id`, indices/IDs, counts và test tái lập |
| EDA kéo dài | Chưa bàn giao đến 21/11 | Ưu tiên split, leakage, drift, contract; dừng biểu đồ trang trí |

## 13. Phối hợp và bàn giao giữa các thành viên

### Data đến ML

Người 1 bàn giao feature groups, split/indices, `split_id`, metric và preprocessing contract. Người 2 xác nhận Logistic Regression và mô hình cây nạp đúng split, không tự tạo split khác cho bảng so sánh chính. Mọi thay đổi feature cần phản hồi lại để cập nhật contract.

### Data đến DL

Người 1 bàn giao cùng split và quy tắc encoding/scaling; Người 3 xác nhận thứ tự feature, dtype `float32`, dense/sparse và mapping ID. Scaler/encoder của MLP phải được fit trên train partition và lưu cùng checkpoint/config.

### ML và DL đến Data

Người 2/3 gửi lại run metadata, shape input/prediction, danh sách feature thực tế và lỗi dữ liệu gặp phải. Người 1 kiểm tra chéo rằng các run chính thức dùng cùng `split_id` và không leakage.

### Data đến Integration

Trước khi tạo submission, Người 1 đối chiếu test ID với `sample_submission.csv`, số dòng, thứ tự cột, missing/inf và khoảng xác suất. Người 3 sở hữu bước tích hợp cuối; Người 1 không thay thế model hoặc probability khi kiểm tra schema.
