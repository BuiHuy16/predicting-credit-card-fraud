# Hướng dẫn triển khai Deep Learning và Integration

**Người thực hiện:** Người 3 - Deep Learning & Integration

**Thời hạn hoàn thành:** 30/11/2026; submission thử mục tiêu 26/11 và submission cuối 29/11/2026

**Nguồn phạm vi:** `roadmap.md`, `docs/01_DATA_AUDIT.md` và giao thức validation sẽ được tạo tại `docs/02_VALIDATION_PROTOCOL.md`

Tài liệu này hướng dẫn xây dựng MLP bằng PyTorch cho dữ liệu tabular và tích hợp kết quả ML/DL thành một pipeline dự đoán có thể kiểm tra. DL không được mặc định là tốt hơn ML. Model cuối được chọn bằng kết quả validation thật trên cùng `split_id`, khả năng tái lập và độ tin cậy của inference; blending chỉ là P2 khi có bằng chứng cải thiện.

## 1. Mục tiêu

- Xây dựng MLP PyTorch P0 nhận dữ liệu sau preprocessing và xuất logit cho binary classification.
- Tạo training/validation loop có metric, checkpoint, early stopping và reproducibility.
- Thử một số cấu hình MLP/treatment imbalance trong ngân sách, không mở rộng kiến trúc ngoài nhu cầu.
- Thu nhận prediction probabilities từ ML và DL trên cùng validation để so sánh công bằng.
- Chọn model cuối dựa trên bằng chứng; tùy chọn thử blending nếu còn thời gian và validation cải thiện.
- Dự đoán `datasets/raw/test.csv`, tạo và kiểm tra `outputs/submissions/submission.csv`.
- Ghi submission và điểm Kaggle thực, nếu có, tại `docs/04_KAGGLE_SUBMISSION.md`.

## 2. Phạm vi trách nhiệm

### 2.1. Deep Learning

- Chuẩn bị code PyTorch theo dependency đã có, không cài thêm trong phạm vi guide.
- Nhận và kiểm tra split/preprocessing contract từ Người 1.
- Triển khai dataset/dataloader, MLP, loss, optimizer, loop, checkpoint và evaluation.
- Quản lý seed, device, dtype, batch, early stopping và log thực nghiệm.
- Xuất validation/test probabilities đúng schema.

### 2.2. Integration

- Xác định và thực thi giao diện dữ liệu/prediction giữa ba phần.
- Thu thập run hợp lệ của ML/DL, kiểm tra cùng `split_id` và tính lại/đối chiếu metric.
- Tạo bảng so sánh tại `docs/03_MODEL_COMPARISON.md` khi có kết quả thật.
- Chọn model cuối; thử blending P2 chỉ sau khi các pipeline đơn lẻ ổn định.
- Chạy inference test, tạo submission, kiểm tra schema/ID/range và ghi nhận nộp Kaggle.
- Viết kiểm thử tích hợp và hướng dẫn tái lập.

Ngoài phạm vi:

- Thay split/feature contract một phía hoặc dùng test/leaderboard để tuning.
- Dùng LLM, web/backend hoặc cloud production trong giải pháp.
- Khẳng định DL tốt hơn vì kiến trúc phức tạp hơn.
- Ghi điểm Kaggle, ROC-AUC hoặc thứ hạng chưa xảy ra.

## 3. Trạng thái hiện tại

- `torch>=2.2,<3.0` có trong `requirements.txt`, nhưng repository chưa có code DL, model, training loop hoặc checkpoint.
- `src/models/`, `src/training/`, `src/evaluation/` và `src/prediction/` mới có `__init__.py`.
- Initial Data Audit xác nhận train có 699.960 dòng, target fraud 1,669953%, 13 numerical, 5 categorical và 6 binary features; không missing và schema train/test khớp.
- Test có 299.984 dòng; ID khớp sample submission cả giá trị lẫn thứ tự.
- Chưa có split chính thức, preprocessor fit, input dimension, model result, ML probabilities hay Kaggle submission thực.
- Các file `docs/02_VALIDATION_PROTOCOL.md`, `docs/03_MODEL_COMPARISON.md`, `docs/04_KAGGLE_SUBMISSION.md` là sản phẩm cần xây dựng ở các giai đoạn tương ứng, không phải tài liệu đã hoàn thành.

## 4. Điều kiện đầu vào và dependencies

### 4.1. Đầu vào từ Người 1

- Feature contract và danh sách cột dùng/loại.
- Training/validation indices, `split_id`, seed chung và validation protocol.
- Quy tắc encoder/scaler, input feature order, output dtype/shape.
- Cách xử lý unknown category và cách lưu/load preprocessor.

Preprocessor cho MLP phải fit trên training partition. Nếu Người 1 chỉ cung cấp factory/contract, Người 3 chịu trách nhiệm fit và lưu instance cùng run DL.

### 4.2. Đầu vào từ Người 2

- Validation probabilities với `id,y_true,is_fraud,run_id,model,split_id`.
- Test probabilities `id,is_fraud` cho ứng viên ML.
- Metric/config/artifact path và lệnh load/inference.
- Xác nhận prediction probability là lớp fraud và model load không refit.

### 4.3. Môi trường

- Python theo README; PyTorch, NumPy, pandas, scikit-learn, PyYAML và pytest theo requirements.
- Code phải chạy CPU; GPU là tăng tốc tùy chọn, không phải điều kiện duy nhất để tái lập.
- Device, phiên bản PyTorch, CUDA/cuDNN nếu có và hardware chính phải được log.
- Không giả định GPU tồn tại. Batch size phải điều chỉnh theo memory thực tế.

## 5. Công nghệ, thuật toán và phương pháp

### 5.1. Kiến trúc MLP khởi điểm

Roadmap gợi ý:

```text
Input sau preprocessing
  -> Linear(input_dim, 128) -> ReLU -> Dropout
  -> Linear(128, 64) -> ReLU -> Dropout
  -> Linear(64, 1)
  -> logit
```

Đây là điểm xuất phát, không phải kiến trúc tối ưu đã kiểm chứng. Có thể thay width/depth/dropout có giới hạn và lưu config từng run. Output layer không dùng sigmoid khi training với `BCEWithLogitsLoss`.

### 5.2. Loss và probability

- Dùng `BCEWithLogitsLoss` cho binary classification: model xuất tensor logit shape nhất quán, target float 0/1.
- Chỉ áp dụng `torch.sigmoid(logits)` khi cần probability cho metric/inference.
- Không áp dụng sigmoid trong model rồi lại dùng `BCEWithLogitsLoss`; việc này sai vì loss đã kết hợp sigmoid ổn định số học.
- Tính `pos_weight` từ training partition nếu thử weighting; không dùng validation/test để tính.
- So sánh weighted/unweighted bằng cùng protocol; không mặc định weighting cải thiện ROC-AUC/PR-AUC.

### 5.3. Optimizer và regularization

- Adam là lựa chọn khởi điểm; learning rate, weight decay, batch size, epochs và dropout là hyperparameters của run.
- Dùng early stopping theo validation metric đã chốt; lưu best checkpoint, không mặc định checkpoint epoch cuối là tốt nhất.
- Gradient clipping chỉ dùng khi có dấu hiệu gradient bất ổn và phải ghi config.
- Batch normalization là tùy chọn thí nghiệm, không thêm đồng thời quá nhiều biến thể làm khó diễn giải.

### 5.4. Metric

- ROC-AUC chính, PR-AUC phụ quan trọng do fraud hiếm.
- Precision, recall, F1 kèm threshold; loss dùng để tối ưu chứ không thay thế metric đánh giá.
- Metric validation tính trên toàn bộ probability/target của epoch, không trung bình AUC theo batch.
- Cùng định nghĩa PR-AUC và threshold với ML theo `docs/02_VALIDATION_PROTOCOL.md`.

## 6. Các bước triển khai chi tiết

### 6.1. Chuẩn bị dữ liệu cho PyTorch

1. Nạp split theo `split_id`, xác nhận disjoint/coverage/class counts.
2. Fit preprocessor MLP chỉ trên training partition; transform validation/test.
3. Xác nhận feature order, `input_dim`, output finite và không có ID/target trong tensor.
4. Chuyển feature sang `float32`; target sang float 0/1 với shape tương thích loss.
5. Giữ ID ngoài tensor để ghép prediction trở lại đúng hàng.
6. Nếu ma trận one-hot sparse không phù hợp dense MLP, đánh giá memory trước khi convert. Có thể dùng encoding/representation khác theo contract, không tự dense hóa toàn bộ rồi gây OOM.
7. Tạo `Dataset`/`DataLoader`; chỉ shuffle training loader, không shuffle validation/test hoặc phải giữ index để khôi phục thứ tự chắc chắn.
8. Dùng worker/pinned memory phù hợp OS/device; mặc định an toàn trước, tối ưu sau.

### 6.2. Kiểm soát seed và reproducibility

- Ghi seed split và seed training riêng.
- Set seed cho Python, NumPy, PyTorch CPU và CUDA nếu có.
- Seed DataLoader worker/generator khi dùng nhiều worker.
- Bật deterministic algorithms khi khả thi và ghi rõ giới hạn/chi phí; một số GPU operation vẫn có thể không hoàn toàn deterministic.
- Lưu feature/preprocessor config, model state, optimizer state nếu cần resume, epoch tốt nhất và metric tốt nhất.
- Không ghi đè checkpoint tốt nhất bằng checkpoint cuối kém hơn.

### 6.3. Xây dựng model

1. Model nhận `input_dim` từ dữ liệu transform, không hard-code theo số cột raw.
2. Tạo danh sách hidden dimensions/config rõ ràng.
3. Dùng ReLU làm activation khởi điểm; dropout chỉ active ở `model.train()`.
4. Output `Linear(..., 1)` trả logit. Chuẩn hóa shape bằng quy ước nhất quán, tránh `squeeze()` làm mất batch dimension khi batch size 1.
5. Kiểm tra forward pass với batch nhỏ: shape, dtype, finite logits và gradient tồn tại.
6. Ghi tổng số parameter để so sánh độ phức tạp giữa biến thể.

### 6.4. Training loop

Mỗi epoch:

1. Gọi `model.train()`; zero gradients, forward logits, tính loss, backward, optimizer step.
2. Tích lũy loss theo số mẫu, không trung bình sai giữa batch kích thước khác nhau.
3. Gọi validation loop sau epoch; không cập nhật gradient/model.
4. Log epoch, train loss, val loss, val ROC-AUC, val PR-AUC, learning rate và thời gian.
5. Cập nhật best checkpoint theo metric/chiều tối ưu đã chốt.
6. Tăng patience khi không cải thiện đủ `min_delta`; dừng khi hết patience.
7. Nếu loss NaN/inf, dừng run, lưu diagnostic và kiểm tra input/LR/gradient; không tiếp tục báo metric.

### 6.5. Validation loop

- Gọi `model.eval()` và `torch.no_grad()` hoặc inference mode.
- Thu thập logits/targets/IDs cho toàn bộ validation.
- Chuyển logits sang probability đúng một lần bằng sigmoid.
- Tính metric trên toàn tập; xác nhận probability finite và `[0,1]`.
- Xuất validation prediction đúng ID/thứ tự, kèm `run_id`, model và `split_id`.
- Tính metric bằng hàm dùng chung với ML hoặc đối chiếu hai implementation bằng test.

### 6.6. Xử lý class imbalance có kiểm chứng

Thứ tự ưu tiên thử nghiệm:

1. Baseline không weighting để có mốc đối chiếu.
2. `pos_weight = n_negative / n_positive` tính trên training partition, nếu tài nguyên cho phép.
3. Sampler/resampling chỉ khi có giả thuyết rõ; tránh dùng đồng thời weighted sampler và `pos_weight` mà không phân tích vì có thể bù hai lần.
4. Không oversample validation/test.
5. So sánh ROC-AUC, PR-AUC, threshold metrics, calibration và training stability; giữ phương án tốt hơn theo protocol, không theo train loss.

### 6.7. Thử nghiệm cấu hình MLP

Giới hạn search vào một số trục:

- Width/depth: ví dụ cấu hình roadmap và một biến thể nhỏ/lớn vừa phải.
- Dropout/weight decay.
- Learning rate.
- Batch size theo memory.
- Weighted/unweighted loss.

Mỗi run phải có config snapshot. Chỉ thay một nhóm yếu tố có thể diễn giải; không chạy tổ hợp rộng khi chưa có end-to-end pipeline. Nếu đến 24/11 MLP chưa ổn định, dùng MLP nhỏ với early stopping theo roadmap.

### 6.8. Checkpoint và inference DL

Checkpoint tối thiểu chứa:

- `model_state_dict`, architecture config và `input_dim`.
- Best epoch/metric, training seed, `split_id`, `run_id`.
- Preprocessor artifact/path/checksum và feature order.
- Phiên bản PyTorch/thiết bị chính; optimizer state nếu cần resume.

Khi load để inference:

1. Khởi tạo đúng architecture từ config.
2. Load state, chuyển device, gọi `eval()`.
3. Load đúng preprocessor; không refit trên test.
4. Predict theo batch, sigmoid logits, ghép ID và khôi phục thứ tự test.
5. Chạy equivalence test trên sample cố định trước khi tạo submission.

### 6.9. Thu thập và kiểm tra kết quả ML/DL

Chỉ nhận run vào bảng so sánh nếu:

- Cùng `split_id`, cùng validation IDs và cùng `y_true` theo thứ tự/khóa.
- Probability là lớp fraud, hữu hạn và trong `[0,1]`.
- Có config, seed, preprocessing, artifact path và thời gian.
- Metric được tính lại bằng hàm chung hoặc khớp trong tolerance.
- Không có flag leakage hoặc lỗi serialization/inference.

Merge prediction theo `id`, không nối theo vị trí mơ hồ. Sau merge, assert số hàng đúng, ID unique, không missing và `y_true` khớp giữa các file.

### 6.10. So sánh và chọn model cuối

Tạo `docs/03_MODEL_COMPARISON.md` bằng kết quả thực, gồm:

- Bảng model/run/split, ROC-AUC, PR-AUC, threshold metrics, train time, inference/reproducibility notes.
- Train-validation gap hoặc dấu hiệu overfitting.
- Rủi ro leakage/drift và tài nguyên.
- Lý do chọn model cuối; không chọn dựa trên độ phức tạp hoặc kỳ vọng DL thắng.
- Public LB nếu có phải ở cột riêng, không trộn với validation ROC-AUC.

Nếu metric gần nhau, ưu tiên pipeline ổn định, dễ load và ít rủi ro hơn; ghi nhận độ không chắc chắn.

### 6.11. Blending tùy chọn

Chỉ thử khi model ML và DL đơn lẻ đã hoàn thành, predictions cùng validation và còn thời gian:

1. Blend probabilities bằng trọng số đơn giản, ví dụ convex combination.
2. Giới hạn số trọng số thử; không tối ưu tinh vi trên một validation duy nhất.
3. Đánh giá bằng cùng metric và ghi run/config riêng.
4. Chỉ chọn blend nếu cải thiện đủ rõ và inference có thể tái lập.
5. Nếu không cải thiện, báo kết quả trung thực và dùng model đơn tốt nhất.

Blending là P2, không được làm chậm submission đầu tiên.

### 6.12. Tạo submission

1. Chọn run/model cuối theo quyết định đã ghi.
2. Load pipeline/checkpoint/preprocessor đã freeze; không refit trên test.
3. Đọc `datasets/raw/test.csv`, giữ `id` nguyên vẹn và đúng thứ tự.
4. Predict xác suất `is_fraud`; không threshold thành nhãn 0/1.
5. Tạo DataFrame theo đúng thứ tự cột của `sample_submission.csv`: `id,is_fraud`.
6. Kiểm tra đúng 299.984 dòng theo audit, ID unique/non-null và khớp test/sample cả giá trị lẫn thứ tự.
7. Kiểm tra probability numeric, finite, không missing và trong `[0,1]`.
8. Ghi `outputs/submissions/submission.csv`; vì thư mục bị Git ignore, lưu manifest/config tạo file tại `experiments/comparison/` hoặc tài liệu.
9. Mở lại CSV vừa ghi và chạy toàn bộ schema check lần cuối để phát hiện lỗi serialization/index.

### 6.13. Ghi nhận Kaggle

Sau khi nộp thật, tạo/cập nhật `docs/04_KAGGLE_SUBMISSION.md` với:

- Thời gian nộp và tên file.
- `run_id`, model/blend, `split_id`, config/artifact/commit nếu có.
- Validation metrics và điểm Public LB thực do Kaggle trả về, ở trường riêng.
- Trạng thái accepted/error và thông báo lỗi nếu có.
- Link/hình minh chứng nếu nhóm lưu giữ.

Không ghi điểm, thứ hạng hoặc trạng thái accepted trước khi quan sát kết quả thật. Nếu không thể truy cập Kaggle, ghi trở ngại và phương án xử lý theo Gate B.

## 7. Đường dẫn file cần tạo hoặc chỉnh sửa

Các file sau là đầu ra dự kiến khi triển khai:

| Đường dẫn | Hành động | Nội dung |
|---|---|---|
| `configs/dl.yaml` | Tạo | Architecture, optimizer, loss, batch, epochs, early stopping, seed |
| `src/models/mlp.py` | Tạo | MLP trả logits |
| `src/training/train_dl.py` | Tạo | DataLoader, train/validation loop, checkpoint/log |
| `src/evaluation/metrics.py` | Tạo/dùng chung | Metric nhất quán ML/DL |
| `src/prediction/predict_dl.py` | Tạo | Load checkpoint/preprocessor và predict probabilities |
| `src/prediction/make_submission.py` | Tạo | Chọn prediction, kiểm tra và ghi submission |
| `experiments/dl/` | Ghi artifact nhẹ | Config snapshot, metrics, curves/manifest |
| `experiments/comparison/` | Ghi artifact nhẹ | Bảng so sánh, selection/blend/submission manifest |
| `outputs/models/` | Ghi artifact sinh ra | DL checkpoint, bị Git ignore |
| `outputs/submissions/submission.csv` | Tạo | Submission cuối đúng schema, bị Git ignore |
| `docs/03_MODEL_COMPARISON.md` | Tạo khi có kết quả | So sánh ML/DL thật |
| `docs/04_KAGGLE_SUBMISSION.md` | Tạo sau nộp thật | Minh chứng và điểm/trạng thái thật |
| `tests/test_mlp.py` | Tạo | Forward shape/logits/gradient |
| `tests/test_dl_training.py` | Tạo | Smoke train/validation và checkpoint |
| `tests/test_prediction_contract.py` | Tạo | Schema probability ML/DL |
| `tests/test_submission.py` | Tạo | Columns, rows, ID, finite/range/order |

Không tạo file tài liệu kết quả với số liệu placeholder có vẻ như kết quả thật. Nếu cần template, dùng `TBD`/dấu gạch rõ ràng và chỉ điền sau run/nộp.

## 8. Kiểm thử và phương pháp đánh giá

### 8.1. Kiểm thử DL

- Forward batch size 1 và >1 trả shape quy định, logits finite.
- `BCEWithLogitsLoss` nhận đúng dtype/shape target; model không sigmoid hai lần.
- Một optimizer step làm gradient/parameter cập nhật; validation không tạo gradient.
- `train()`/`eval()` được chuyển đúng, dropout tắt khi inference.
- Seed cố định tạo kết quả smoke test ổn định trong tolerance phù hợp.
- Best checkpoint load được và tái tạo prediction sample.

### 8.2. Kiểm thử prediction contract

- ID unique, không missing; số hàng bằng partition.
- `y_true` và validation IDs khớp giữa ML/DL.
- `split_id` giống nhau; probability fraud finite và `[0,1]`.
- Metric tính lại khớp metadata trong tolerance.
- Merge theo ID không làm mất/nhân đôi dòng.

### 8.3. Kiểm thử submission

- Tên/thứ tự cột đúng sample: `id,is_fraud`.
- Số dòng bằng test (299.984 theo audit hiện tại).
- ID khớp test/sample cả giá trị lẫn thứ tự; không duplicate.
- Probability numeric, finite, không NaN/inf, trong `[0,1]`.
- Đọc lại CSV sau khi ghi vẫn đạt mọi assertion.
- Manifest truy được đến run/config/model tạo submission.

### 8.4. Đánh giá model

- ROC-AUC chính, PR-AUC và threshold metrics từ cùng validation.
- Early stopping/model selection không dùng test/Public LB.
- Báo cáo overfitting, training stability và tài nguyên.
- Kết luận dựa trên run hợp lệ, không che giấu run thất bại hoặc giới hạn.

## 9. Đầu ra và sản phẩm bàn giao

### Deep Learning

- MLP PyTorch chạy end-to-end với training/validation loop.
- Config, log, learning curves và validation probabilities.
- Best checkpoint cùng preprocessor/feature order và hàm load/inference.
- So sánh các cấu hình MLP bằng số đo thực.

### Integration

- Prediction contract đã kiểm tra cho ML/DL.
- `docs/03_MODEL_COMPARISON.md` với bảng công bằng và quyết định model cuối.
- `outputs/submissions/submission.csv` đúng schema.
- Submission manifest và `docs/04_KAGGLE_SUBMISSION.md` sau khi nộp thật.
- Kiểm thử tích hợp/submission và hướng dẫn tái lập.

## 10. Tiêu chí nghiệm thu Definition of Done

- [ ] MLP dùng PyTorch, output logit và `BCEWithLogitsLoss` đúng cách.
- [ ] Preprocessor/scaler/encoder fit train-only và được lưu cùng run.
- [ ] Training/validation loop, early stopping, best checkpoint và log chạy thật.
- [ ] Seed/device/version/config được ghi; checkpoint load lại được.
- [ ] ROC-AUC, PR-AUC và threshold metrics dùng cùng `split_id` với ML.
- [ ] Xử lý imbalance được so sánh có kiểm chứng, không bù hai lần vô căn cứ.
- [ ] Validation prediction ML/DL khớp ID/target và qua contract checks.
- [ ] Model cuối được chọn bằng bằng chứng, không mặc định DL hơn ML.
- [ ] Blending, nếu có, là run P2 tái lập được và cải thiện validation.
- [ ] Submission đúng 2 cột, đúng số dòng/ID/order, probability `[0,1]` và không missing/inf.
- [ ] Kaggle result chỉ được ghi sau khi nộp thật; Public LB tách khỏi validation.
- [ ] Kiểm thử tích hợp đạt và cả nhóm có thể truy ngược submission về run/model.

## 11. Deadline và checklist công việc

| Hạn | Công việc | Bàn giao/Gate |
|---|---|---|
| 19/11 | Chuẩn bị PyTorch skeleton, device/seed và data interface | Code smoke-ready |
| 20/11 | Smoke test MLP trên dữ liệu nhỏ, chưa kết luận chất lượng | Phát hiện lỗi shape/loss |
| 21/11 | Train MLP cơ bản trên split chính thức | Gate A: val probability/checkpoint đầu |
| 22/11 | Thử dropout, early stopping, `pos_weight` nếu phù hợp | Ít nhất 1 DL end-to-end |
| 23-24/11 | Tuning giới hạn, chọn checkpoint, xuất probabilities | Bảng metric/curve DL |
| 25/11 | Thu ML output, tích hợp inference, tạo submission thử | CSV hợp lệ lần đầu |
| 26/11 | Nộp thử Kaggle và ghi nhận kết quả thật | Gate B: accepted hoặc blocker có kế hoạch |
| 27/11 | Viết phương pháp DL và bảng so sánh | Dự thảo báo cáo |
| 28/11 | Chọn model cuối; blending chỉ nếu còn thời gian/có lợi | Gate C: freeze pipeline |
| 29/11 | Nộp submission cuối, lưu minh chứng, khóa model | Hoàn tất trước hạn nội bộ |
| 30/11 | Rà soát DL, integration, CSV và demo | Bàn giao toàn bộ trước hạn |

Nếu MLP chưa ổn vào 24/11, thu nhỏ kiến trúc và dùng early stopping. Nếu chưa có submission vào 25/11, dùng model ML ổn định tốt nhất để tạo/nộp CSV trước.

## 12. Rủi ro và phương án xử lý

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Dense input gây OOM | Convert one-hot lỗi hoặc batch không chạy | Đo memory, điều chỉnh encoding/dtype/batch; không dense toàn bộ mù quáng |
| Double sigmoid/sai loss | Loss bất thường, probability méo | Model trả logits; sigmoid chỉ tại metric/inference; unit test |
| Shape lỗi ở batch size 1 | Tensor thành scalar | Tránh `squeeze()` không chỉ định; test batch size 1 |
| Overfitting MLP | Train loss giảm, val metric xấu | MLP nhỏ, dropout/weight decay, early stopping, dừng tuning |
| Kết quả không tái lập | Run chênh lệch lớn | Log seed/device/version; deterministic settings; thử nhiều seed nếu đủ thời gian |
| Weighting làm xấu calibration/metric | PR/ROC không tăng | So sánh baseline; giữ phương án có bằng chứng, không ép weighting |
| ML/DL không cùng validation | ID/`split_id` mismatch | Từ chối so sánh; tái xuất predictions theo split chính thức |
| Probability/ID lệch thứ tự | Submission schema đúng nhưng mapping sai | Merge/check theo ID, assert order với test/sample, đọc lại CSV |
| Public LB lệch validation | Kaggle score khác kỳ vọng | Kiểm tra drift/split/leakage; không overfit leaderboard |
| Kaggle không nhận file | Error schema/rows/range | Chạy test submission, bám sample, nộp thử 26/11 |
| Integration trễ | Chưa có model tốt nhất | Tạo submission từ pipeline ổn định trước, cải tiến sau |

## 13. Phối hợp và bàn giao giữa các thành viên

### Nhận từ Data và EDA

Người 3 xác nhận `split_id`, validation IDs, feature order, preprocessing contract và seed trước run chính thức. Mọi thay đổi encoding/dtype phải được ghi trong DL config; không thay hàng validation.

### Nhận từ Machine Learning

Người 3 kiểm tra validation/test prediction, config và model load của Người 2. Nếu schema hoặc metric không khớp, trả lại lỗi cụ thể theo ID/run thay vì tự sửa file không có dấu vết.

### Bàn giao ngược để kiểm tra chéo

- Gửi validation predictions/metrics/config/checkpoint manifest cho Người 1 kiểm tra split/leakage.
- Gửi bảng so sánh và quyết định model cuối cho Người 2 xác nhận ML run được load đúng.
- Nhờ Người 1 đối chiếu test/sample ID và Người 2 xác nhận inference ML nếu model cuối/blend có thành phần ML.

### Quyết định và bàn giao cuối

Cả nhóm cùng xác nhận:

- Bảng so sánh chỉ gồm run hợp lệ trên cùng protocol.
- Model/blend cuối và lý do chọn đã được ghi.
- Submission truy được tới `run_id`, config, preprocessor và artifact.
- `docs/04_KAGGLE_SUBMISSION.md` chỉ chứa sự kiện/điểm thực.
- Không còn blocker P0 và toàn bộ sản phẩm được bàn giao trước hoặc trong ngày 30/11/2026.
