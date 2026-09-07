# Parkinson Voice Feature Screening v1.0.0

## Phạm vi

Research prototype, không dùng để chẩn đoán. Mô hình nhận 20 acoustic features
từ measurement protocol tương thích với UCI; không nhận WAV/MP3 và không tự trích
xuất đặc trưng âm thanh.

## Pipeline và quyết định

- `StandardScaler` → L2 `LogisticRegression` (`C=0.01`, `class_weight=balanced`).
- Recording chỉ có `screening_score`; subject score là median của các recording.
- Threshold OOF của deployment là `0.420608566861`.
- Ít recording hơn ngưỡng training sẽ tạo cảnh báo `INSUFFICIENT_RECORDINGS`.

## Đánh giá

- Nested stratified subject CV: 4 outer × 3 inner trên 32 subject; mỗi subject làm
  outer-test đúng một lần.
- Primary metric: Balanced Accuracy = 0.6250.
- Macro-F1 = 0.6135; ROC-AUC = 0.7396.
- Bootstrap 95% CI dùng 5.000 mẫu ở cấp subject; đây là ước lượng nghiên cứu nội bộ.
- Chưa có external patient cohort hoặc clinical validation.
