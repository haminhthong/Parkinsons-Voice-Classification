# Dữ liệu

`parkinsons.csv` là bộ Parkinsons của UCI: 195 bản ghi giọng nói từ 32 bệnh nhân, gồm `name`, 22 đặc trưng số và nhãn `status`.

- `status = 0`: không mắc Parkinson trong dữ liệu nguồn.
- `status = 1`: mắc Parkinson trong dữ liệu nguồn.
- Một bệnh nhân có nhiều bản ghi; `subject_id` được suy ra bằng cách bỏ hậu tố lần ghi khỏi `name`.

Nguồn: [UCI Parkinsons](https://archive.ics.uci.edu/dataset/174/parkinsons), DOI `10.24432/C59C74`.

Dữ liệu nhỏ, lệch lớp và không đại diện cho thực hành lâm sàng. Chỉ sử dụng cho nghiên cứu/học tập, không dùng để chẩn đoán.

## Data card rút gọn

- Kích thước: 195 dòng, 24 cột nguồn.
- Đơn vị độc lập: 32 bệnh nhân; 24 lớp 1 và 8 lớp 0.
- Một dòng: một phép đo giọng nói, không phải một bệnh nhân độc lập.
- Mỗi subject có 6–7 recording; số recording không được dùng thay cho cỡ mẫu độc lập.
- Schema: `name`, 22 đặc trưng số và `status`.
- Missing value: pipeline dừng nếu có ô thiếu, dữ liệu không hữu hạn hoặc cột sai kiểu.
- Provenance: file được lưu nguyên trạng để tái lập; `subject_id` chỉ được dẫn xuất lúc chạy.
- Known gaps: không có demographic metadata, site, thiết bị, thời điểm thu âm hay external cohort.
- License: kiểm tra điều khoản nguồn UCI trước khi tái phân phối ngoài mục đích portfolio/học tập.
