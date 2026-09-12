"""Giao diện Web ứng dụng Streamlit cho Phân loại Giọng nói Parkinson.

Cho phép nghiên cứu viên tải lên CSV chứa 20 hoặc 22 đặc trưng âm thanh,
hiển thị score recording và decision ở cấp độ subject, biểu đồ score
và cho phép xuất tệp kết quả dự đoán dạng CSV.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.settings import ARTIFACT_PATH, RESEARCH_WARNING
from parkinson_voice.data import ORIGINAL_FEATURES
from parkinson_voice.predict import load_bundle, predict_records

# Cấu hình giao diện trang web Streamlit
st.set_page_config(
    page_title="Parkinson Voice Feature Screening Demo",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Sàng lọc Đặc trưng Giọng nói Parkinson (Research Prototype)")
st.warning(
    f"⚠️ {RESEARCH_WARNING} Mô hình là nguyên mẫu nghiên cứu, không phải thiết bị y tế hay chẩn đoán lâm sàng."
)

st.markdown(
    "Tải lên tệp CSV chứa cột `name` và 20 hoặc 22 đặc trưng tần số/biên độ giọng nói từ UCI. "
    "**Lưu ý:** Hệ thống nhận bảng đặc trưng âm học số, **không nhận audio thô (WAV/MP3)**. "
    "Mô hình chỉ tạo **screening score từng recording**, sau đó **gộp median theo subject** "
    "và hiển thị cảnh báo dải đo huấn luyện (nếu có)."
)


with st.expander("📋 Xem Schema tệp CSV bắt buộc"):
    st.code("name, " + ", ".join(ORIGINAL_FEATURES), language=None)

uploaded_file = st.file_uploader("Tải lên tệp CSV dữ liệu giọng nói", type=["csv"])

if uploaded_file is not None:
    try:
        input_frame = pd.read_csv(uploaded_file)
        bundle = load_bundle(ARTIFACT_PATH)
        record_results, subject_results = predict_records(input_frame, bundle)
    except Exception as exc:
        st.error(f"❌ Không thể xử lý tệp: {exc}")
        st.stop()

    st.success(
        f"✅ Đã xử lý thành công {len(record_results)} bản ghi âm của "
        f"{len(subject_results)} subject bằng mô hình **Logistic Regression (L2)**."
    )
    st.caption(
        " Quy tắc gộp từ OOF Train: "
        f"Phương pháp gộp `{bundle['aggregation']}`, "
        f"Ngưỡng nội bộ `{float(bundle['decision_threshold']):.2f}`."
    )

    st.subheader("📊 Kết quả Sàng lọc theo Bệnh nhân (Subject-Level)")
    st.dataframe(
        subject_results.style.format(
            {
                "subject_screening_score": "{:.1%}",
            }
        ),
        use_container_width=True,
    )

    chart_frame = subject_results.set_index("subject_id")[["subject_screening_score"]]
    st.bar_chart(chart_frame, y_label="Điểm sàng lọc (screening_score)", horizontal=False)

    st.subheader("📝 Kết quả chi tiết từng Bản ghi âm (Record-Level)")
    st.dataframe(record_results, use_container_width=True)

    st.subheader("📈 Trực quan hóa Đặc trưng Giọng nói")
    feature = st.selectbox("Chọn đặc trưng phân tích", bundle["feature_columns"])
    feature_chart = input_frame.assign(subject_id=record_results["subject_id"])
    st.bar_chart(feature_chart, x="subject_id", y=feature, y_label=feature)

    # Nút tải xuống kết quả CSV
    output = io.StringIO()
    subject_results.to_csv(output, index=False)
    st.download_button(
        "📥 Tải tệp kết quả dự đoán CSV",
        data=output.getvalue().encode("utf-8-sig"),
        file_name="parkinsons_subject_predictions.csv",
        mime="text/csv",
    )
