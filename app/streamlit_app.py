"""Giao diện Web Streamlit Chuyên nghiệp: Phân loại Giọng nói Parkinson (Leakage-Aware).

Ứng dụng web phong cách Clinical Research Dashboard:
- Tab 1: Sàng lọc & Dự đoán Bệnh nhân (1-Click Sample Demo, Upload CSV, Gộp Median, Cảnh báo dải đo P1-P99).
- Tab 2: Thí nghiệm Đối chứng Rò rỉ Dữ liệu (Naive Split vs Nested Subject-Level CV).
- Tab 3: Báo cáo Đánh giá Mô hình & Biểu đồ Khoa học Canonical.
- Tab 4: Hợp đồng 20 Đặc trưng Âm học & Dải đo Sinh lý học.
"""

from __future__ import annotations

import io
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from app.settings import ARTIFACT_PATH, PROJECT_ROOT, RESEARCH_WARNING
from parkinson_voice.predict import load_bundle, predict_records, sanitize_csv_value

# Đường dẫn tài nguyên bổ trợ
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
NAIVE_AUDIT_PATH = PROJECT_ROOT / "artifacts" / "naive_split_audit.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
SAMPLE_FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "inference_valid.csv"
DATA_PATH = PROJECT_ROOT / "data" / "parkinsons.csv"

# Cấu hình giao diện trang web Streamlit
st.set_page_config(
    page_title="Parkinson Voice Screening & Leakage Audit",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách Y sinh học hiện đại (Clinical Research Dashboard)
st.markdown(
    """
    <style>
    /* Font and General Setup */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2338 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 18px;
        line-height: 1.5;
    }
    
    /* Tag Pills */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .badge-blue {
        background: rgba(14, 165, 233, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-emerald {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #C084FC;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }
    .badge-amber {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }

    /* Metric Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.9rem;
        font-weight: 800;
    }
    .kpi-desc {
        color: #64748B;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    /* Warning Box */
    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 20px;
        color: #FDE68A;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_bundle():
    """Nạp và lưu cache model bundle để tối ưu hiệu năng suy luận."""
    if not ARTIFACT_PATH.is_file():
        return None
    return load_bundle(ARTIFACT_PATH)


@st.cache_data(show_spinner=False)
def load_cached_metrics():
    """Đọc file metrics.json từ artifacts."""
    if METRICS_PATH.is_file():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def load_cached_naive_audit():
    """Đọc file naive_split_audit.json từ artifacts."""
    if NAIVE_AUDIT_PATH.is_file():
        try:
            return json.loads(NAIVE_AUDIT_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


@st.cache_data(show_spinner=False)
def get_sample_inference_data() -> pd.DataFrame:
    """Tạo dữ liệu mẫu đa dạng gồm nhiều subject từ dataset gốc (đã gỡ nhãn status)."""
    if SAMPLE_FIXTURE_PATH.is_file():
        return pd.read_csv(SAMPLE_FIXTURE_PATH)
    if DATA_PATH.is_file():
        df = pd.read_csv(DATA_PATH)
        df_clean = df.drop(columns=["status"], errors="ignore")
        return df_clean.head(18)
    return pd.DataFrame()


bundle = get_cached_bundle()
metrics_data = load_cached_metrics()
naive_audit_data = load_cached_naive_audit()

# --- SIDEBAR THÔNG TIN VÀ CẤU HÌNH ---
with st.sidebar:
    st.markdown("### 🎙️ Parkinson Voice AI")
    st.caption("Hệ thống Sàng lọc Đặc trưng Âm học & Kiểm toán Rò rỉ Dữ liệu")
    st.divider()

    st.markdown("#### ⚙️ Thông số Mô hình Triển khai")
    threshold_val = float(bundle["decision_threshold"]) if bundle else 0.4206
    aggregation_rule = bundle["aggregation"] if bundle else "median"

    st.markdown("- **Mô hình**: `L2 Logistic Regression`")
    st.markdown("- **Tối ưu siêu tham số**: `C=0.01, balanced`")
    st.markdown(f"- **Quy tắc gộp**: `{aggregation_rule.upper()}`")
    st.markdown(f"- **Ngưỡng quyết định OOF**: `{threshold_val:.4f}`")
    st.markdown("- **Đặc trưng mô hình**: `20 đặc trưng số`")

    st.divider()
    st.markdown("#### 🛡️ Nguyên tắc Thiết kế")
    st.info(
        "**Không Rò rỉ Nhóm (Zero Group Leakage):**\n"
        "Mọi bước chuẩn hóa, tune tham số và đánh giá đều cô lập triệt để ở cấp độ bệnh nhân (`subject`)."
    )

    st.divider()
    st.caption("Dự án nghiên cứu nguồn mở • Giấy phép MIT")


# --- HERO HEADER BANNER ---
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-title">
            <span>🎙️</span> Sàng Lọc Đặc Trưng Giọng Nói Parkinson
        </div>
        <div class="hero-subtitle">
            Hệ thống phân tích âm học ở cấp độ Bệnh nhân (Subject-Level) với Giao thức Đánh giá Chống Rò rỉ Dữ liệu (Leakage-Aware Machine Learning).
        </div>
        <div class="badge-container">
            <span class="badge badge-emerald">🛡️ Nested Subject CV (4 Outer × 3 Inner)</span>
            <span class="badge badge-blue">🤖 Logistic Regression (L2, C=0.01)</span>
            <span class="badge badge-purple">📐 Gộp Median Cấp Đối Tượng</span>
            <span class="badge badge-amber">⚡ Ngưỡng Nội bộ: {threshold_val:.4f}</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Medical Disclaimer Banner
st.markdown(
    f"""
    <div class="disclaimer-banner">
        <strong>⚠️ Tuyên bố Giới hạn & An toàn Y tế (Medical Disclaimer):</strong> {RESEARCH_WARNING}
        Hệ thống nhận bảng đặc trưng âm học đo đạc sẵn từ âm nguyên âm kéo dài <code>/a/</code>,
        <strong>không nhận trực tiếp tệp âm thanh thô (WAV/MP3)</strong>.
        Kết quả sàng lọc mang tính chất thực nghiệm khoa học, không có giá trị thay thế chẩn đoán y khoa.
    </div>
    """,
    unsafe_allow_html=True,
)

if bundle is None:
    st.error(
        f"❌ Không tìm thấy model artifact tại: `{ARTIFACT_PATH}`. "
        "Vui lòng chạy `python -m parkinson_voice.train` để sinh mô hình trước."
    )
    st.stop()


# --- TABS GIAO DIỆN CHÍNH ---
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔬 Sàng Lọc Bệnh Nhân (Screening)",
        "🛡️ Đối Chứng Rò Rỉ Dữ Liệu (Leakage Audit)",
        "📊 Báo Cáo Đánh Giá Mô Hình (Evaluation)",
        "📋 Hợp Đồng Đặc Trưng & Dải Đo (Contract & Ranges)",
    ]
)

# ==============================================================================
# TAB 1: SÀNG LỌC & DỰ ĐOÁN BỆNH NHÂN
# ==============================================================================
with tab1:
    st.markdown("### 📥 Nạp Dữ Liệu Đặc Trưng Âm Học")

    input_source = st.radio(
        "Chọn nguồn dữ liệu đầu vào:",
        ["🎯 Dùng dữ liệu mẫu thử nhanh (1-Click Demo Sample)", "📁 Tải lên tệp CSV của bạn"],
        horizontal=True,
    )

    df_to_process: pd.DataFrame | None = None

    if input_source == "🎯 Dùng dữ liệu mẫu thử nhanh (1-Click Demo Sample)":
        sample_df = get_sample_inference_data()
        if not sample_df.empty:
            st.success(
                f"💡 Đã nạp thành công bộ dữ liệu mẫu gồm **{len(sample_df)} bản ghi âm** "
                f"của **{sample_df['name'].apply(lambda x: x.rsplit('_', 1)[0]).nunique()} đối tượng**."
            )
            with st.expander("👁️ Xem trước bảng dữ liệu mẫu (5 dòng đầu)"):
                st.dataframe(sample_df.head(5), use_container_width=True)
            df_to_process = sample_df
        else:
            st.warning("Không tìm thấy tệp dữ liệu mẫu. Vui lòng tải lên tệp CSV.")

    else:
        uploaded_file = st.file_uploader(
            "Tải lên tệp CSV chứa cột 'name' và 20 hoặc 22 đặc trưng âm thanh",
            type=["csv"],
            help="Tệp CSV cần có cột name (vd: phon_R01_S01_1) và các cột đặc trưng âm học chuẩn UCI.",
        )
        if uploaded_file is not None:
            try:
                df_to_process = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"❌ Không thể đọc tệp CSV: {e}")

    # Xử lý và suy luận khi có dữ liệu
    if df_to_process is not None and not df_to_process.empty:
        try:
            with st.spinner("Đang trích xuất đặc trưng và tính toán điểm sàng lọc..."):
                record_results, subject_results = predict_records(df_to_process, bundle)
        except Exception as exc:
            st.error(f"❌ Lỗi suy luận: {exc}")
            st.stop()

        n_subjects = len(subject_results)
        n_records = len(record_results)
        n_positive = int((subject_results["screening_result"] == "above_internal_threshold").sum())
        positive_rate = (n_positive / n_subjects * 100) if n_subjects > 0 else 0
        total_warnings = sum(len(w) for w in subject_results["warnings"])

        st.divider()

        # Hiển thị KPI Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Bệnh nhân (Subjects)</div>
                    <div class="kpi-value">{n_subjects}</div>
                    <div class="kpi-desc">Tổng số đối tượng phân tích</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Bản ghi âm (Recordings)</div>
                    <div class="kpi-value">{n_records}</div>
                    <div class="kpi-desc">Trung bình {(n_records / n_subjects):.1f} bản ghi/người</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Vượt Ngưỡng Nội Bộ</div>
                    <div class="kpi-value">{n_positive} <span style="font-size:1.1rem; color:#EF4444;">({positive_rate:.0f}%)</span></div>
                    <div class="kpi-desc">Screening Score ≥ {threshold_val:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Cảnh Báo Kỹ Thuật</div>
                    <div class="kpi-value">{total_warnings}</div>
                    <div class="kpi-desc">Ngoại dải P1-P99 hoặc &lt;3 bản ghi</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # BẢNG KẾT QUẢ SUBJECT-LEVEL
        st.markdown("### 📊 Kết Quả Sàng Lọc theo Từng Bệnh Nhân (Subject-Level)")
        st.caption(
            "Điểm số của từng bệnh nhân được tính bằng **Median** các bản ghi âm lặp lại, "
            f"so sánh với ngưỡng nghiên cứu nội bộ `{threshold_val:.4f}`."
        )

        display_df = subject_results.copy()
        display_df["Tình trạng Sàng lọc"] = display_df["screening_result"].apply(
            lambda x: (
                "🔴 Vượt ngưỡng nội bộ (Lưu ý)"
                if x == "above_internal_threshold"
                else "🟢 Dưới ngưỡng nội bộ (Thấp)"
            )
        )
        display_df["Điểm Sàng lọc"] = display_df["subject_screening_score"].apply(
            lambda s: f"{s:.1%}"
        )
        display_df["Số bản ghi"] = display_df["n_recordings"]
        display_df["Cảnh báo"] = display_df["warnings"].apply(
            lambda w: ", ".join(w) if w else "Không có"
        )

        st.dataframe(
            display_df[
                [
                    "subject_id",
                    "Số bản ghi",
                    "Điểm Sàng lọc",
                    "Tình trạng Sàng lọc",
                    "Cảnh báo",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # BIỂU ĐỒ TRỰC QUAN ĐIỂM SÀNG LỌC
        st.markdown("#### 📈 Phân Bố Điểm Sàng Lọc So Với Ngưỡng Quyết Định")

        fig, ax = plt.subplots(figsize=(10, max(3.5, len(subject_results) * 0.45)))
        plot_df = subject_results.sort_values("subject_screening_score", ascending=True)
        bar_colors = [
            "#EF4444" if score >= threshold_val else "#10B981"
            for score in plot_df["subject_screening_score"]
        ]

        y_positions = range(len(plot_df))
        ax.barh(
            y_positions,
            plot_df["subject_screening_score"],
            color=bar_colors,
            alpha=0.85,
            edgecolor="#334155",
            height=0.6,
        )
        ax.axvline(
            threshold_val,
            color="#F59E0B",
            linestyle="--",
            linewidth=2,
            label=f"Decision Threshold ({threshold_val:.4f})",
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_df["subject_id"], fontsize=9)
        ax.set_xlabel("Điểm Sàng lọc Median (Subject Screening Score)", fontsize=10)
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # CHI TIẾT RECORDING-LEVEL
        with st.expander("🔍 Xem Chi Tiết Từng Bản Ghi Âm Đơn Lẻ (Recording-Level Breakdown)"):
            st.markdown(
                "Mỗi bản ghi âm được mô hình dự đoán một điểm số độc lập trước khi gộp Median:"
            )
            st.dataframe(
                record_results[
                    [
                        "recording_id",
                        "subject_id",
                        "screening_score",
                        "feature_warnings",
                    ]
                ].assign(
                    screening_score=record_results["screening_score"].apply(lambda s: f"{s:.2%}"),
                    warnings_summary=record_results["feature_warnings"].apply(
                        lambda w: f"{len(w)} cảnh báo ngoại dải" if w else "Trong dải chuẩn"
                    ),
                ),
                use_container_width=True,
            )

        # TRÌNH PHÂN TÍCH ĐẶC TRƯNG ÂM HỌC
        with st.expander("🎼 Phân Tích & Đối Chiếu Đặc Trưng Âm Học (Feature Inspector)"):
            selected_feature = st.selectbox(
                "Chọn đặc trưng âm học cần khảo sát giá trị:",
                bundle["feature_columns"],
                index=0,
            )
            training_ranges = bundle.get("feature_ranges") or bundle.get(
                "training_feature_ranges", {}
            )
            if selected_feature in training_ranges:
                p1, p99 = training_ranges[selected_feature]
                st.caption(
                    f"Dải giá trị huấn luyện (P1 - P99) của `{selected_feature}`: "
                    f"**[{p1:.4f}  đến  {p99:.4f}]**"
                )

            chart_df = pd.DataFrame(
                {
                    "subject_id": record_results["subject_id"].astype(str),
                    "feature_value": df_to_process[selected_feature].astype(float),
                }
            )

            fig_feat, ax_feat = plt.subplots(figsize=(10, 4))
            unique_subjects = chart_df["subject_id"].unique()
            x_indices = range(len(unique_subjects))

            for idx, s_id in enumerate(unique_subjects):
                vals = chart_df[chart_df["subject_id"] == s_id]["feature_value"].to_numpy()
                ax_feat.scatter(
                    [idx] * len(vals),
                    vals,
                    alpha=0.75,
                    s=65,
                    color="#38BDF8",
                    edgecolors="#0284C7",
                    label="Bản ghi âm" if idx == 0 else "",
                )
                med_val = float(pd.Series(vals).median())
                ax_feat.plot(
                    [idx - 0.25, idx + 0.25],
                    [med_val, med_val],
                    color="#F59E0B",
                    linewidth=2.5,
                    label="Median Subject" if idx == 0 else "",
                )

            if selected_feature in training_ranges:
                p1, p99 = training_ranges[selected_feature]
                ax_feat.axhline(
                    p1,
                    color="#10B981",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"P1 Tham chiếu ({p1:.4f})",
                )
                ax_feat.axhline(
                    p99,
                    color="#EF4444",
                    linestyle="--",
                    alpha=0.8,
                    linewidth=1.5,
                    label=f"P99 Tham chiếu ({p99:.4f})",
                )

            ax_feat.set_xticks(list(x_indices))
            ax_feat.set_xticklabels(unique_subjects, rotation=35, ha="right", fontsize=9)
            ax_feat.set_ylabel(selected_feature, fontsize=10)
            ax_feat.set_xlabel("Mã bệnh nhân (Subject ID)", fontsize=10)
            ax_feat.grid(axis="y", linestyle=":", alpha=0.4)
            ax_feat.legend(loc="upper right", fontsize=8)
            fig_feat.tight_layout()
            st.pyplot(fig_feat)
            plt.close(fig_feat)

        # XUẤT FILE CSV KẾT QUẢ
        st.divider()
        csv_buffer = io.StringIO()
        clean_export = subject_results.copy()
        clean_export["subject_id"] = clean_export["subject_id"].map(sanitize_csv_value)
        clean_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Tải xuống Báo cáo Kết quả Sàng lọc dạng CSV",
            data=csv_buffer.getvalue().encode("utf-8-sig"),
            file_name="parkinsons_screening_results.csv",
            mime="text/csv",
            help="Tệp CSV đã được áp dụng cơ chế bảo vệ sanitize chống CSV formula injection.",
        )


# ==============================================================================
# TAB 2: ĐỐI CHỨNG RÒ RỈ DỮ LIỆU (LEAKAGE AUDIT)
# ==============================================================================
with tab2:
    st.markdown("### 🛡️ Thí Nghiệm Đối Chứng: Tại Sao Rò Rỉ Dữ Liệu Lại Nguy Hiểm?")
    st.markdown(
        """
        Nhiều nghiên cứu y sinh áp dụng chia dữ liệu ngẫu nhiên theo dòng (**Record-Level Split**).
        Đối với dữ liệu có nhiều bản ghi âm lặp lại của cùng một bệnh nhân, việc này gây ra hiện tượng
        **Rò rỉ Danh tính Người nói (Vocal Tract Identity Leakage)**:
        """
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.error("❌ Giao thức Sai Lầm: Random Record Split (Naive)")
        st.markdown(
            """
            - **Cách chia**: Chia ngẫu nhiên 80% train / 20% test theo từng dòng bản ghi.
            - **Hiện tượng**: Bản ghi 1 của Bệnh nhân A ở tập Train, bản ghi 2 của Bệnh nhân A ở tập Test.
            - **Hậu quả**: Mô hình ghi nhớ cấu trúc thanh quản và thiết bị ghi âm của bệnh nhân A, thay vì học bệnh lý Parkinson.
            - **Accuracy kiểm thử**: **~92.3%** (Lạc quan giả tạo).
            - **Bệnh nhân trùng lặp trong tập Test**: **24 / 24 bệnh nhân (100%)**!
            """
        )

    with col_b:
        st.success("✅ Giao thức Chuẩn mực: Nested Subject-Level CV")
        st.markdown(
            """
            - **Cách chia**: Chia toàn bộ bệnh nhân thành các Fold độc lập (4 Outer × 3 Inner).
            - **Hiện tượng**: Bệnh nhân ở tập kiểm thử hoàn toàn xa lạ với tập huấn luyện.
            - **Hậu quả**: Phản ánh chính xác năng lực phát hiện bất thường trên người mới đến khám.
            - **Balanced Accuracy thực tế**: **~62.5%** (Chỉ số trung thực).
            - **Bệnh nhân trùng lặp trong tập Test**: **0 / 32 bệnh nhân (0%)**!
            """
        )

    st.divider()

    st.markdown("#### 📊 So Sánh Trực Quan Thí Nghiệm Đối Chứng")

    comparison_data = pd.DataFrame(
        {
            "Chỉ số Đánh giá": [
                "Độ chính xác (Accuracy / Balanced Acc)",
                "Tỷ lệ Bệnh nhân Test bị Rò rỉ vào Train",
                "Độ tin cậy trong Ứng dụng Thực tế",
            ],
            "Naive Record Split (Sai lệch)": [
                "~92.3% (Lạc quan ảo)",
                "100% (24/24 subjects)",
                "Rất thấp (Fail khi gặp người mới)",
            ],
            "Nested Subject CV (Chuẩn khoa học)": [
                "~62.5% (Ước lượng trung thực)",
                "0% (Tuyệt đối không rò rỉ)",
                "Cao (Đúng năng lực tổng quát hóa)",
            ],
        }
    )
    st.table(comparison_data)

    if (FIGURES_DIR / "cross_fitted_subject_scores.png").is_file():
        st.markdown("#### 🔬 Phân bố Điểm số Out-of-Fold trên Bệnh nhân Thực tế")
        st.image(
            str(FIGURES_DIR / "cross_fitted_subject_scores.png"),
            caption="Hình 1: Điểm số Cross-fitted của từng bệnh nhân kiểm thử ngoài fold. Xanh: Nhóm khỏe mạnh (Control), Đỏ: Bệnh nhân Parkinson.",
            use_container_width=True,
        )


# ==============================================================================
# TAB 3: BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (MODEL EVALUATION)
# ==============================================================================
with tab3:
    st.markdown("### 📊 Đánh Giá Toàn Diện Mô Hình L2 Logistic Regression")

    if metrics_data:
        nested_metrics = metrics_data.get("nested_cv_subject", {})
        deployment_metrics = metrics_data.get("deployment_oof", {})

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric(
                "Balanced Accuracy",
                f"{nested_metrics.get('Balanced Accuracy', 0.625):.3f}",
                help="Chỉ số chính chống mất cân bằng lớp 3:1",
            )
        with m_col2:
            st.metric(
                "ROC-AUC",
                f"{nested_metrics.get('ROC-AUC', 0.740):.3f}",
                help="Khả năng phân biệt xếp hạng xác suất",
            )
        with m_col3:
            st.metric(
                "Sensitivity (Recall)",
                f"{nested_metrics.get('Recall/Sensitivity', 0.750):.3f}",
                help="Tỷ lệ phát hiện đúng người mắc Parkinson",
            )
        with m_col4:
            st.metric(
                "Specificity",
                f"{nested_metrics.get('Specificity', 0.500):.3f}",
                help="Tỷ lệ nhận diện đúng người khỏe mạnh",
            )

        st.markdown(
            f"""
            - **Macro-F1**: `{nested_metrics.get("F1-macro", 0.614):.3f}`
            - **Brier Score**: `{nested_metrics.get("Brier score", 0.215):.3f}` (Đo độ chuẩn xác xác suất)
            - **Khoảng tin cậy 95% Bootstrap Bệnh nhân**:
              - Balanced Accuracy 95% CI: `[0.44  –  0.81]`
              - ROC-AUC 95% CI: `[0.55  –  0.90]`
            """
        )
    else:
        st.info("Chưa tìm thấy tệp metrics.json từ artifacts.")

    st.divider()

    st.markdown("#### 🖼️ Bộ Biểu Đồ Nghiên Cứu Khoa Học (Canonical Portfolio Figures)")

    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        if (FIGURES_DIR / "inner_logistic_selection.png").is_file():
            st.image(
                str(FIGURES_DIR / "inner_logistic_selection.png"),
                caption="Hình 2: Lựa chọn siêu tham số C và class_weight trong vòng lặp Nested Inner OOF.",
                use_container_width=True,
            )
        if (FIGURES_DIR / "fixed_feature_contract.png").is_file():
            st.image(
                str(FIGURES_DIR / "fixed_feature_contract.png"),
                caption="Hình 3: Tính ổn định tuyệt đối của hợp đồng 20 đặc trưng số sau khi loại bỏ dư thừa đại số.",
                use_container_width=True,
            )

    with fig_col2:
        if (FIGURES_DIR / "threshold_search.png").is_file():
            st.image(
                str(FIGURES_DIR / "threshold_search.png"),
                caption="Hình 4: Tối ưu hóa ngưỡng quyết định nội bộ dựa trên Balanced Accuracy trên Inner OOF.",
                use_container_width=True,
            )
        if (FIGURES_DIR / "cross_fitted_subject_scores.png").is_file():
            st.image(
                str(FIGURES_DIR / "cross_fitted_subject_scores.png"),
                caption="Hình 5: Phân bố điểm số cross-fitted out-of-fold cho toàn bộ 32 đối tượng nghiên cứu.",
                use_container_width=True,
            )


# ==============================================================================
# TAB 4: HỢP ĐỒNG ĐẶC TRƯNG & DẢI ĐO (FEATURE CONTRACT & RANGES)
# ==============================================================================
with tab4:
    st.markdown("### 📋 Hợp Đồng 20 Đặc Trưng Âm Học & Loại Bỏ Dư Thừa Đại Số")

    st.markdown(
        """
        Bộ dữ liệu gốc từ UCI có **22 đặc trưng âm học**, tuy nhiên dự án loại bỏ có chủ đích 
        **2 đặc trưng dư thừa đại số hoàn hảo** trước khi đưa vào mô hình:
        - $\\text{Jitter:DDP} = 3 \\times \\text{MDVP:RAP}$ *(Bội số xác định 3 lần)*
        - $\\text{Shimmer:DDA} = 3 \\times \\text{Shimmer:APQ3}$ *(Bội số xác định 3 lần)*
        
        Việc loại bỏ này giúp triệt tiêu hoàn toàn hiện tượng đa cộng tuyến hoàn hảo, 
        giúp ma trận nghịch đảo trong hồi quy tuyến tính ổn định về mặt số học.
        """
    )

    st.markdown("#### 📑 Danh Mục 20 Đặc Trưng Mô Hình Theo Nhóm Sinh Lý Học")

    feature_categories = {
        "Tần số Cơ bản (Fundamental Frequency)": [
            "MDVP:Fo(Hz) - Tần số cơ bản trung bình",
            "MDVP:Fhi(Hz) - Tần số cơ bản cực đại",
            "MDVP:Flo(Hz) - Tần số cơ bản cực tiểu",
        ],
        "Biến thiên Tần số (Jitter - Đo độ rung tần số)": [
            "MDVP:Jitter(%) - Jitter tương đối theo %",
            "MDVP:Jitter(Abs) - Jitter tuyệt đối theo micro-giây",
            "MDVP:RAP - Relative Amplitude Perturbation",
            "MDVP:PPQ - Five-point Period Perturbation Quotient",
        ],
        "Biến thiên Biên độ (Shimmer - Đo độ rung biên độ)": [
            "MDVP:Shimmer - Shimmer cục bộ",
            "MDVP:Shimmer(dB) - Shimmer tính theo dB",
            "Shimmer:APQ3 - Three-point Amplitude Perturbation Quotient",
            "Shimmer:APQ5 - Five-point Amplitude Perturbation Quotient",
            "MDVP:APQ - 11-point Amplitude Perturbation Quotient",
        ],
        "Tỷ số Nhiễu và Hài âm (Harmonics & Noise)": [
            "NHR - Noise-to-Harmonics Ratio",
            "HNR - Harmonics-to-Noise Ratio",
        ],
        "Đặc trưng Động lực học Phi tuyến (Nonlinear Measures)": [
            "RPDE - Recurrence Period Density Entropy (Đo tính hỗn loạn của dao động thanh quản)",
            "DFA - Detrended Fluctuation Analysis (Độ tự tương đồng fractal)",
            "spread1 - Phân bố tần số phi tuyến 1",
            "spread2 - Phân bố tần số phi tuyến 2",
            "D2 - Correlation Dimension",
            "PPE - Pitch Period Entropy (Đo biến thiên chu kỳ cao độ)",
        ],
    }

    for cat_name, feats in feature_categories.items():
        with st.expander(f"🔹 {cat_name} ({len(feats)} đặc trưng)"):
            for f in feats:
                st.markdown(f"- `{f}`")

    # Dải giá trị P1 - P99
    st.markdown("#### 📏 Bảng Dải Đo Huấn Luyện Tham Chiếu (P1 - P99)")
    st.caption(
        "Nếu một bản ghi âm có giá trị nằm ngoài dải này, hệ thống sẽ phát cảnh báo ngoại dải đo."
    )

    tr_ranges = bundle.get("feature_ranges") or bundle.get("training_feature_ranges", {})
    if tr_ranges:
        range_rows = [
            {
                "Đặc trưng": feat,
                "Ngưỡng dưới (P1)": f"{low:.5f}",
                "Ngưỡng trên (P99)": f"{high:.5f}",
            }
            for feat, (low, high) in tr_ranges.items()
        ]
        st.dataframe(pd.DataFrame(range_rows), use_container_width=True, height=350)
