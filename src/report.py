"""Trực quan hóa kết quả canonical và sinh biểu đồ báo cáo.

Tự động đọc các tệp dữ liệu báo cáo từ thư mục `artifacts/` và xuất ra 4 biểu đồ hình ảnh
dùng cho tài liệu minh họa (README.md / Portfolio / Presentation):
Các tên file cũ vẫn được giữ để không hỏng liên kết README, nhưng nội dung
được lấy từ nested subject CV, fixed feature contract và median aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Lưu đồ họa Matplotlib với cấu hình chuẩn dpi=180 và giải phóng bộ nhớ.

    Args:
        figure: Đối tượng plt.Figure.
        path: Đường dẫn tệp ảnh đầu ra.
    """
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def create_portfolio_figures(
    artifact_dir: str | Path = "artifacts",
    output_dir: str | Path = "reports/figures",
) -> list[Path]:
    """Sinh toàn bộ 4 biểu đồ chất lượng cao trực tiếp từ các artifact đã được huấn luyện.

    Args:
        artifact_dir: Thư mục chứa các tệp artifact CSV và JSON (mặc định 'artifacts').
        output_dir: Thư mục lưu xuất các biểu đồ ảnh PNG (mặc định 'reports/figures').

    Returns:
        list[Path]: Danh sách các đường dẫn tệp ảnh đã sinh thành công.
    """
    artifact_dir = Path(artifact_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ưu tiên artifact canonical; fallback chỉ để tái tạo report lịch sử.
    evaluation_dir = artifact_dir / "evaluation"
    benchmark_path = evaluation_dir / "inner_selection.csv"
    scores_path = evaluation_dir / "cross_fitted_subject_predictions.csv"
    threshold_path = evaluation_dir / "threshold_search.csv"
    benchmark = pd.read_csv(
        benchmark_path if benchmark_path.exists() else artifact_dir / "model_benchmark.csv"
    )
    scores = pd.read_csv(
        scores_path if scores_path.exists() else artifact_dir / "holdout_subject_predictions.csv"
    )
    threshold_search = pd.read_csv(
        threshold_path if threshold_path.exists() else artifact_dir / "oof_threshold_search.csv"
    )
    stability_path = artifact_dir / "feature_selection_stability.csv"
    stability = (
        pd.read_csv(stability_path)
        if not scores_path.exists() and stability_path.exists()
        else pd.DataFrame(
            {"Feature": ["20 model features (fixed contract)"], "Selection frequency": [1.0]}
        )
    )
    if "Aggregation" not in threshold_search.columns:
        threshold_search["Aggregation"] = "median"
    raw_metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    release_metadata_path = artifact_dir / "releases" / "v1.0.0" / "metadata.json"
    release_metadata = (
        json.loads(release_metadata_path.read_text(encoding="utf-8"))
        if release_metadata_path.exists()
        else {}
    )
    decision_threshold = float(
        release_metadata.get(
            "decision_threshold",
            raw_metrics.get("deployment_oof", {}).get("decision_threshold", 0.5),
        )
    )

    paths: list[Path] = []

    # Biểu đồ 1: So sánh cấu hình Logistic Regression trong inner OOF
    figure, axis = plt.subplots(figsize=(9, 5))
    if "Model" in benchmark.columns:
        benchmark_plot = benchmark.sort_values("Subject F1-macro mean")
        axis.errorbar(
            benchmark_plot["Subject F1-macro mean"],
            benchmark_plot["Model"],
            xerr=benchmark_plot["Subject F1-macro std"],
            fmt="o",
            color="#1f5f8b",
            ecolor="#9ecae1",
            capsize=4,
        )
        axis.set(xlim=(0, 1), xlabel="F1-macro CV trung bình ± 1 độ lệch chuẩn")
    else:
        selection_plot = benchmark.sort_values("Balanced Accuracy")
        labels = selection_plot.apply(
            lambda row: f"C={row['C']}, weight={row['class_weight']}", axis=1
        )
        axis.barh(labels, selection_plot["Balanced Accuracy"], color="#1f5f8b")
        axis.set(xlim=(0, 1), xlabel="Inner OOF Balanced Accuracy")
    axis.set_title("Inner selection của Logistic Regression theo subject")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output_dir / "model_benchmark.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 2: Cross-fitted score theo subject
    score_column = "subject_score" if "subject_score" in scores.columns else "probability"
    scores_plot = scores.sort_values(score_column)
    colors = scores_plot["status"].map({0: "#2ca25f", 1: "#de2d26"})
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.scatter(
        scores_plot["subject_id"],
        scores_plot[score_column],
        c=colors,
        s=85,
    )
    axis.axhline(
        decision_threshold,
        color="#222222",
        linestyle="--",
        label=f"Ngưỡng sàng lọc OOF = {decision_threshold:.3f}",
    )
    axis.set(
        ylim=(0, 1),
        xlabel="Subject cross-fitted",
        ylabel="Screening score",
    )
    axis.set_title("Cross-fitted screening score theo subject")
    axis.tick_params(axis="x", rotation=35)
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2ca25f",
            label="Nhãn thật = 0",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#de2d26",
            label="Nhãn thật = 1",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle="--",
            label=f"Ngưỡng sàng lọc OOF = {decision_threshold:.3f}",
        ),
    ]
    axis.legend(handles=legend_items)
    axis.grid(axis="y", alpha=0.25)
    paths.append(output_dir / "holdout_probabilities.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 3: Contract 20 đặc trưng cố định, không chạy feature selection
    stability_plot = stability.head(15).sort_values("Selection frequency")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        stability_plot["Feature"],
        stability_plot["Selection frequency"],
        color="#4c78a8",
    )
    axis.set(xlim=(0, 1), xlabel="Tỷ lệ contract")
    axis.set_title("Feature contract cố định (không phải feature selection)")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output_dir / "feature_selection_stability.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 4: So sánh quy tắc gộp xác suất và quét ngưỡng quyết định OOF
    figure, axis = plt.subplots(figsize=(9, 5))
    for aggregation, group in threshold_search.groupby("Aggregation"):
        ordered = group.sort_values("Threshold")
        axis.plot(
            ordered["Threshold"],
            ordered["Balanced Accuracy"],
            label=aggregation,
        )
    axis.axvline(
        decision_threshold,
        color="#222222",
        linestyle="--",
        label="Ngưỡng sàng lọc tối ưu OOF",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Threshold",
        ylabel="Balanced Accuracy OOF",
    )
    axis.set_title("So sánh threshold và cách gộp xác suất trên OOF Train")
    axis.legend()
    axis.grid(alpha=0.25)
    paths.append(output_dir / "threshold_aggregation.png")
    _save_figure(figure, paths[-1])

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo biểu đồ portfolio từ artifact.")
    parser.add_argument("--artifacts", default="artifacts", help="Thư mục chứa artifact.")
    parser.add_argument("--output", default="reports/figures", help="Thư mục xuất ảnh.")
    arguments = parser.parse_args()
    for figure_path in create_portfolio_figures(arguments.artifacts, arguments.output):
        print(figure_path)
