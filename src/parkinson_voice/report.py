"""Sinh các biểu đồ canonical từ artifact evaluation của pipeline."""

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

    # Báo cáo chỉ đọc artifact canonical; thiếu artifact là lỗi cấu hình.
    evaluation_dir = artifact_dir / "evaluation"
    selection_path = evaluation_dir / "inner_selection.csv"
    scores_path = evaluation_dir / "cross_fitted_subject_predictions.csv"
    threshold_path = evaluation_dir / "threshold_search.csv"
    required_paths = (selection_path, scores_path, threshold_path, artifact_dir / "metrics.json")
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Thiếu artifact báo cáo canonical: {missing_paths}")
    inner_selection = pd.read_csv(selection_path)
    scores = pd.read_csv(scores_path)
    threshold_search = pd.read_csv(threshold_path)
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
    stability = pd.DataFrame(
        {
            "Feature": ["20 model features (fixed contract)"],
            "Contract frequency": [1.0],
        }
    )

    paths: list[Path] = []

    # Biểu đồ 1: So sánh cấu hình Logistic Regression trong inner OOF
    figure, axis = plt.subplots(figsize=(9, 5))
    selection_plot = (
        inner_selection.groupby(["C", "class_weight"], dropna=False, as_index=False)[
            "Balanced Accuracy"
        ]
        .mean()
        .sort_values("Balanced Accuracy")
    )
    labels = selection_plot.apply(lambda row: f"C={row['C']}, weight={row['class_weight']}", axis=1)
    axis.barh(labels, selection_plot["Balanced Accuracy"], color="#1f5f8b")
    axis.set(xlim=(0, 1), xlabel="Inner OOF Balanced Accuracy trung bình")
    axis.set_title("Inner selection của Logistic Regression theo subject")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output_dir / "inner_logistic_selection.png")
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
    paths.append(output_dir / "cross_fitted_subject_scores.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 3: Contract 20 đặc trưng cố định, không chạy feature selection
    stability_plot = stability.sort_values("Contract frequency")
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        stability_plot["Feature"],
        stability_plot["Contract frequency"],
        color="#4c78a8",
    )
    axis.set(xlim=(0, 1), xlabel="Tỷ lệ contract")
    axis.set_title("Feature contract cố định (không phải feature selection)")
    axis.grid(axis="x", alpha=0.25)
    paths.append(output_dir / "fixed_feature_contract.png")
    _save_figure(figure, paths[-1])

    # Biểu đồ 4: Quét threshold quyết định trên OOF subject-level.
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
    axis.set_title("Quét threshold trên OOF subject-level")
    axis.legend()
    axis.grid(alpha=0.25)
    paths.append(output_dir / "threshold_search.png")
    _save_figure(figure, paths[-1])

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo biểu đồ portfolio từ artifact.")
    parser.add_argument("--artifacts", default="artifacts", help="Thư mục chứa artifact.")
    parser.add_argument("--output", default="reports/figures", help="Thư mục xuất ảnh.")
    arguments = parser.parse_args()
    for figure_path in create_portfolio_figures(arguments.artifacts, arguments.output):
        print(figure_path)
