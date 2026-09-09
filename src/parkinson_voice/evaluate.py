"""Đánh giá leakage-aware ở cấp độ subject.

Cung cấp các hàm tạo phân chia K-Fold theo bệnh nhân, tính toán chỉ số hiệu năng
(Sensitivity/Recall, Specificity, Balanced Accuracy, F1-Macro, ROC-AUC, Brier score, ECE),
gộp score theo subject, tìm ngưỡng thống kê trên OOF và tính khoảng tin cậy
95% bằng subject bootstrap. Không có metric nào trong module này nên được gọi
là clinical validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from parkinson_voice.data import SUBJECT_COLUMN, TARGET_COLUMN, build_subject_table
from parkinson_voice.utils import normalize_aggregation


def make_subject_folds(
    frame: pd.DataFrame, *, n_splits: int = 5, random_state: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tạo danh sách các fold Cross-Validation chia phân tầng ở cấp độ bệnh nhân.

    Phân chia danh sách bệnh nhân duy nhất thành `n_splits` fold, sau đó ánh xạ chỉ số
    trở lại toàn bộ các dòng bản ghi tương ứng của bệnh nhân đó. Đảm bảo mọi validation fold
    đều chứa đủ cả 2 lớp (nhãn 0 và 1) và không bị rò rỉ bệnh nhân giữa fit/validation.

    Args:
        frame: DataFrame chứa toàn bộ bản ghi dữ liệu.
        n_splits: Số lượng fold Cross-Validation (mặc định 5).
        random_state: Seed ngẫu nhiên để tái lập cách chia fold.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: Danh sách các cặp (fit_indices, valid_indices).

    Raises:
        ValueError: Nếu số bệnh nhân ở lớp ít nhất không đủ để tạo `n_splits` fold.
        AssertionError: Nếu phát hiện rò rỉ bệnh nhân hoặc validation fold thiếu 1 lớp.
    """
    if n_splits < 2:
        raise ValueError("n_splits phải từ 2 trở lên.")

    subject_table = build_subject_table(frame).reset_index(drop=True)
    class_counts = subject_table[TARGET_COLUMN].value_counts()

    if len(class_counts) != 2:
        raise ValueError("Dataset phải có đúng hai lớp subject: 0 và 1.")

    if class_counts.min() < n_splits:
        raise ValueError(
            f"Không thể tạo {n_splits} fold có đủ hai lớp; lớp ít nhất chỉ có "
            f"{int(class_counts.min())} bệnh nhân."
        )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for subject_fit, subject_valid in splitter.split(
        subject_table,
        subject_table[TARGET_COLUMN],
    ):
        fit_ids = set(subject_table.iloc[subject_fit][SUBJECT_COLUMN])
        valid_ids = set(subject_table.iloc[subject_valid][SUBJECT_COLUMN])

        # Kiểm tra bảo vệ không rò rỉ nhóm bệnh nhân
        if not fit_ids.isdisjoint(valid_ids):
            raise AssertionError("Phát hiện rò rỉ nhóm bệnh nhân trong cross-validation.")

        fit_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(fit_ids).to_numpy())
        valid_index = np.flatnonzero(frame[SUBJECT_COLUMN].isin(valid_ids).to_numpy())

        # Đảm bảo fold đánh giá luôn có cả 2 lớp 0 và 1
        if frame.iloc[valid_index][TARGET_COLUMN].nunique() != 2:
            raise AssertionError("Fold validation bắt buộc phải có cả lớp 0 và lớp 1.")

        folds.append((fit_index, valid_index))

    return folds


def calculate_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Tính toán bộ chỉ số đánh giá hiệu năng thống nhất cho bài toán phân loại y tế.

    Bao gồm các chỉ số: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity (Độ nhạy),
    Specificity (Độ đặc hiệu), F1-macro, ROC-AUC và Brier Score.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_pred: Nhãn dự đoán nhị phân (0 hoặc 1).
        y_score: Xác suất dự đoán liên tục hoặc điểm decision score.

    Returns:
        dict[str, float]: Từ điển chứa tên các chỉ số và giá trị đo lường tương ứng.

    Raises:
        ValueError: Nếu y_true không chứa đủ 2 lớp nhãn {0, 1}.
    """
    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    if np.unique(y_true).size != 2:
        raise ValueError("Không thể đánh giá chỉ số: y_true bắt buộc phải có đủ hai lớp 0 và 1.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall/Sensitivity": recall,
        "Specificity": specificity,
        "NPV": npv,
        "F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
    }

    if np.all((y_score >= 0.0) & (y_score <= 1.0)):
        metrics["Brier score"] = brier_score_loss(y_true, y_score)
    else:
        metrics["Brier score"] = np.nan

    return metrics


def calculate_clinical_likelihood_ratios(
    sensitivity: float,
    specificity: float,
) -> dict[str, float | None]:
    """Tính toán tỷ số khả dĩ dương và âm (Likelihood Ratios: LR+, LR-) cho phân tích khám phá.

    LR+ = Sensitivity / (1 - Specificity)
    LR- = (1 - Sensitivity) / Specificity

    Lưu ý: Chỉ dùng cho mục đích báo cáo khám phá trong nghiên cứu sàng lọc,
    không dùng để suy diễn chẩn đoán lâm sàng độc lập.

    Args:
        sensitivity: Độ nhạy (Recall/Sensitivity) trong khoảng [0, 1].
        specificity: Độ đặc hiệu (Specificity) trong khoảng [0, 1].

    Returns:
        dict[str, float | None]: Tỷ số khả dĩ LR+ và LR- (hoặc None nếu chia cho 0).
    """
    lr_pos = (sensitivity / (1.0 - specificity)) if specificity < 1.0 else None
    lr_neg = ((1.0 - sensitivity) / specificity) if specificity > 0.0 else None
    return {"LR+": lr_pos, "LR-": lr_neg}


def aggregate_subject_predictions(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
    aggregation: str = "median",
) -> pd.DataFrame:
    """Gộp recording scores thành score và decision ở cấp subject.

    Args:
        frame: DataFrame chứa cột `subject_id` và `status`.
        probabilities: Mảng xác suất dự đoán của từng bản ghi âm.
        threshold: Ngưỡng quyết định nhị phân (mặc định 0.5).
        aggregation: Quy tắc gộp xác suất; production chỉ dùng `median`.

    Returns:
        pd.DataFrame: Bảng kết quả tổng hợp theo bệnh nhân.

    Raises:
        ValueError: Nếu tên quy tắc gộp không nằm trong danh sách hỗ trợ.
    """
    aggregation = normalize_aggregation(aggregation)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold phải nằm trong [0, 1].")
    probabilities = np.asarray(probabilities, dtype=float)
    if len(frame) != len(probabilities):
        raise ValueError("Số recording và số score phải bằng nhau.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Recording score phải hữu hạn và nằm trong [0, 1].")

    records = pd.DataFrame(
        {
            SUBJECT_COLUMN: frame[SUBJECT_COLUMN].to_numpy(),
            TARGET_COLUMN: frame[TARGET_COLUMN].to_numpy(),
            "probability": probabilities,
        }
    )

    subjects = records.groupby(SUBJECT_COLUMN, as_index=False).agg(
        status=(TARGET_COLUMN, "first"),
        probability=("probability", aggregation),
        recordings=("probability", "size"),
    )
    subjects["prediction"] = (subjects["probability"] >= threshold).astype(int)
    return subjects


def select_decision_threshold(
    subjects: pd.DataFrame,
) -> tuple[float, pd.DataFrame]:
    """Tìm ngưỡng thống kê tối ưu trên subject-level OOF.

    Mục tiêu canonical là Balanced Accuracy. Đây là threshold thống kê cho
    nghiên cứu, không phải một ngưỡng vận hành lâm sàng.

    Args:
        subjects: DataFrame đã gộp dự đoán ở cấp độ bệnh nhân.
    Returns:
        tuple[float, pd.DataFrame]: Ngưỡng tối ưu và bảng tìm kiếm ứng viên.
    """

    y_true = subjects[TARGET_COLUMN].to_numpy(dtype=int)
    probabilities = subjects["probability"].to_numpy(dtype=float)
    # ROC-AUC và Brier không phụ thuộc threshold; chỉ tính một lần cho cả bảng.
    score_metrics = calculate_metrics(
        y_true,
        (probabilities >= 0.5).astype(int),
        probabilities,
    )
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 201),
                probabilities,
            ]
        )
    )
    rows = []
    for threshold in candidates:
        prediction = (probabilities >= threshold).astype(int)
        tn = int(np.count_nonzero((y_true == 0) & (prediction == 0)))
        fp = int(np.count_nonzero((y_true == 0) & (prediction == 1)))
        fn = int(np.count_nonzero((y_true == 1) & (prediction == 0)))
        tp = int(np.count_nonzero((y_true == 1) & (prediction == 1)))
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        npv = tn / (tn + fn) if tn + fn else 0.0
        f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        rows.append(
            {
                "Threshold": float(threshold),
                "Accuracy": float(np.mean(y_true == prediction)),
                "Balanced Accuracy": (recall + specificity) / 2,
                "Precision": precision,
                "Recall/Sensitivity": recall,
                "Specificity": specificity,
                "NPV": npv,
                "F1-macro": (f1_positive + f1_negative) / 2,
                "ROC-AUC": score_metrics["ROC-AUC"],
                "Brier score": score_metrics["Brier score"],
            }
        )

    table = pd.DataFrame(rows)
    # Xếp hạng ứng viên theo thứ tự ưu tiên: Balanced Accuracy -> F1-macro -> Specificity
    ranked = table.assign(distance_from_default=(table["Threshold"] - 0.5).abs()).sort_values(
        ["Balanced Accuracy", "F1-macro", "Specificity", "distance_from_default"],
        ascending=[False, False, False, True],
    )
    return float(ranked.iloc[0]["Threshold"]), table


def expected_calibration_error(y_true, probabilities, *, n_bins: int = 5) -> float:
    """Tính sai số hiệu chỉnh kỳ vọng (Expected Calibration Error - ECE).

    ECE đo lường khoảng cách giữa xác suất dự đoán của mô hình và tần suất thực tế của lớp dương.
    Giá trị ECE càng gần 0 thể hiện xác suất xuất ra càng có độ tin cậy thực tế cao.

    Lưu ý quan trọng về cỡ mẫu:
    - Trên tập kiểm thử nhỏ (ví dụ 8 bệnh nhân chia 5 bin), ECE chỉ mang tính mô tả (descriptive only)
      vì mỗi bin có quá ít hoặc không có quan sát.
    - Đánh giá chất lượng calibration thực chất nên dựa chủ yếu vào OOF train hoặc nested CV folds.

    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        probabilities: Xác suất dự đoán từ mô hình [0, 1].
        n_bins: Số lượng khoảng bin chia xác suất (mặc định 5 bin).

    Returns:
        float: Giá trị chỉ số ECE.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(probabilities, edges[1:-1]), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        mask = bins == bin_index
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(error)


def bootstrap_subject_confidence_intervals(
    subjects: pd.DataFrame, *, n_bootstrap: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    """Ước lượng Khoảng Tin Cậy 95% (95% CI) bằng Patient Cluster Bootstrap.

    Thực hiện lấy mẫu có hoàn lại 5,000 lần ở cấp độ subject. Loại các mẫu
    bootstrap không hợp lệ (mẫu chỉ chứa duy nhất 1 lớp nhãn).

    Args:
        subjects: DataFrame kết quả dự đoán của từng bệnh nhân.
        n_bootstrap: Số lần lấy mẫu ngẫu nhiên (mặc định 5,000 lần).
        random_state: Seed ngẫu nhiên.

    Returns:
        pd.DataFrame: Bảng điểm ước lượng điểm (Point estimate) và khoảng tin cậy 95% CI.
    """
    point = calculate_metrics(
        subjects["status"],
        subjects["prediction"],
        subjects["probability"],
    )
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap phải lớn hơn 0.")

    # Bootstrap chỉ có 32 subject; dùng NumPy tránh tạo 5.000 DataFrame và gọi
    # lại nhiều hàm sklearn trong vòng lặp, nhưng vẫn giữ nguyên công thức metric.
    rng = np.random.default_rng(random_state)
    y_true = subjects["status"].to_numpy(dtype=int)
    y_pred = subjects["prediction"].to_numpy(dtype=int)
    y_score = subjects["probability"].to_numpy(dtype=float)
    samples: list[dict[str, float]] = []

    for sample_indices in rng.integers(0, len(subjects), size=(n_bootstrap, len(subjects))):
        sampled_true = y_true[sample_indices]
        if np.unique(sampled_true).size != 2:
            continue
        sampled_pred = y_pred[sample_indices]
        sampled_score = y_score[sample_indices]
        tn = int(np.count_nonzero((sampled_true == 0) & (sampled_pred == 0)))
        fp = int(np.count_nonzero((sampled_true == 0) & (sampled_pred == 1)))
        fn = int(np.count_nonzero((sampled_true == 1) & (sampled_pred == 0)))
        tp = int(np.count_nonzero((sampled_true == 1) & (sampled_pred == 1)))

        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        npv = tn / (tn + fn) if tn + fn else 0.0
        f1_positive = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
        positive_scores = sampled_score[sampled_true == 1]
        negative_scores = sampled_score[sampled_true == 0]
        score_comparison = positive_scores[:, None] - negative_scores[None, :]
        roc_auc = float(
            (np.count_nonzero(score_comparison > 0) + 0.5 * np.count_nonzero(score_comparison == 0))
            / score_comparison.size
        )
        samples.append(
            {
                "Accuracy": float(np.mean(sampled_true == sampled_pred)),
                "Balanced Accuracy": (recall + specificity) / 2,
                "Precision": precision,
                "Recall/Sensitivity": recall,
                "Specificity": specificity,
                "NPV": npv,
                "F1-macro": (f1_positive + f1_negative) / 2,
                "ROC-AUC": roc_auc,
                "Brier score": float(np.mean((sampled_true - sampled_score) ** 2)),
            }
        )

    distribution = pd.DataFrame(samples)
    if distribution.empty:
        raise ValueError("Bootstrap không tạo được mẫu có đủ hai lớp.")
    return pd.DataFrame(
        [
            {
                "Metric": metric,
                "Point estimate": value,
                "CI 2.5%": distribution[metric].quantile(0.025),
                "CI 97.5%": distribution[metric].quantile(0.975),
                "Valid bootstrap samples": len(distribution),
            }
            for metric, value in point.items()
        ]
    )
