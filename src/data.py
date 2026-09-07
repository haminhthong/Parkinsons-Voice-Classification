"""Nạp, kiểm tra schema và quản lý subject identity.

Cung cấp các hàm kiểm tra tính hợp lệ của dữ liệu (schema validation), trích xuất mã bệnh nhân
(subject_id) từ tên tệp ghi âm, và thực hiện phân chia tập dữ liệu ở cấp độ bệnh nhân
(Patient-level split) nhằm chống rò rỉ dữ liệu (Data Leakage).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Tên cột tiêu chuẩn của bộ dữ liệu UCI Parkinsons.
ID_COLUMN = "name"
TARGET_COLUMN = "status"
SUBJECT_COLUMN = "subject_id"

# Danh sách 22 đặc trưng tần số/biên độ giọng nói gốc từ UCI Dataset
ORIGINAL_FEATURES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE",
]


def subject_id_from_name(name: str) -> str:
    """Trích xuất mã bệnh nhân (subject_id) từ tên bản ghi âm.

    Ví dụ: 'phon_R01_S01_1' -> 'phon_R01_S01'

    Args:
        name: Tên của bản ghi âm (chuỗi định dạng `..._SXX_Y`).

    Returns:
        Mã định danh bệnh nhân (chứa các bản ghi âm của cùng một người).

    Raises:
        ValueError: Nếu tên bản ghi không tuân theo đúng quy tắc đặt tên.
    """
    value = str(name).strip()
    if not re.match(r"^.+_[^_]+$", value):
        raise ValueError(f"Tên bản ghi không đúng định dạng: {name!r}")
    return value.rsplit("_", 1)[0]


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    require_name: bool = True,
) -> pd.DataFrame:
    """Kiểm tra tính hợp lệ của DataFrame đầu vào và trích xuất thông tin bệnh nhân.

    Thực hiện kiểm tra nghiêm ngặt:
    - Kiểm tra xem có đủ 22 đặc trưng số bắt buộc hay không.
    - Ép kiểu dữ liệu số và phát hiện giá trị khuyết (NaN/inf).
    - Trích xuất cột `subject_id` từ `name`.
    - Đảm bảo mỗi bệnh nhân có nhãn đồng nhất cho mọi bản ghi.

    Args:
        frame: DataFrame chứa dữ liệu đầu vào.
        require_target: Nếu True, yêu cầu phải có cột nhãn `status`.
        require_name: Nếu True, yêu cầu phải có cột định danh `name`.

    Returns:
        pd.DataFrame: Bản sao DataFrame đã kiểm tra và trích xuất `subject_id`.

    Raises:
        ValueError: Nếu thiếu cột, dữ liệu sai định dạng hoặc nhãn không đồng nhất.
    """
    if frame.empty:
        raise ValueError("CSV không có bản ghi.")

    required = set(ORIGINAL_FEATURES)

    if require_target:
        required.add(TARGET_COLUMN)
    if require_name:
        required.add(ID_COLUMN)

    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {missing}")

    result = frame.copy()
    numeric_columns = ORIGINAL_FEATURES + ([TARGET_COLUMN] if require_target else [])

    # Ép kiểu dữ liệu về số và kiểm tra lỗi định dạng
    for column in numeric_columns:
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Cột {column!r} phải chứa dữ liệu số.") from exc

    # Kiểm tra giá trị khuyết (NaN) hoặc không hữu hạn (±inf)
    if result[numeric_columns].isna().any().any():
        raise ValueError("Dữ liệu có giá trị thiếu (NaN).")
    if not np.isfinite(result[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Dữ liệu có giá trị không hữu hạn (NaN hoặc ±inf).")

    # Kiểm tra nhãn status (chỉ gồm 0 và 1)
    if require_target and not set(result[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("status chỉ được gồm hai nhãn 0 và 1.")

    # Trích xuất mã bệnh nhân và kiểm tra tính đồng nhất của nhãn
    if require_name:
        if result[ID_COLUMN].isna().any():
            raise ValueError("Cột name không được để trống.")
        result[SUBJECT_COLUMN] = result[ID_COLUMN].map(subject_id_from_name)

        if require_target:
            label_counts = result.groupby(SUBJECT_COLUMN)[TARGET_COLUMN].nunique()
            if not label_counts.eq(1).all():
                invalid = label_counts[label_counts.ne(1)].index.tolist()
                raise ValueError(f"Nhãn không nhất quán trong các bản ghi của bệnh nhân: {invalid}")

    return result


def load_data(path: str | Path) -> pd.DataFrame:
    """Đọc dữ liệu từ tệp CSV và kiểm tra tính hợp lệ.

    Args:
        path: Đường dẫn tới tệp CSV chứa dữ liệu.

    Returns:
        pd.DataFrame: DataFrame hợp lệ đã trích xuất `subject_id`.
    """
    return validate_dataframe(pd.read_csv(path), require_target=True, require_name=True)


def build_subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Tạo bảng đại diện duy nhất 1 dòng cho mỗi bệnh nhân (`subject_id`).

    Dùng để thực hiện phân chia và kiểm tra Cross-Validation ở cấp độ bệnh nhân.
    Hàm holdout cũ vẫn được giữ để tương thích test/experiment; production dùng nested CV.

    Args:
        frame: DataFrame đầy đủ các bản ghi.

    Returns:
        pd.DataFrame: Bảng gom nhóm theo bệnh nhân gồm cột `subject_id`, `status` và `recordings`.
    """
    return frame.groupby(SUBJECT_COLUMN, as_index=False).agg(
        status=(TARGET_COLUMN, "first"), recordings=(ID_COLUMN, "size")
    )


def subject_holdout_split(
    frame: pd.DataFrame, *, test_size: float = 0.25, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phân chia tập dữ liệu Train / Test độc lập theo từng bệnh nhân.

    Đảm bảo 100% bệnh nhân ở tập Test hoàn toàn chưa từng xuất hiện trong tập Train,
    ngăn ngừa triệt để rò rỉ dữ liệu giữa các bản ghi âm của cùng một người.

    Args:
        frame: DataFrame chứa toàn bộ bản ghi.
        test_size: Tỷ lệ tập kiểm tra (mặc định 0.25).
        random_state: Seed ngẫu nhiên để tái lập kết quả chia.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (train_frame, test_frame)

    Raises:
        AssertionError: Nếu có bất kỳ bệnh nhân nào xuất hiện ở cả 2 tập.
    """
    subjects = build_subject_table(frame)
    train_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_size,
        random_state=random_state,
        stratify=subjects[TARGET_COLUMN],
    )

    train_ids = set(train_subjects[SUBJECT_COLUMN])
    test_ids = set(test_subjects[SUBJECT_COLUMN])

    # Kiểm tra bảo vệ chống rò rỉ tập dữ liệu
    if not train_ids.isdisjoint(test_ids):
        raise AssertionError("Phát hiện bệnh nhân xuất hiện ở cả tập huấn luyện và tập kiểm tra.")

    return (
        frame[frame[SUBJECT_COLUMN].isin(train_ids)].copy(),
        frame[frame[SUBJECT_COLUMN].isin(test_ids)].copy(),
    )
