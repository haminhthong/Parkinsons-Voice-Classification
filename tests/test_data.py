import numpy as np
import pandas as pd
import pytest

from src.data import ORIGINAL_FEATURES, TARGET_COLUMN, validate_dataframe


def test_dataset_has_exactly_22_original_features(frame):
    assert len(ORIGINAL_FEATURES) == 22
    assert set(ORIGINAL_FEATURES).issubset(frame.columns)


def test_status_contains_only_zero_and_one(frame):
    assert set(frame[TARGET_COLUMN].unique()) == {0, 1}


def test_inference_missing_column_is_rejected(frame):
    invalid = frame.drop(columns=[ORIGINAL_FEATURES[0], TARGET_COLUMN])
    with pytest.raises(ValueError, match="Thiếu cột"):
        validate_dataframe(invalid, require_target=False)


def test_non_binary_status_is_rejected(frame):
    invalid = frame.copy()
    invalid.loc[invalid.index[0], TARGET_COLUMN] = 2
    with pytest.raises(ValueError, match="0 và 1"):
        validate_dataframe(invalid)


def test_empty_csv_is_rejected():
    with pytest.raises(ValueError, match="không có bản ghi"):
        validate_dataframe(pd.DataFrame())


def test_non_numeric_feature_is_rejected(frame):
    invalid = frame.copy()
    invalid["MDVP:Fo(Hz)"] = invalid["MDVP:Fo(Hz)"].astype(object)
    invalid.loc[0, "MDVP:Fo(Hz)"] = "không phải số"

    with pytest.raises(ValueError, match="phải chứa dữ liệu số"):
        validate_dataframe(invalid)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_feature_is_rejected(frame, value):
    invalid = frame.copy()
    invalid.loc[0, "MDVP:Fo(Hz)"] = value

    with pytest.raises(ValueError):
        validate_dataframe(invalid)


def test_inconsistent_subject_label_is_rejected(frame):
    invalid = frame.copy()
    subject = invalid.loc[0, "subject_id"] if "subject_id" in invalid.columns else "phon_R01_S01"
    if "subject_id" in invalid.columns:
        invalid = invalid.drop(columns=["subject_id"])

    rows = invalid.index[invalid["name"].str.startswith(subject)]
    invalid.loc[rows[0], TARGET_COLUMN] = 1 - invalid.loc[rows[0], TARGET_COLUMN]

    with pytest.raises(ValueError, match="Nhãn không nhất quán"):
        validate_dataframe(invalid)
