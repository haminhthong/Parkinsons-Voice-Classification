"""Kiểm thử aggregation, threshold và bootstrap ở cấp subject."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    aggregate_subject_predictions,
    bootstrap_subject_confidence_intervals,
    expected_calibration_error,
    make_subject_folds,
    select_decision_threshold,
)


def test_expected_calibration_error_is_bounded():
    value = expected_calibration_error([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert 0 <= value <= 1


def test_bootstrap_confidence_intervals_are_valid(frame):
    _, valid_index = make_subject_folds(frame, n_splits=4, random_state=42)[0]
    valid = frame.iloc[valid_index]
    subjects = valid.groupby("subject_id", as_index=False).agg(status=("status", "first"))
    subjects["probability"] = np.where(subjects["status"].eq(1), 0.8, 0.2)
    subjects["prediction"] = (subjects["probability"] >= 0.5).astype(int)

    intervals = bootstrap_subject_confidence_intervals(subjects, n_bootstrap=100)
    assert intervals["Valid bootstrap samples"].min() > 0
    assert (intervals["CI 2.5%"] <= intervals["CI 97.5%"]).all()


def test_production_aggregation_is_median(frame):
    probabilities = np.linspace(0.05, 0.95, len(frame))
    subjects = aggregate_subject_predictions(frame, probabilities, aggregation="median")
    assert len(subjects) == frame["subject_id"].nunique()
    assert subjects["probability"].between(0, 1).all()


def test_non_median_aggregation_is_rejected(frame):
    with pytest.raises(ValueError, match="median"):
        aggregate_subject_predictions(frame, np.full(len(frame), 0.5), aggregation="mean")


def test_threshold_selection_prioritizes_balanced_accuracy():
    subjects = pd.DataFrame(
        {
            "status": [0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.6, 0.9],
        }
    )
    threshold, table = select_decision_threshold(subjects)
    selected = table.loc[np.isclose(table["Threshold"], threshold)].iloc[0]
    assert 0 <= threshold <= 1
    assert selected["Balanced Accuracy"] == table["Balanced Accuracy"].max()
