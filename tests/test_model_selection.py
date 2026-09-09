"""Kiểm thử lựa chọn cấu hình duy nhất của pipeline production."""

import numpy as np

from parkinson_voice.evaluate import make_subject_folds
from parkinson_voice.features import MODEL_FEATURES, make_logistic_pipeline
from parkinson_voice.model_selection import search_logistic_configuration


def test_logistic_search_returns_subject_level_candidates(frame):
    folds = make_subject_folds(frame, n_splits=3, random_state=42)
    selected = search_logistic_configuration(
        frame,
        folds,
        feature_columns=MODEL_FEATURES,
        C_values=[0.1, 1.0],
        class_weights=[None],
    )

    assert len(selected["comparison"]) == 2
    assert selected["C"] in {0.1, 1.0}
    assert 0 <= selected["threshold"] <= 1
    assert selected["subjects"]["subject_id"].is_unique


def test_logistic_pipeline_has_only_scaler_and_model():
    pipeline = make_logistic_pipeline(C=0.1)
    assert list(pipeline.named_steps) == ["scale", "model"]
    assert pipeline.named_steps["model"].get_params()["solver"] == "liblinear"
    assert np.isfinite(pipeline.named_steps["model"].get_params()["C"])
