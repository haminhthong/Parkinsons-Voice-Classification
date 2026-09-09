import pytest

from parkinson_voice.evaluate import make_subject_folds
from parkinson_voice.features import (
    MODEL_FEATURES,
    REDUNDANT_FEATURES,
    make_logistic_pipeline,
    make_pipeline,
)
from parkinson_voice.model_selection import nested_subject_cross_fitted
from parkinson_voice.predict import load_bundle, predict_subject_records


def test_production_pipeline_has_only_scaler_and_model():
    pipeline = make_pipeline(make_logistic_pipeline().named_steps["model"])
    assert list(pipeline.named_steps) == ["scale", "model"]
    assert len(MODEL_FEATURES) == 20
    assert set(REDUNDANT_FEATURES).isdisjoint(MODEL_FEATURES)


def test_nested_cv_returns_each_subject_once(frame):
    _, predictions, _ = nested_subject_cross_fitted(
        frame,
        outer_splits=4,
        inner_splits=3,
        C_values=[0.1],
        class_weights=[None],
    )
    assert len(predictions) == frame["subject_id"].nunique()
    assert predictions["subject_id"].is_unique
    assert set(predictions["subject_id"]) == set(frame["subject_id"])
    assert predictions["subject_score"].between(0, 1).all()


def test_nested_folds_are_group_disjoint(frame):
    for fit_index, valid_index in make_subject_folds(frame, n_splits=4):
        assert set(frame.iloc[fit_index].subject_id).isdisjoint(frame.iloc[valid_index].subject_id)


def test_recording_has_score_only_and_subject_gets_decision(frame, artifact_path):
    bundle = load_bundle(artifact_path)
    recordings = (
        frame[frame["subject_id"] == frame["subject_id"].iloc[0]]
        .head(2)
        .drop(columns=["name", "subject_id", "status"])
        .to_dict(orient="records")
    )
    result = predict_subject_records("subject-test", recordings, bundle)
    assert result["screening_result"] in {"model-positive", "model-negative"}
    assert len(result["recording_scores"]) == 2
    assert "predicted_status" not in result


def test_status_is_rejected_at_inference(frame, artifact_path):
    bundle = load_bundle(artifact_path)
    recording = frame.head(1).drop(columns=["name", "subject_id"]).to_dict(orient="records")
    with pytest.raises(ValueError, match="status"):
        predict_subject_records("subject-test", recording, bundle)
