import joblib
import numpy as np
import sklearn

from parkinson_voice.data import TARGET_COLUMN
from parkinson_voice.predict import load_bundle, predict_records, predict_subject_records


def test_saved_model_can_reload_and_reproduce_predictions(frame, artifact_path):
    inference = frame.drop(columns=[TARGET_COLUMN, "subject_id"]).head(12)
    first_bundle = load_bundle(artifact_path)
    first_records, first_subjects = predict_records(inference, first_bundle)
    second_bundle = joblib.load(artifact_path)
    second_records, second_subjects = predict_records(inference, second_bundle)
    np.testing.assert_allclose(first_records["screening_score"], second_records["screening_score"])
    assert first_subjects.equals(second_subjects)


def test_probabilities_are_between_zero_and_one(frame, artifact_path):
    inference = frame.drop(columns=[TARGET_COLUMN, "subject_id"]).head(20)
    records, subjects = predict_records(inference, load_bundle(artifact_path))
    assert records["screening_score"].between(0, 1).all()
    assert subjects["subject_screening_score"].between(0, 1).all()


def test_artifact_does_not_require_post_hoc_calibration(artifact_path):
    bundle = load_bundle(artifact_path)
    assert bundle["model_type"] == "logistic_regression"
    assert "calibrated" not in bundle["calibration"]


def test_artifact_records_oof_selected_patient_rule(artifact_path):
    bundle = load_bundle(artifact_path)
    assert bundle["aggregation"] == "median"
    assert 0 < bundle["decision_threshold"] < 1


def test_artifact_contains_environment_metadata(artifact_path):
    bundle = joblib.load(artifact_path)

    required = {
        "artifact_version",
        "schema_version",
        "python_version",
        "sklearn_version",
        "data_sha256",
        "feature_columns",
        "decision_threshold",
        "aggregation",
    }

    assert required.issubset(bundle)
    assert bundle["sklearn_version"] == sklearn.__version__


def test_predict_subject_records_produces_structured_report(frame, artifact_path):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(3)
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )
    bundle = load_bundle(artifact_path)
    report = predict_subject_records("sub_test", recordings, bundle)
    assert report["subject_id"] == "sub_test"
    assert 0 <= report["subject_screening_score"] <= 1
    assert report["screening_result"] in {"model-positive", "model-negative"}
    assert report["n_recordings"] == 3
    assert len(report["recording_scores"]) == 3


def test_predict_subject_records_detects_ood_value(frame, artifact_path):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(3).copy()
    sample_sub.loc[sample_sub.index[0], "MDVP:Fo(Hz)"] = 99999.0
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )
    bundle = load_bundle(artifact_path)
    bundle["feature_p1_p99"] = {"MDVP:Fo(Hz)": (50.0, 300.0)}
    report = predict_subject_records("sub_ood", recordings, bundle)
    assert report["reliability"] == "limited"
    assert any("FEATURE_OUTSIDE_TRAINING_RANGE" in w for w in report["warnings"])
