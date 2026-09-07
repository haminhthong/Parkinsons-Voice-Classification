import io
from unittest.mock import Mock

import pandas as pd
from fastapi.testclient import TestClient

from app.api import app
from app.settings import MAX_UPLOAD_BYTES

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "scope": "research-only"}


def test_predict_endpoint_accepts_csv(data_path):
    content = pd.read_csv(data_path).drop(columns=["status"]).to_csv(index=False)
    with io.BytesIO(content.encode()) as stream:
        response = client.post("/predict", files={"file": ("sample.csv", stream, "text/csv")})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["records"]) == 195
    assert len(payload["subjects"]) == 32
    assert "không dùng để chẩn đoán" in payload["warning"]
    assert payload["aggregation"] == "median"
    assert 0 < payload["decision_threshold"] < 1


def test_empty_csv_returns_422():
    response = client.post(
        "/predict",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 422


def test_large_upload_returns_413():
    content = b"x" * (MAX_UPLOAD_BYTES + 1)
    response = client.post(
        "/predict",
        files={"file": ("large.csv", content, "text/csv")},
    )
    assert response.status_code == 413


def test_internal_error_does_not_expose_path(monkeypatch, data_path):
    monkeypatch.setattr(
        "app.api.predict_records",
        Mock(side_effect=RuntimeError(r"C:\secret\artifact\internal_file.py")),
    )

    with data_path.open("rb") as stream:
        response = client.post("/predict", files={"file": ("sample.csv", stream, "text/csv")})

    assert response.status_code == 500
    assert "C:\\secret" not in response.text


def test_predict_subject_endpoint_success(frame):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(3)
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )

    response = client.post(
        "/predict/subject",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["subject_id"] == "test_subject_01"
    assert "subject_screening_score" in payload
    assert "screening_result" in payload
    assert payload["reliability"] == "limited"
    assert payload["n_recordings"] == 3
    assert len(payload["recording_scores"]) == 3


def test_predict_subject_endpoint_forbids_status_field(frame):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(1)
    recordings = sample_sub.drop(columns=["name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )
    assert "status" in recordings[0]

    response = client.post(
        "/predict/subject",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 422


def test_predict_subject_endpoint_warns_on_single_recording(frame):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(1)
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )

    response = client.post(
        "/predict/subject",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["reliability"] == "limited"
    assert any("INSUFFICIENT_RECORDINGS" in w for w in payload["warnings"])
