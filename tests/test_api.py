from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "scope": "research-only"}


def test_predict_endpoint_success(frame):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(3)
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )

    response = client.post(
        "/predict",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["subject_id"] == "test_subject_01"
    assert "subject_screening_score" in payload
    assert payload["screening_result"] in {"above_internal_threshold", "below_internal_threshold"}
    assert payload["n_recordings"] == 3
    assert len(payload["recording_scores"]) == 3
    assert "warning" in payload


def test_predict_endpoint_forbids_status_field(frame):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(1)
    recordings = sample_sub.drop(columns=["name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )
    assert "status" in recordings[0]

    response = client.post(
        "/predict",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 422


def test_predict_endpoint_warns_on_fewer_recordings(frame):
    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(1)
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )

    response = client.post(
        "/predict",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 200
    payload = response.json()
    assert any("FEWER_RECORDINGS_WARNING" in w for w in payload["warnings"])


def test_predict_endpoint_empty_recordings():
    response = client.post(
        "/predict",
        json={"subject_id": "test_subject_01", "recordings": []},
    )
    assert response.status_code == 422


def test_internal_error_does_not_expose_path(monkeypatch, frame):
    monkeypatch.setattr(
        "app.api.predict_subject_records",
        Mock(side_effect=RuntimeError(r"C:\secret\artifact\internal_file.py")),
    )

    sample_sub = frame[frame["subject_id"] == frame["subject_id"].iloc[0]].head(1)
    recordings = sample_sub.drop(columns=["status", "name", "subject_id"], errors="ignore").to_dict(
        orient="records"
    )

    response = client.post(
        "/predict",
        json={"subject_id": "test_subject_01", "recordings": recordings},
    )
    assert response.status_code == 500
    assert "C:\\secret" not in response.text
