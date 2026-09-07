"""Cấu hình dùng chung cho ứng dụng API và UI."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "releases" / "v1.0.0" / "model.joblib"
RESEARCH_WARNING = (
    "Chỉ phục vụ nghiên cứu và học tập, không dùng để chẩn đoán hoặc thay thế tư vấn y khoa."
)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
MAX_INFERENCE_ROWS = 10_000
ALLOWED_CONTENT_TYPES = {
    "text/csv",
    "application/csv",
    "application/vnd.ms-excel",
    "text/plain",
}
