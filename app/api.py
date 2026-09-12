"""Ứng dụng REST API FastAPI cho sàng lọc đặc trưng giọng nói Parkinson.

Cung cấp các endpoint:
- GET `/health`: Kiểm tra trạng thái hoạt động của dịch vụ API.
- POST `/predict`: Sàng lọc đối tượng từ danh sách các bản ghi đặc trưng âm học.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.settings import ARTIFACT_PATH, RESEARCH_WARNING
from parkinson_voice.predict import load_bundle, predict_subject_records

logger = logging.getLogger(__name__)


class SubjectInferenceRequest(BaseModel):
    """Schema yêu cầu suy luận theo cấp bệnh nhân (chặn nhận nhãn huấn luyện)."""

    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(
        ...,
        min_length=1,
        description="Định danh bệnh nhân / đối tượng khảo sát.",
    )
    recordings: list[dict[str, float]] = Field(
        ...,
        min_length=1,
        description="Danh sách các bản ghi âm chứa đặc trưng âm học.",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Quản lý vòng đời ứng dụng FastAPI, nạp sẵn artifact mô hình khi khởi động."""
    if ARTIFACT_PATH.is_file():
        app.state.model_bundle = load_bundle(ARTIFACT_PATH)
    else:
        app.state.model_bundle = None
    yield
    app.state.model_bundle = None


app = FastAPI(
    title="Parkinson Voice Feature Screening API",
    description=(
        "API sàng lọc đặc trưng âm học giọng nói Parkinson ở cấp độ subject "
        "(Research Screening Prototype). "
        "Lưu ý: Chỉ nhận bảng đặc trưng âm học số, KHÔNG nhận audio thô (WAV/MP3). "
        "Không dùng cho mục đích chẩn đoán y tế."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint kiểm tra sức khỏe hệ thống API (Health Check)."""
    return {"status": "ok", "scope": "research-only"}


@app.post("/predict")
@app.post("/predict/subject", include_in_schema=False)
@app.post("/v1/screen/subject", include_in_schema=False)
async def predict(
    request: Request,
    payload: SubjectInferenceRequest,
) -> dict:
    """Endpoint suy luận sàng lọc cho một đối tượng từ danh sách bản ghi âm.

    Đầu vào: subject_id và danh sách recordings chứa 20 hoặc 22 đặc trưng âm học.
    Nghiêm cấm: Truyền trường nhãn 'status' hoặc các cột train-only (trả về mã 422).
    """
    for idx, recording in enumerate(payload.recordings):
        if "status" in recording:
            raise HTTPException(
                status_code=422,
                detail=f"Bản ghi thứ {idx + 1} chứa nhãn 'status'. Endpoint suy luận không nhận nhãn huấn luyện.",
            )

    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        if ARTIFACT_PATH.is_file():
            bundle = load_bundle(ARTIFACT_PATH)
        else:
            raise HTTPException(
                status_code=500,
                detail="Không tìm thấy mô hình suy luận trên hệ thống.",
            )

    try:
        result = await run_in_threadpool(
            predict_subject_records,
            payload.subject_id,
            payload.recordings,
            bundle,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi suy luận cấp bệnh nhân ngoài dự kiến")
        raise HTTPException(
            status_code=500,
            detail="Dịch vụ không thể xử lý yêu cầu.",
        ) from exc

    return {
        "warning": RESEARCH_WARNING,
        **result,
    }
