"""Ứng dụng REST API FastAPI cho dịch vụ sàng lọc đặc trưng giọng nói Parkinson.

Cung cấp các endpoint:
- GET `/health`: Kiểm tra trạng thái hoạt động của dịch vụ API.
- POST `/predict`: Tiện ích batch/research cho CSV chứa 20 hoặc 22 đặc trưng âm học.
- POST `/v1/screen/subject`: Endpoint canonical nhận nhiều recording của một subject.
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.settings import (
    ALLOWED_CONTENT_TYPES,
    ARTIFACT_PATH,
    MAX_INFERENCE_ROWS,
    MAX_UPLOAD_BYTES,
    RESEARCH_WARNING,
)
from src.predict import load_bundle, predict_records, predict_subject_records

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
        description="Danh sách các bản ghi âm, mỗi bản ghi chứa các đặc trưng âm học đã trích xuất.",
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
        "API sàng lọc đặc trưng âm học giọng nói Parkinson chống rò rỉ dữ liệu "
        "(Research Screening Prototype). "
        "Lưu ý: Chỉ nhận bảng 20 hoặc 22 đặc trưng âm học đã trích xuất sẵn (CSV/JSON), "
        "KHÔNG nhận file âm thanh thô WAV/MP3. Không dùng cho mục đích chẩn đoán y tế."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Endpoint kiểm tra sức khỏe hệ thống API (Health Check)."""
    return {"status": "ok", "scope": "research-only"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    """Cho biết model bundle đã sẵn sàng nhận inference hay chưa."""
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None and ARTIFACT_PATH.is_file():
        bundle = load_bundle(ARTIFACT_PATH)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")
    return {"status": "ready", "model_version": str(bundle["model_version"])}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    """Trả về contract triển khai, không trả thông tin nhãn của input."""
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None and ARTIFACT_PATH.is_file():
        bundle = load_bundle(ARTIFACT_PATH)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model chưa sẵn sàng.")
    return {
        "model_version": str(bundle["model_version"]),
        "model_type": bundle["model_type"],
        "feature_count": len(bundle["feature_columns"]),
        "aggregation": bundle["aggregation"],
        "decision_threshold": float(bundle["decision_threshold"]),
        "training_dataset": "UCI Parkinsons",
        "scope": "research-only",
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
) -> dict:
    """Endpoint nhận tệp CSV dữ liệu giọng nói và trả về dự đoán phân loại."""
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        if not (file.filename and file.filename.lower().endswith(".csv")):
            raise HTTPException(
                status_code=415,
                detail="Content-Type không được hỗ trợ.",
            )

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Tệp vượt quá giới hạn 2 MB.",
        )

    try:
        frame = pd.read_csv(io.BytesIO(content))
    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=422,
            detail="Không thể đọc cấu trúc CSV.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail="Lỗi khi đọc dữ liệu CSV.",
        ) from exc

    if frame.empty:
        raise HTTPException(status_code=422, detail="CSV không có bản ghi.")

    if len(frame) > MAX_INFERENCE_ROWS:
        raise HTTPException(
            status_code=413,
            detail="CSV vượt quá giới hạn số dòng cho phép.",
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
        records, subjects = await run_in_threadpool(
            predict_records,
            frame,
            bundle,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Lỗi inference ngoài dự kiến")
        raise HTTPException(
            status_code=500,
            detail="Dịch vụ không thể xử lý yêu cầu.",
        ) from exc

    return {
        "warning": RESEARCH_WARNING,
        "model": bundle["champion_name"],
        "model_version": str(bundle["model_version"]),
        "aggregation": bundle["aggregation"],
        "decision_threshold": float(bundle["decision_threshold"]),
        "records": records.to_dict(orient="records"),
        "subjects": subjects.to_dict(orient="records"),
    }


@app.post("/v1/screen/subject")
@app.post("/predict/subject", include_in_schema=False)
async def predict_subject(
    request: Request,
    payload: SubjectInferenceRequest,
) -> dict:
    """Endpoint suy luận sàng lọc cho một đối tượng từ cấu trúc JSON gồm nhiều bản ghi.

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
        "model": bundle["champion_name"],
        **result,
    }
