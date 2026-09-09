FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/project/src:/project

WORKDIR /project
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /project

FROM base AS api
USER root
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS ui
USER root
COPY requirements-ui.txt .
RUN pip install --no-cache-dir -r requirements-ui.txt
USER appuser
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
