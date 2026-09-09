"""Sinh notebook Colab tự chứa để tái lập kết quả repository."""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").is_dir():
    ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "02_colab_reproducible.ipynb"
INCLUDED_FILES = [
    "configs/default.json",
    "data/parkinsons.csv",
    *[
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in sorted((ROOT / "src").glob("*.py"))
    ],
]


def make_payload() -> str:
    """Nén mã nguồn và dữ liệu tối thiểu vào một chuỗi base64."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative_path in INCLUDED_FILES:
            path = ROOT / relative_path
            info = zipfile.ZipInfo(relative_path, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}


def py(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip(),
    }


def build() -> Path:
    payload = make_payload()
    expected_metrics = (ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
        "accelerator": "CPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
    }
    notebook["cells"] = [
        md("""
# Leakage-Aware Parkinson’s Voice Classification — Google Colab

Notebook này tái lập đúng quy trình canonical của repository: dữ liệu bảng gồm 22
đặc trưng acoustic được kiểm tra, loại 2 đặc trưng dư thừa còn 20 đặc trưng cố định,
sau đó đánh giá bằng nested stratified subject CV (4 outer × 3 inner). Production chỉ
dùng `StandardScaler → LogisticRegression`, gộp median ở cấp subject và chọn threshold
từ OOF train.

Đây là prototype nghiên cứu/sàng lọc, không phải công cụ chẩn đoán. Chọn **Runtime → Run all**.
"""),
        md("## 1. Khóa môi trường chạy"),
        py("""
import os, subprocess, sys
IN_COLAB = "google.colab" in sys.modules
PACKAGES = [
    "pandas==2.2.3", "numpy==2.1.3", "scikit-learn==1.7.1",
    "joblib==1.4.2", "matplotlib==3.9.2",
]
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES], check=True)
os.environ.setdefault("MPLBACKEND", "Agg")
print("Môi trường:", "Google Colab" if IN_COLAB else "Python cục bộ")
"""),
        md("""
## 2. Khôi phục dự án tối thiểu

Notebook tự chứa dữ liệu, cấu hình và toàn bộ module `src` cần cho audit, train và
inference. Không phụ thuộc đường dẫn Windows, Google Drive hoặc GitHub.
"""),
        py(f"""
import base64, io, shutil, zipfile
from pathlib import Path

PAYLOAD = "{payload}"
PROJECT_DIR = (
    Path("/content/parkinsons-voice-classification")
    if IN_COLAB
    else Path.cwd() / "parkinsons-colab-runtime"
)
if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)
PROJECT_DIR.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(PROJECT_DIR)
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print("Thư mục chạy:", PROJECT_DIR)
"""),
        md("## 3. Kiểm tra phiên bản, checksum và schema"),
        py("""
import json, joblib, numpy as np, pandas as pd, sklearn
from src.data import ORIGINAL_FEATURES, SUBJECT_COLUMN, TARGET_COLUMN, load_data
from src.features import MODEL_FEATURES, REDUNDANT_FEATURES
from src.utils import sha256_file

expected_versions = {
    "pandas": "2.2.3", "numpy": "2.1.3", "scikit-learn": "1.7.1",
    "joblib": "1.4.2",
}
actual_versions = {
    "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__,
    "joblib": joblib.__version__,
}
if IN_COLAB:
    assert actual_versions == expected_versions, (actual_versions, expected_versions)
DATA_PATH = Path("data/parkinsons.csv")
DATA_SHA256 = "32e6040916d2f5b80b49589d925a92bd25420687c76be19d72e37205e104abe6"
assert sha256_file(DATA_PATH) == DATA_SHA256
frame = load_data(DATA_PATH)
assert len(frame) == 195 and frame[SUBJECT_COLUMN].nunique() == 32
assert len(ORIGINAL_FEATURES) == 22 and len(MODEL_FEATURES) == 20
assert set(REDUNDANT_FEATURES) == {"Jitter:DDP", "Shimmer:DDA"}
assert set(frame[TARGET_COLUMN].unique()) == {0, 1}
print(f"✅ {len(frame)} recordings | 32 subjects | 22 source features | 20 model features")
display(frame.head(3))
"""),
        md("## 4. Audit phân chia theo subject"),
        py("""
from src.audit import build_data_manifest
from src.evaluate import make_subject_folds

manifest = build_data_manifest(frame, DATA_PATH)
assert manifest["duplicate_recording_names"] == 0
assert manifest["duplicate_full_feature_vectors"] == 0
folds = make_subject_folds(frame, n_splits=4, random_state=42)
for number, (fit_index, valid_index) in enumerate(folds, 1):
    fit_ids = set(frame.iloc[fit_index][SUBJECT_COLUMN])
    valid_ids = set(frame.iloc[valid_index][SUBJECT_COLUMN])
    assert fit_ids.isdisjoint(valid_ids)
    assert frame.iloc[valid_index][TARGET_COLUMN].nunique() == 2
    print(f"Fold {number}: subject disjoint, validation đủ hai lớp")
print("✅ Không có overlap subject trong outer CV")
"""),
        md("## 5. Huấn luyện canonical và sinh artifact"),
        py("""
from src.train import train

ARTIFACT_DIR = Path("artifacts")
comparison = train(DATA_PATH, ARTIFACT_DIR)
display(comparison)
"""),
        md("## 6. Kiểm tra kết quả và artifact release"),
        py(f"""
from src.predict import load_bundle

EXPECTED = json.loads({json.dumps(expected_metrics, ensure_ascii=False)})
ACTUAL = json.loads((ARTIFACT_DIR / "metrics.json").read_text(encoding="utf-8"))
assert ACTUAL["dataset"]["dataset_sha256"] == EXPECTED["dataset"]["dataset_sha256"]
assert ACTUAL["dataset"]["n_recordings"] == 195
assert ACTUAL["dataset"]["n_subjects"] == 32
assert ACTUAL["selection"] == EXPECTED["selection"]
for metric in ("Balanced Accuracy", "F1-macro", "ROC-AUC"):
    assert np.isclose(
        ACTUAL["nested_cv_subject"][metric],
        EXPECTED["nested_cv_subject"][metric],
        rtol=0,
        atol=1e-12,
    )
bundle = load_bundle(ARTIFACT_DIR / "releases" / "v1.0.0" / "model.joblib")
assert bundle["feature_columns"] == MODEL_FEATURES
assert bundle["aggregation"] == "median"
print("✅ Kết quả canonical và release model khớp repository")
print(json.dumps(ACTUAL, ensure_ascii=False, indent=2))
"""),
        md("## 7. Suy luận dữ liệu mới — tùy chọn"),
        py("""
from src.predict import load_bundle, predict_records

inference_frame = frame.drop(columns=[TARGET_COLUMN, SUBJECT_COLUMN])
record_results, subject_results = predict_records(inference_frame, bundle)
assert "screening_score" in record_results
assert "subject_screening_score" in subject_results
display(subject_results.head())
print("Input inference chỉ dùng name và feature; không truyền status.")
"""),
        md("""
## Kết luận

Notebook hoàn tất khi tất cả assertion màu xanh: checksum/schema đúng, outer CV không
overlap subject, kết quả nested CV khớp artifact và release model dùng đúng 20-feature
contract. Các score chỉ là tín hiệu nghiên cứu; cần cohort ngoài và validation lâm sàng
trước mọi diễn giải y tế.
"""),
    ]
    OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
