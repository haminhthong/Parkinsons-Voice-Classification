"""Đồng bộ vùng kết quả sinh tự động trong README với artifact canonical."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
README_PATH = PROJECT_ROOT / "README.md"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
START_MARKER = "<!-- GENERATED_RESULTS_START -->"
END_MARKER = "<!-- GENERATED_RESULTS_END -->"


def generate_results_markdown() -> str:
    """Tạo nội dung kết quả README từ metrics.json hiện tại."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    dataset = metrics["dataset"]
    nested = metrics["nested_cv_subject"]
    deployment = metrics["deployment_oof"]
    protocol = metrics["evaluation_protocol"]
    selection = metrics["selection"]
    return "\n".join(
        [
            "### Kết quả nested subject-level evaluation",
            "",
            (
                f"- **Protocol:** `{protocol['outer_folds']} outer × "
                f"{protocol['inner_folds']} inner`, unit=`{protocol['unit']}`"
            ),
            f"- **Primary metric:** `{protocol['primary_metric']}`",
            (
                f"- **Cross-fitted subjects:** `{dataset['n_subjects']}`; "
                "mỗi subject xuất hiện đúng một lần."
            ),
            f"- **Pooled Balanced Accuracy:** `{nested['Balanced Accuracy']:.4f}`",
            f"- **Pooled Macro-F1:** `{nested['F1-macro']:.4f}`",
            f"- **Pooled ROC-AUC:** `{nested['ROC-AUC']:.4f}`",
            "",
            "### Deployment contract",
            "",
            "- `StandardScaler → L2 Logistic Regression` với `aggregation=median`.",
            (f"- `C={selection['C']}`, `class_weight={selection['class_weight']}`."),
            f"- Full-data group-OOF threshold: `{deployment['decision_threshold']}`.",
            (
                f"- Artifact được fit trên toàn bộ `{dataset['n_subjects']}` subject "
                "sau khi protocol khóa; chưa có external cohort."
            ),
        ]
    )


def update_readme(*, check_only: bool = False) -> int:
    """Cập nhật hoặc kiểm tra vùng kết quả tự động trong README."""
    if not README_PATH.exists() or not METRICS_PATH.exists():
        print("Thiếu README.md hoặc artifacts/metrics.json.", file=sys.stderr)
        return 1

    content = README_PATH.read_text(encoding="utf-8")
    if START_MARKER not in content or END_MARKER not in content:
        print("README thiếu marker vùng kết quả tự động.", file=sys.stderr)
        return 1

    start = content.index(START_MARKER) + len(START_MARKER)
    end = content.index(END_MARKER)
    current = content[start:end].strip()
    generated = generate_results_markdown().strip()

    if check_only:
        if current != generated:
            print("README.md không đồng bộ với artifacts/metrics.json.", file=sys.stderr)
            return 1
        print("README.md results section is up to date.")
        return 0

    updated = content[:start] + "\n\n" + generated + "\n\n" + content[end:]
    README_PATH.write_text(updated, encoding="utf-8")
    print("Đã cập nhật vùng kết quả README.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đồng bộ kết quả README từ artifact.")
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không ghi file.")
    arguments = parser.parse_args()
    sys.exit(update_readme(check_only=arguments.check))
