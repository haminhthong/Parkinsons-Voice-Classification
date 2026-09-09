from parkinson_voice.audit import run_naive_split_audit
from parkinson_voice.utils import sha256_file


def test_data_checksum_is_stable(data_path):
    checksum = sha256_file(data_path)
    assert checksum == sha256_file(data_path)
    assert len(checksum) == 64


def test_naive_split_audit_is_reproducible(data_path, tmp_path):
    result = run_naive_split_audit(data_path=data_path, artifact_dir=tmp_path)

    assert result["split_unit"] == "record"
    assert result["overlapping_test_subjects"] > 0
    assert 0.0 <= result["accuracy"] <= 1.0
    assert (tmp_path / "naive_split_audit.json").is_file()
    assert (tmp_path / "naive_split_predictions.csv").is_file()
