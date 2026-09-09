from pathlib import Path

import pytest

from parkinson_voice.data import load_data


@pytest.fixture(scope="session")
def data_path() -> Path:
    return Path(__file__).parents[1] / "data" / "parkinsons.csv"


@pytest.fixture(scope="session")
def frame(data_path):
    return load_data(data_path)


@pytest.fixture(scope="session")
def artifact_path() -> Path:
    return Path(__file__).parents[1] / "artifacts" / "releases" / "v1.0.0" / "model.joblib"
