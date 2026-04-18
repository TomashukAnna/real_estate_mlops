from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, model_store


class DummyModel:
    def predict(self, frame):
        # Возвращаем детерминированное значение для smoke-тестов.
        return [123456.78]


@pytest.fixture()
def client(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump(DummyModel(), model_path)
    metadata_path.write_text(
        '{"model_version":"test-model-v1"}',
        encoding="utf-8",
    )

    import os

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_METADATA_PATH"] = str(metadata_path)
    model_store.load()
    return TestClient(app)


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_ready"] is True
    assert payload["model_version"] == "test-model-v1"


def test_predict(client: TestClient):
    response = client.post(
        "/predict",
        json={
            "region": 2661,
            "building_type": 1,
            "level": 5,
            "levels": 10,
            "year": 2025,
            "month": 4,
            "rooms": 2,
            "area": 52.4,
            "kitchen_area": 9.8,
            "object_type": 1,
            "weekday_number": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_version"] == "test-model-v1"
    assert payload["prediction"] == 123456.78
