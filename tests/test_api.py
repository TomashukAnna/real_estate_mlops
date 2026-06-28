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
    inference_log_path = tmp_path / "predictions.jsonl"

    joblib.dump(DummyModel(), model_path)
    metadata_path.write_text(
        '{"model_version":"test-model-v1"}',
        encoding="utf-8",
    )

    import os

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_METADATA_PATH"] = str(metadata_path)
    os.environ["INFERENCE_LOG_PATH"] = str(inference_log_path)
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


def test_predict_skip_logging(client: TestClient, tmp_path: Path):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text("", encoding="utf-8")
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
            "log_event": False,
        },
    )
    assert response.status_code == 200
    assert log_path.read_text(encoding="utf-8").strip() == ""


def test_predict_accepts_actual_price(client: TestClient):
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
            "actual_price_per_m2": 145000.0,
        },
    )

    assert response.status_code == 200


def test_reload_model(client: TestClient):
    response = client.post("/reload-model")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_ready"] is True
    assert payload["model_version"] == "test-model-v1"


def test_predict_logs_unlabeled_observation_as_null(
    client: TestClient,
    tmp_path: Path,
):
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
            "actual_price_per_m2": -1,
        },
    )

    assert response.status_code == 200
    body = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8")
    assert '"actual_price_per_m2": null' in body


def test_metrics_endpoint(client: TestClient):
    client.get("/health")
    client.post(
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
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "real_estate_api_requests_total" in body
    assert "real_estate_api_request_latency_seconds" in body
    assert "real_estate_api_predictions_total" in body
    assert "real_estate_api_model_ready" in body
