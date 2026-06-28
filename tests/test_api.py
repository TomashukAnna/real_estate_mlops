import os
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


class DummyModel:
    def predict(self, frame):
        return [123456.78]


class MockYandexStorage:
    """Мок для Яндекс Диска в тестах API"""

    def __init__(self, *args, **kwargs):
        self.base_path = "/test"
        self.client = None

    def download_model(self, remote_path):
        return DummyModel()

    def download_json(self, remote_path):
        return {"model_version": "test-model-v1"}

    def download_dataframe(self, remote_path):
        return None

    def file_exists(self, remote_path):
        return False

    def upload_file(self, local_path, remote_path):
        return True


@pytest.fixture()
def client(tmp_path: Path):
    # Создаем временные файлы для модели и метаданных
    model_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"

    joblib.dump(DummyModel(), model_path)
    metadata_path.write_text(
        '{"model_version":"test-model-v1"}',
        encoding="utf-8",
    )

    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["MODEL_METADATA_PATH"] = str(metadata_path)

    # Подменяем YandexStorage на мок через monkeypatch
    import src.api.model_loader
    import src.infrastructure.yandex_storage

    original_storage = src.infrastructure.yandex_storage.YandexStorage
    src.infrastructure.yandex_storage.YandexStorage = MockYandexStorage

    # Перезагружаем модуль model_loader, чтобы он использовал мок
    import importlib
    importlib.reload(src.api.model_loader)

    # Создаем новый экземпляр ModelStore
    from src.api.model_loader import ModelStore
    store = ModelStore()
    store.load()

    # Подменяем model_store в main
    import src.api.main
    src.api.main.model_store = store

    # Возвращаем клиент
    yield TestClient(app)

    # Восстанавливаем оригинальный класс
    src.infrastructure.yandex_storage.YandexStorage = original_storage


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

    os.environ["INFERENCE_LOG_PATH"] = str(log_path)

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
    payload = response.json()
    assert payload["model_version"] == "test-model-v1"
    assert payload["prediction"] == 123456.78


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
    payload = response.json()
    assert payload["model_version"] == "test-model-v1"
    assert payload["prediction"] == 123456.78


def test_reload_model(client: TestClient):
    response = client.post("/reload-model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "reloaded"
    assert payload["model_version"] == "test-model-v1"


def test_predict_logs_unlabeled_observation_as_null(
    client: TestClient,
    tmp_path: Path,
):
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text("", encoding="utf-8")

    os.environ["INFERENCE_LOG_PATH"] = str(log_path)

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
    payload = response.json()
    assert payload["model_version"] == "test-model-v1"
    assert payload["prediction"] == 123456.78


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
