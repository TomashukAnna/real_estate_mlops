import importlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def _prepare_files(tmp_path: Path) -> None:
    dataset_path = tmp_path / "cleaned_data.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "region,building_type,level,levels,year,month,rooms,area,kitchen_area,object_type,weekday_number,geo_lat,geo_lon,price_per_m2",
                "2661,1,5,10,2025,4,2,52.4,9.8,1,2,59.939,30.315,145000.0",
                "3446,2,7,16,2025,4,3,74.0,13.5,11,5,60.050,30.350,157500.0",
            ]
        ),
        encoding="utf-8",
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "test-model-v1",
                "features": [
                    "region",
                    "building_type",
                    "level",
                    "levels",
                    "year",
                    "month",
                    "rooms",
                    "area",
                    "kitchen_area",
                    "object_type",
                    "weekday_number",
                ],
            }
        ),
        encoding="utf-8",
    )
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        "\n".join(
            [
                "mlflow:",
                "  experiment_name: test-experiment",
                "  registry:",
                "    model_name: test-model",
                "    alias: champion",
                "    stage: Staging",
                "model:",
                "  random_forest:",
                "    n_estimators: 10",
                "    max_depth: 4",
                "    random_state: 42",
            ]
        ),
        encoding="utf-8",
    )
    drift_path = tmp_path / "drift.json"
    drift_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-18T12:00:00+00:00",
                "model_version": "test-model-v1",
                "window": {
                    "observation_count": 2,
                    "labeled_observation_count": 1,
                },
                "data_drift": {
                    "score": 0.11,
                    "features": {"area": {"score": 0.11}},
                },
                "prediction_drift": {"available": True, "score": 0.05},
                "target_drift": {"available": True, "score": 0.02},
                "concept_drift": {"available": True, "score": 0.03},
            }
        ),
        encoding="utf-8",
    )
    os.environ["BFF_DATASET_PATH"] = str(dataset_path)
    os.environ["MODEL_METADATA_PATH"] = str(metadata_path)
    os.environ["BFF_PARAMS_PATH"] = str(params_path)
    os.environ["DRIFT_REPORT_PATH"] = str(drift_path)
    os.environ["INFERENCE_LOG_PATH"] = str(tmp_path / "predictions.jsonl")


def _fake_inference_predict(url: str, payload: dict) -> dict:
    assert payload.get("log_event") is False
    actual = float(payload["actual_price_per_m2"])
    return {
        "prediction": actual * 0.9,
        "model_version": "bff-test",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def test_map_sample_and_drift_summary(tmp_path: Path):
    _prepare_files(tmp_path)

    import src.bff.main as bff_main

    bff_main = importlib.reload(bff_main)
    client = TestClient(bff_main.app)

    with patch(
        "src.bff.dataset_service._read_json_response",
        side_effect=_fake_inference_predict,
    ):
        sample_response = client.get("/map/sample?size=2")
    assert sample_response.status_code == 200
    sample_payload = sample_response.json()
    assert sample_payload["total_available"] == 2
    assert len(sample_payload["points"]) == 2
    for point in sample_payload["points"]:
        assert point["marker_color"] == "#d14343"
        assert "underprediction" in point["flags"]

    drift_response = client.get("/drift/summary")
    assert drift_response.status_code == 200
    drift_payload = drift_response.json()
    assert drift_payload["available"] is True
    assert drift_payload["observation_count"] == 2
    assert drift_payload["data_drift"]["status"] == "ok"


def test_experiments_and_retrain_endpoints(tmp_path: Path):
    _prepare_files(tmp_path)

    import src.bff.main as bff_main

    bff_main = importlib.reload(bff_main)
    bff_main.mlflow_service.summary = lambda limit=20: {
        "available": True,
        "tracking_uri": "http://host.docker.internal:5000",
        "experiment_name": "test-experiment",
        "experiment_id": "1",
        "runs": [],
        "registry": {
            "available": True,
            "model_name": "test-model",
            "alias": "champion",
            "alias_version": "3",
            "stage": "Staging",
            "tracking_uri": "http://host.docker.internal:5000",
            "error": None,
        },
        "error": None,
    }
    bff_main.retrain_manager.start = lambda: {
        "status": "running",
        "started_at": "2026-04-18T12:00:00+00:00",
        "finished_at": None,
        "model_version": None,
        "error": None,
        "logs": "",
    }
    bff_main.retrain_manager.status = lambda: {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "model_version": None,
        "error": None,
        "logs": "",
    }
    client = TestClient(bff_main.app)

    experiments_response = client.get("/experiments/summary")
    assert experiments_response.status_code == 200
    assert experiments_response.json()["registry"]["alias_version"] == "3"

    retrain_response = client.post("/retrain")
    assert retrain_response.status_code == 200
    assert retrain_response.json()["status"] == "running"

    status_response = client.get("/retrain/status")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "idle"


def test_recent_predictions_backfill_actual_price_from_dataset(tmp_path: Path):
    _prepare_files(tmp_path)
    inference_log_path = Path(os.environ["INFERENCE_LOG_PATH"])
    inference_log_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-04-18T12:05:00+00:00",
                "model_version": "test-model-v1",
                "prediction": 150000.0,
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
                "actual_price_per_m2": None,
            }
        ),
        encoding="utf-8",
    )

    import src.bff.main as bff_main

    bff_main = importlib.reload(bff_main)
    client = TestClient(bff_main.app)

    response = client.get("/predictions/recent?limit=1")

    assert response.status_code == 200
    payload = response.json()[0]
    assert payload["actual_price_per_m2"] == 145000.0
    assert payload["listing_id"] == 0
    assert payload["lat"] == 59.939
    assert payload["lon"] == 30.315
    assert payload["marker_color"] == "#1f9d55"
    assert "overprediction" in payload["flags"]


def test_recent_predictions_skip_unknown_points(tmp_path: Path):
    _prepare_files(tmp_path)
    inference_log_path = Path(os.environ["INFERENCE_LOG_PATH"])
    inference_log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-18T12:05:00+00:00",
                        "model_version": "test-model-v1",
                        "prediction": 150000.0,
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
                        "actual_price_per_m2": None,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-18T12:06:00+00:00",
                        "model_version": "test-model-v1",
                        "prediction": 999999.0,
                        "region": 2661,
                        "building_type": 1,
                        "level": 35,
                        "levels": 37,
                        "year": 2025,
                        "month": 11,
                        "rooms": 7,
                        "area": 240.0,
                        "kitchen_area": 40.0,
                        "object_type": 1,
                        "weekday_number": 5,
                        "actual_price_per_m2": None,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    import src.bff.main as bff_main

    bff_main = importlib.reload(bff_main)
    client = TestClient(bff_main.app)

    response = client.get("/predictions/recent?limit=5")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["listing_id"] == 0
