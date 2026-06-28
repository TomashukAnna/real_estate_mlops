import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src.monitoring.drift_core import build_reference_profile
from src.monitoring.drift_service import calculate_once


def test_calculate_once_generates_report(tmp_path: Path):
    x_reference = pd.DataFrame(
        {
            "region": [2661, 2661, 3446, 3446],
            "building_type": [1, 1, 2, 2],
            "level": [2, 4, 6, 8],
            "levels": [5, 9, 12, 16],
            "year": [2023, 2023, 2024, 2024],
            "month": [1, 2, 3, 4],
            "rooms": [1, 2, 2, 3],
            "area": [35.0, 48.0, 60.0, 75.0],
            "kitchen_area": [8.0, 10.0, 12.0, 14.0],
            "object_type": [1, 1, 11, 11],
            "weekday_number": [0, 1, 2, 3],
        }
    )
    y_reference = pd.Series([100000.0, 120000.0, 140000.0, 160000.0])
    y_pred_reference = pd.Series([102000.0, 118000.0, 138000.0, 159000.0])
    metadata = {
        "model_version": "test-model-v1",
        "metrics": {
            "mae": 2000.0,
            "rmse": 2200.0,
            "mape": 0.02,
            "r2": 0.95,
        },
        "reference_profile": build_reference_profile(
            x_reference=x_reference,
            y_reference=y_reference,
            y_prediction=y_pred_reference.to_numpy(),
            baseline_metrics={
                "mae": 2000.0,
                "rmse": 2200.0,
                "mape": 0.02,
                "r2": 0.95,
            },
        ),
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-12T10:00:00+00:00",
                        "model_version": "test-model-v1",
                        "region": 2661,
                        "building_type": 1,
                        "level": 3,
                        "levels": 9,
                        "year": 2025,
                        "month": 4,
                        "rooms": 2,
                        "area": 49.0,
                        "kitchen_area": 9.5,
                        "object_type": 1,
                        "weekday_number": 2,
                        "prediction": 121000.0,
                        "actual_price_per_m2": 123000.0,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-12T11:00:00+00:00",
                        "model_version": "test-model-v1",
                        "region": 3446,
                        "building_type": 2,
                        "level": 7,
                        "levels": 16,
                        "year": 2025,
                        "month": 4,
                        "rooms": 3,
                        "area": 74.0,
                        "kitchen_area": 13.5,
                        "object_type": 11,
                        "weekday_number": 5,
                        "prediction": 158000.0,
                        "actual_price_per_m2": 157500.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    args = Namespace(
        metadata_path=str(metadata_path),
        inference_log_path=str(log_path),
        report_path=str(tmp_path / "latest.json"),
        html_report_path=str(tmp_path / "latest.html"),
        window_size=100,
    )

    report = calculate_once(args)

    assert report["window"]["observation_count"] == 2
    assert report["target_drift"]["available"] is True
    assert report["concept_drift"]["available"] is True
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.html").exists()
