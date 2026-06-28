from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

DEFAULT_FEATURES = [
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
]


@dataclass(frozen=True)
class BffSettings:
    dataset_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("BFF_DATASET_PATH", "data/processed/cleaned_data.csv")
        )
    )
    inference_log_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "INFERENCE_LOG_PATH",
                "reports/inference/predictions.jsonl")))
    drift_report_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DRIFT_REPORT_PATH",
                "reports/drift/latest_drift_report.json")))
    params_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "BFF_PARAMS_PATH",
                "params.yaml")))
    registry_result_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "BFF_REGISTRY_RESULT_PATH",
                "reports/registry_result.json")))
    model_metadata_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("MODEL_METADATA_PATH", "models/model/metadata.json")
        )
    )
    inference_api_url: str = field(
        default_factory=lambda: os.getenv(
            "BFF_INFERENCE_API_URL",
            "http://api:8000",
        ).rstrip("/")
    )
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://host.docker.internal:5000",
        )
    )
    map_sample_size: int = field(
        default_factory=lambda: int(os.getenv("BFF_MAP_SAMPLE_SIZE", "120"))
    )
    high_error_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("BFF_HIGH_ERROR_THRESHOLD", "0.15")
        )
    )
    data_drift_warning: float = field(
        default_factory=lambda: float(
            os.getenv(
                "BFF_DATA_DRIFT_WARNING",
                "0.2")))
    concept_drift_warning: float = field(
        default_factory=lambda: float(
            os.getenv("BFF_CONCEPT_DRIFT_WARNING", "0.25")
        )
    )

    @property
    def predict_url(self) -> str:
        return f"{self.inference_api_url}/predict"

    @property
    def reload_url(self) -> str:
        return f"{self.inference_api_url}/reload-model"


def feature_columns(metadata_path: Path) -> List[str]:
    if not metadata_path.exists():
        return DEFAULT_FEATURES.copy()
    try:
        import json

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_FEATURES.copy()
    features = payload.get("features")
    if isinstance(features, list) and features:
        return [str(item) for item in features]
    return DEFAULT_FEATURES.copy()
