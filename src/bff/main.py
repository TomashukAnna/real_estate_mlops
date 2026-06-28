from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.bff.config import BffSettings
from src.bff.dataset_service import DatasetService
from src.bff.mlflow_service import MlflowSummaryService
from src.bff.retrain_service import RetrainManager
from src.bff.schemas import (
    DriftSummaryResponse,
    ExperimentsSummaryResponse,
    MapSampleResponse,
    MapSelectionRequest,
    MarkerPoint,
    RecentPrediction,
    RetrainStatusResponse,
)
from src.monitoring.drift_core import (
    calculate_concept_drift,
    calculate_feature_drift,
    calculate_prediction_drift,
    calculate_target_drift,
)

app = FastAPI(title="Real Estate UI BFF", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = BffSettings()
dataset_service = DatasetService(settings)
mlflow_service = MlflowSummaryService(settings)
retrain_manager = RetrainManager(settings)


def _metric_status(score: float, warning_threshold: float) -> str:
    if score <= 0:
        return "ok"
    if score >= warning_threshold:
        return "warning"
    return "ok"


def _empty_drift_summary() -> Dict[str, Any]:
    return {
        "available": False,
        "generated_at": None,
        "model_version": None,
        "observation_count": 0,
        "labeled_observation_count": 0,
        "data_drift": {
            "available": False,
            "score": 0.0,
            "status": "unknown"
        },
        "prediction_drift": {
            "available": False,
            "score": 0.0,
            "status": "unknown",
        },
        "target_drift": {
            "available": False,
            "score": 0.0,
            "status": "unknown"
        },
        "concept_drift": {
            "available": False,
            "score": 0.0,
            "status": "unknown",
        },
        "drifted_features": [],
    }


def _drift_summary_from_live_observations() -> Dict[str, Any] | None:
    metadata_path = settings.model_metadata_path
    log_path = settings.inference_log_path
    if not metadata_path.exists() or not log_path.exists():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference_profile = metadata.get("reference_profile")
    if not reference_profile:
        return None

    observations = pd.read_json(log_path, lines=True)
    if observations.empty:
        return None

    if "timestamp" in observations.columns:
        observations["timestamp"] = pd.to_datetime(
            observations["timestamp"],
            utc=True,
            errors="coerce",
        )
        observations = observations.sort_values("timestamp")
    if "actual_price_per_m2" in observations.columns:
        observations["actual_price_per_m2"] = pd.to_numeric(
            observations["actual_price_per_m2"],
            errors="coerce",
        )
    if "prediction" in observations.columns:
        observations["prediction"] = pd.to_numeric(
            observations["prediction"],
            errors="coerce",
        )

    feature_drift, data_score, _ = calculate_feature_drift(
        observations=observations,
        reference_profile=reference_profile,
    )
    prediction_drift = calculate_prediction_drift(
        predictions=observations.get("prediction", pd.Series(dtype=float)),
        reference_profile=reference_profile,
    )
    labeled = observations.dropna(subset=["actual_price_per_m2"]).copy()
    target_drift = calculate_target_drift(
        actual_target=labeled.get(
            "actual_price_per_m2",
            pd.Series(dtype=float)
        ),
        reference_profile=reference_profile,
    )
    concept_drift = calculate_concept_drift(
        labeled_observations=labeled,
        baseline_metrics=metadata.get("metrics", {}),
        actual_column="actual_price_per_m2",
        prediction_column="prediction",
    )
    prediction_score = (
        float(prediction_drift.get("score", 0.0))
        if prediction_drift.get("available", False)
        else 0.0
    )
    target_score = (
        float(target_drift.get("score", 0.0))
        if target_drift.get("available", False)
        else 0.0
    )
    concept_score = (
        float(concept_drift.get("score", 0.0))
        if concept_drift.get("available", False)
        else 0.0
    )
    drifted_features = sorted(
        feature
        for feature, details in feature_drift.items()
        if details.get("status") != "ok"
    )
    return {
        "available": True,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "model_version": metadata.get("model_version"),
        "observation_count": int(len(observations)),
        "labeled_observation_count": int(len(labeled)),
        "data_drift": {
            "available": True,
            "score": float(data_score),
            "status": _metric_status(
                float(data_score), settings.data_drift_warning
            ),
        },
        "prediction_drift": {
            "available": prediction_drift.get("available", False),
            "score": prediction_score,
            "status": _metric_status(
                prediction_score, settings.data_drift_warning
            ),
        },
        "target_drift": {
            "available": target_drift.get("available", False),
            "score": target_score,
            "status": _metric_status(
                target_score, settings.data_drift_warning
            ),
        },
        "concept_drift": {
            "available": concept_drift.get("available", False),
            "score": concept_score,
            "status": _metric_status(
                concept_score, settings.concept_drift_warning
            ),
        },
        "drifted_features": drifted_features,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "dataset_rows": dataset_service.total_rows(),
        "tracking_uri": settings.mlflow_tracking_uri,
        "inference_api_url": settings.inference_api_url,
    }


@app.get("/map/sample", response_model=MapSampleResponse)
def map_sample(
    size: int = Query(default=settings.map_sample_size, ge=1, le=500),
) -> Dict[str, Any]:
    return dataset_service.sample_points(size=size)


@app.post("/map/add-random", response_model=MarkerPoint)
def map_add_random(request_body: MapSelectionRequest) -> Dict[str, Any]:
    try:
        return dataset_service.add_random_point(request_body.excluded_ids)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/predictions/recent", response_model=list[RecentPrediction])
def recent_predictions(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[Dict[str, Any]]:
    return dataset_service.recent_predictions(limit=limit)


@app.get("/drift/summary", response_model=DriftSummaryResponse)
def drift_summary() -> Dict[str, Any]:
    live_report = _drift_summary_from_live_observations()
    if live_report is not None:
        return live_report

    path = settings.drift_report_path
    if not path.exists():
        return _empty_drift_summary()

    report = json.loads(path.read_text(encoding="utf-8"))
    data_score = float(report["data_drift"]["score"])
    prediction_score = float(report["prediction_drift"].get("score", 0.0))
    target_available = bool(report["target_drift"].get("available", False))
    target_score = (
        float(report["target_drift"].get("score", 0.0))
        if target_available
        else 0.0
    )
    concept_available = bool(report["concept_drift"].get("available", False))
    concept_score = (
        float(report["concept_drift"].get("score", 0.0))
        if concept_available
        else 0.0
    )
    return {
        "available": True,
        "generated_at": report.get("generated_at"),
        "model_version": report.get("model_version"),
        "observation_count": int(report["window"]["observation_count"]),
        "labeled_observation_count": int(
            report["window"]["labeled_observation_count"]
        ),
        "data_drift": {
            "available": True,
            "score": data_score,
            "status": _metric_status(
                data_score, settings.data_drift_warning
            ),
        },
        "prediction_drift": {
            "available": report["prediction_drift"].get("available", True),
            "score": prediction_score,
            "status": _metric_status(
                prediction_score, settings.data_drift_warning
            ),
        },
        "target_drift": {
            "available": target_available,
            "score": target_score,
            "status": _metric_status(
                target_score, settings.data_drift_warning
            ),
        },
        "concept_drift": {
            "available": concept_available,
            "score": concept_score,
            "status": _metric_status(
                concept_score,
                settings.concept_drift_warning,
            ),
        },
        "drifted_features": sorted(report["data_drift"]["features"].keys()),
    }


@app.get("/experiments/summary", response_model=ExperimentsSummaryResponse)
def experiments_summary(
    limit: int = Query(default=20, ge=1, le=100),
) -> Dict[str, Any]:
    return mlflow_service.summary(limit=limit)


@app.post("/retrain", response_model=RetrainStatusResponse)
def retrain() -> Dict[str, Any]:
    try:
        return retrain_manager.start()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status() -> Dict[str, Any]:
    return retrain_manager.status()
