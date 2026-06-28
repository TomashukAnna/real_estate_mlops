from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MapSelectionRequest(BaseModel):
    excluded_ids: List[int] = Field(default_factory=list)


class MarkerPoint(BaseModel):
    listing_id: int
    lat: float
    lon: float
    marker_color: str
    apartment: Dict[str, Any]
    prediction: Optional[float] = None
    actual_price_per_m2: Optional[float] = None
    relative_error: Optional[float] = None
    flags: List[str] = Field(default_factory=list)
    added_at: Optional[str] = None


class MapSampleResponse(BaseModel):
    total_available: int
    points: List[MarkerPoint]


class RecentPrediction(BaseModel):
    timestamp: datetime
    model_version: str
    listing_id: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    marker_color: str
    flags: List[str] = Field(default_factory=list)
    prediction: float
    actual_price_per_m2: Optional[float] = None
    relative_error: Optional[float] = None
    apartment: Dict[str, Any]


class DriftMetric(BaseModel):
    available: bool
    score: float = 0.0
    status: str = "unknown"


class DriftSummaryResponse(BaseModel):
    available: bool
    generated_at: Optional[str] = None
    model_version: Optional[str] = None
    observation_count: int = 0
    labeled_observation_count: int = 0
    data_drift: DriftMetric
    prediction_drift: DriftMetric
    target_drift: DriftMetric
    concept_drift: DriftMetric
    drifted_features: List[str] = Field(default_factory=list)


class ExperimentRun(BaseModel):
    run_id: str
    name: str
    status: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, str] = Field(default_factory=dict)
    tags: Dict[str, str] = Field(default_factory=dict)


class RegistrySummary(BaseModel):
    available: bool
    model_name: Optional[str] = None
    alias: Optional[str] = None
    alias_version: Optional[str] = None
    stage: Optional[str] = None
    tracking_uri: Optional[str] = None
    error: Optional[str] = None


class ExperimentsSummaryResponse(BaseModel):
    available: bool
    tracking_uri: str
    experiment_name: Optional[str] = None
    experiment_id: Optional[str] = None
    runs: List[ExperimentRun] = Field(default_factory=list)
    registry: RegistrySummary
    error: Optional[str] = None


class RetrainStatusResponse(BaseModel):
    status: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    model_version: Optional[str] = None
    error: Optional[str] = None
    logs: str = ""
