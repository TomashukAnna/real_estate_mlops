from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = {"region", "building_type", "object_type"}
EPSILON = 1e-6
DATA_DRIFT_WARNING = 0.1
DATA_DRIFT_CRITICAL = 0.25
CONCEPT_DRIFT_WARNING = 0.2
CONCEPT_DRIFT_CRITICAL = 0.5


@dataclass(frozen=True)
class FeatureDriftResult:
    score: float
    status: str
    current_count: int


def _normalize_counts(counts: Iterable[float]) -> List[float]:
    values = np.asarray(list(counts), dtype=float)
    total = values.sum()
    if total <= 0:
        return [0.0 for _ in values]
    return (values / total).tolist()


def _psi(expected: Iterable[float], actual: Iterable[float]) -> float:
    expected_arr = np.asarray(list(expected), dtype=float) + EPSILON
    actual_arr = np.asarray(list(actual), dtype=float) + EPSILON
    return float(
        np.sum(
            (actual_arr - expected_arr) * np.log(actual_arr / expected_arr)
        )
    )


def _drift_status(score: float) -> str:
    if score >= DATA_DRIFT_CRITICAL:
        return "critical"
    if score >= DATA_DRIFT_WARNING:
        return "warning"
    return "ok"


def _concept_status(score: float) -> str:
    if score >= CONCEPT_DRIFT_CRITICAL:
        return "critical"
    if score >= CONCEPT_DRIFT_WARNING:
        return "warning"
    return "ok"


def _prepare_numeric_edges(values: pd.Series, bins: int = 10) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return np.asarray([0.0, 1.0])

    edges = np.quantile(clean.to_numpy(), np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) == 1:
        center = float(edges[0])
        return np.asarray([center - 0.5, center + 0.5])
    return edges


def _build_numeric_reference(values: pd.Series) -> Dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    edges = _prepare_numeric_edges(clean)
    hist, _ = np.histogram(clean.to_numpy(), bins=edges)
    return {
        "type": "numeric",
        "bin_edges": [float(edge) for edge in edges],
        "reference_distribution": _normalize_counts(hist),
        "summary": {
            "count": int(clean.shape[0]),
            "mean": float(clean.mean()) if not clean.empty else 0.0,
            "std": float(clean.std(ddof=0)) if not clean.empty else 0.0,
            "min": float(clean.min()) if not clean.empty else 0.0,
            "max": float(clean.max()) if not clean.empty else 0.0,
        },
    }


def _build_categorical_reference(values: pd.Series) -> Dict[str, Any]:
    clean = values.dropna().astype(str)
    counts = clean.value_counts()
    return {
        "type": "categorical",
        "categories": counts.index.tolist(),
        "reference_distribution": _normalize_counts(counts.to_list()),
        "summary": {
            "count": int(clean.shape[0]),
            "unique": int(clean.nunique()),
            "top": counts.index[0] if not counts.empty else None,
        },
    }


def build_reference_profile(
    x_reference: pd.DataFrame,
    y_reference: pd.Series,
    y_prediction: np.ndarray,
    baseline_metrics: Dict[str, float],
) -> Dict[str, Any]:
    features: Dict[str, Any] = {}
    for column in x_reference.columns:
        series = x_reference[column]
        if column in CATEGORICAL_FEATURES:
            features[column] = _build_categorical_reference(series)
        else:
            features[column] = _build_numeric_reference(series)

    return {
        "features": features,
        "target": _build_numeric_reference(y_reference),
        "prediction": _build_numeric_reference(pd.Series(y_prediction)),
        "baseline_metrics": baseline_metrics,
        "feature_types": {
            feature: (
                "categorical"
                if feature in CATEGORICAL_FEATURES
                else "numeric"
            )
            for feature in x_reference.columns
        },
    }


def _numeric_drift(
    values: pd.Series,
    reference: Dict[str, Any],
) -> FeatureDriftResult:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    edges = np.asarray(reference["bin_edges"], dtype=float)
    hist, _ = np.histogram(clean.to_numpy(), bins=edges)
    score = _psi(reference["reference_distribution"], _normalize_counts(hist))
    return FeatureDriftResult(
        score=score,
        status=_drift_status(score),
        current_count=int(clean.shape[0]),
    )


def _categorical_drift(
    values: pd.Series,
    reference: Dict[str, Any],
) -> FeatureDriftResult:
    clean = values.dropna().astype(str)
    categories = list(reference["categories"])
    current_counts = clean.value_counts()
    actual = [
        float(current_counts.get(category, 0.0))
        for category in categories
    ]
    unseen_count = max(int(clean.shape[0] - sum(actual)), 0)
    expected = list(reference["reference_distribution"])
    if unseen_count > 0:
        categories = [*categories, "__OTHER__"]
        expected.append(0.0)
        actual.append(float(unseen_count))
    score = _psi(expected, _normalize_counts(actual))
    return FeatureDriftResult(
        score=score,
        status=_drift_status(score),
        current_count=int(clean.shape[0]),
    )


def calculate_feature_drift(
    observations: pd.DataFrame,
    reference_profile: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], float, int]:
    feature_results: Dict[str, Dict[str, Any]] = {}
    scores: List[float] = []
    drifted_features = 0

    for feature, reference in reference_profile["features"].items():
        if feature not in observations.columns:
            continue
        if reference["type"] == "categorical":
            result = _categorical_drift(observations[feature], reference)
        else:
            result = _numeric_drift(observations[feature], reference)
        scores.append(result.score)
        if result.status != "ok":
            drifted_features += 1
        feature_results[feature] = {
            "score": result.score,
            "status": result.status,
            "current_count": result.current_count,
        }

    overall_score = float(np.mean(scores)) if scores else 0.0
    return feature_results, overall_score, drifted_features


def calculate_target_drift(
    actual_target: pd.Series,
    reference_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if actual_target.dropna().empty:
        return {"available": False}
    result = _numeric_drift(actual_target, reference_profile["target"])
    return {
        "available": True,
        "score": result.score,
        "status": result.status,
        "count": result.current_count,
    }


def calculate_prediction_drift(
    predictions: pd.Series,
    reference_profile: Dict[str, Any],
) -> Dict[str, Any]:
    if predictions.dropna().empty:
        return {"available": False}
    result = _numeric_drift(predictions, reference_profile["prediction"])
    return {
        "available": True,
        "score": result.score,
        "status": result.status,
        "count": result.current_count,
    }


def calculate_concept_drift(
    labeled_observations: pd.DataFrame,
    baseline_metrics: Dict[str, float],
    actual_column: str,
    prediction_column: str,
) -> Dict[str, Any]:
    if labeled_observations.empty:
        return {"available": False}

    y_true = pd.to_numeric(
        labeled_observations[actual_column], errors="coerce"
    ).dropna().astype(float)
    y_pred = pd.to_numeric(
        labeled_observations.loc[y_true.index, prediction_column],
        errors="coerce",
    ).dropna().astype(float)
    aligned_index = y_true.index.intersection(y_pred.index)
    if aligned_index.empty:
        return {"available": False}

    y_true = y_true.loc[aligned_index]
    y_pred = y_pred.loc[aligned_index]
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = np.clip(np.abs(y_true.to_numpy()), EPSILON, None)
    mape = float(np.mean(np.abs(errors) / denominator))
    ss_res = float(np.sum(np.square(errors)))
    ss_tot = float(np.sum(np.square(y_true - y_true.mean())))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    baseline_mae = max(float(baseline_metrics.get("mae", EPSILON)), EPSILON)
    baseline_rmse = max(float(baseline_metrics.get("rmse", EPSILON)), EPSILON)
    baseline_mape = max(float(baseline_metrics.get("mape", EPSILON)), EPSILON)

    relative_mae = max((mae - baseline_mae) / baseline_mae, 0.0)
    relative_rmse = max((rmse - baseline_rmse) / baseline_rmse, 0.0)
    relative_mape = max((mape - baseline_mape) / baseline_mape, 0.0)
    drift_score = max(relative_mae, relative_rmse, relative_mape)

    return {
        "available": True,
        "score": drift_score,
        "status": _concept_status(drift_score),
        "count": int(len(aligned_index)),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
        },
        "baseline_metrics": {
            "mae": baseline_mae,
            "rmse": baseline_rmse,
            "mape": baseline_mape,
            "r2": float(baseline_metrics.get("r2", 0.0)),
        },
    }
