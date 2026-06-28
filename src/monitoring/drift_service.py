from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from prometheus_client import Gauge, start_http_server

from src.monitoring.drift_core import (
    calculate_concept_drift,
    calculate_feature_drift,
    calculate_prediction_drift,
    calculate_target_drift,
)

DEFAULT_METADATA_PATH = Path("models/model/metadata.json")
DEFAULT_INFERENCE_LOG_PATH = Path("reports/inference/predictions.jsonl")
DEFAULT_DRIFT_REPORT_PATH = Path("reports/drift/latest_drift_report.json")
DEFAULT_DRIFT_HTML_PATH = Path("reports/drift/latest_drift_report.html")

SERVICE_UP = Gauge(
    "real_estate_drift_service_up",
    "Whether the drift calculation service is healthy.",
)
REPORT_GENERATED_AT = Gauge(
    "real_estate_drift_report_generated_at",
    "Unix timestamp of the latest successful drift report generation.",
)
WINDOW_SIZE = Gauge(
    "real_estate_drift_window_observations",
    "Number of recent observations used in the latest drift calculation.",
)
LABELED_WINDOW_SIZE = Gauge(
    "real_estate_drift_window_labeled_observations",
    "Number of labeled observations used in the latest drift calculation.",
)
DATA_DRIFT_SCORE = Gauge(
    "real_estate_data_drift_score",
    "Average PSI score across all monitored input features.",
)
DATA_DRIFTED_FEATURES = Gauge(
    "real_estate_data_drifted_features_total",
    "Number of features whose current PSI is above the warning threshold.",
)
PREDICTION_DRIFT_SCORE = Gauge(
    "real_estate_prediction_drift_score",
    "PSI score for prediction distribution drift.",
)
TARGET_DRIFT_SCORE = Gauge(
    "real_estate_target_drift_score",
    "PSI score for factual target distribution drift.",
)
CONCEPT_DRIFT_SCORE = Gauge(
    "real_estate_concept_drift_score",
    "Relative degradation score derived from labeled regression metrics.",
)
CONCEPT_DRIFT_MAE = Gauge(
    "real_estate_concept_drift_mae",
    "MAE on the most recent labeled observation window.",
)
CONCEPT_DRIFT_RMSE = Gauge(
    "real_estate_concept_drift_rmse",
    "RMSE on the most recent labeled observation window.",
)
CONCEPT_DRIFT_R2 = Gauge(
    "real_estate_concept_drift_r2",
    "R2 on the most recent labeled observation window.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodically calculate data, target and concept drift."
    )
    parser.add_argument(
        "--metadata-path",
        default=os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH)),
    )
    parser.add_argument(
        "--inference-log-path",
        default=os.getenv(
            "INFERENCE_LOG_PATH",
            str(DEFAULT_INFERENCE_LOG_PATH),
        ),
    )
    parser.add_argument(
        "--report-path",
        default=os.getenv("DRIFT_REPORT_PATH", str(DEFAULT_DRIFT_REPORT_PATH)),
    )
    parser.add_argument(
        "--html-report-path",
        default=os.getenv("DRIFT_HTML_PATH", str(DEFAULT_DRIFT_HTML_PATH)),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=int(os.getenv("DRIFT_WINDOW_SIZE", "500")),
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=int(os.getenv("DRIFT_INTERVAL_SECONDS", "60")),
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.getenv("DRIFT_METRICS_PORT", "8001")),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single calculation and exit.",
    )
    return parser.parse_args()


def load_reference_profile(metadata_path: Path) -> Dict[str, Any]:
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata was not found at '{metadata_path}'."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    reference_profile = metadata.get("reference_profile")
    if not reference_profile:
        raise ValueError(
            "Model metadata does not contain 'reference_profile'. "
            "Retrain the model to generate drift baselines."
        )
    return {
        "model_version": metadata.get("model_version", "unknown"),
        "reference_profile": reference_profile,
        "baseline_metrics": metadata.get("metrics", {}),
    }


def load_observations(log_path: Path, window_size: int) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame()
    frame = pd.read_json(log_path, lines=True)
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp")
    if window_size > 0:
        frame = frame.tail(window_size)
    if "actual_price_per_m2" in frame.columns:
        frame["actual_price_per_m2"] = pd.to_numeric(
            frame["actual_price_per_m2"], errors="coerce"
        )
    if "prediction" in frame.columns:
        frame["prediction"] = pd.to_numeric(
            frame["prediction"],
            errors="coerce",
        )
    return frame


def build_report(
    metadata_info: Dict[str, Any],
    observations: pd.DataFrame,
) -> Dict[str, Any]:
    reference_profile = metadata_info["reference_profile"]
    feature_drift, data_score, drifted_features = calculate_feature_drift(
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
            pd.Series(dtype=float),
        ),
        reference_profile=reference_profile,
    )
    concept_drift = calculate_concept_drift(
        labeled_observations=labeled,
        baseline_metrics=metadata_info["baseline_metrics"],
        actual_column="actual_price_per_m2",
        prediction_column="prediction",
    )
    if "timestamp" in observations.columns and not observations.empty:
        window_start = observations["timestamp"].min().isoformat()
        window_end = observations["timestamp"].max().isoformat()
    else:
        window_start = None
        window_end = None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata_info["model_version"],
        "window": {
            "observation_count": int(len(observations)),
            "labeled_observation_count": int(len(labeled)),
            "window_start": window_start,
            "window_end": window_end,
        },
        "data_drift": {
            "score": data_score,
            "drifted_features": drifted_features,
            "features": feature_drift,
        },
        "prediction_drift": prediction_drift,
        "target_drift": target_drift,
        "concept_drift": concept_drift,
    }


def update_metrics(report: Dict[str, Any]) -> None:
    window = report["window"]
    WINDOW_SIZE.set(window["observation_count"])
    LABELED_WINDOW_SIZE.set(window["labeled_observation_count"])
    DATA_DRIFT_SCORE.set(float(report["data_drift"]["score"]))
    DATA_DRIFTED_FEATURES.set(int(report["data_drift"]["drifted_features"]))
    PREDICTION_DRIFT_SCORE.set(
        float(report["prediction_drift"].get("score", 0.0))
        if report["prediction_drift"].get("available", True)
        else 0.0
    )
    TARGET_DRIFT_SCORE.set(
        float(report["target_drift"].get("score", 0.0))
        if report["target_drift"].get("available", False)
        else 0.0
    )
    if report["concept_drift"].get("available"):
        CONCEPT_DRIFT_SCORE.set(float(report["concept_drift"]["score"]))
        metrics = report["concept_drift"]["metrics"]
        CONCEPT_DRIFT_MAE.set(float(metrics["mae"]))
        CONCEPT_DRIFT_RMSE.set(float(metrics["rmse"]))
        CONCEPT_DRIFT_R2.set(float(metrics["r2"]))
    else:
        CONCEPT_DRIFT_SCORE.set(0.0)
        CONCEPT_DRIFT_MAE.set(0.0)
        CONCEPT_DRIFT_RMSE.set(0.0)
        CONCEPT_DRIFT_R2.set(0.0)
    REPORT_GENERATED_AT.set(time.time())


def write_report(
    report: Dict[str, Any],
    report_path: Path,
    html_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    feature_rows = "".join(
        (
            "<tr>"
            f"<td>{feature}</td>"
            f"<td>{details['status']}</td>"
            f"<td>{details['score']:.4f}</td>"
            f"<td>{details['current_count']}</td>"
            "</tr>"
        )
        for feature, details in report["data_drift"]["features"].items()
    )
    window_summary = (
        f"{report['window']['observation_count']} observations, "
        f"{report['window']['labeled_observation_count']} labeled"
    )
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Real Estate Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
  </style>
</head>
<body>
  <h1>Real Estate Drift Report</h1>
  <p><strong>Generated at:</strong> {report['generated_at']}</p>
  <p><strong>Model version:</strong> {report['model_version']}</p>
  <p><strong>Window:</strong> {window_summary}</p>
  <h2>Summary</h2>
  <ul>
    <li>Data drift score: {report['data_drift']['score']:.4f}</li>
    <li>Prediction drift score:
    {report['prediction_drift'].get('score', 0.0):.4f}</li>
    <li>Target drift score: {report['target_drift'].get('score', 0.0):.4f}</li>
    <li>Concept drift score:
    {report['concept_drift'].get('score', 0.0):.4f}</li>
  </ul>
  <h2>Feature drift</h2>
  <table>
    <thead>
      <tr><th>Feature</th><th>Status</th><th>Score</th><th>Current
      count</th></tr>
    </thead>
    <tbody>
      {feature_rows}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def calculate_once(args: argparse.Namespace) -> Dict[str, Any]:
    metadata_info = load_reference_profile(Path(args.metadata_path))
    observations = load_observations(
        Path(args.inference_log_path),
        window_size=args.window_size,
    )
    if observations.empty:
        raise ValueError(
            "No inference observations found. Send requests to /predict first."
        )
    report = build_report(
        metadata_info=metadata_info,
        observations=observations,
    )
    write_report(
        report=report,
        report_path=Path(args.report_path),
        html_path=Path(args.html_report_path),
    )
    update_metrics(report)
    return report


def run_service(args: argparse.Namespace) -> None:
    start_http_server(args.metrics_port)
    while True:
        try:
            calculate_once(args)
            SERVICE_UP.set(1)
        except Exception:
            SERVICE_UP.set(0)
        time.sleep(args.interval_seconds)


def main() -> None:
    args = parse_args()
    if args.once:
        calculate_once(args)
        SERVICE_UP.set(1)
        return
    run_service(args)


if __name__ == "__main__":
    main()
