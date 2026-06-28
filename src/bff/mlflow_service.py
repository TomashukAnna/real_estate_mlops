from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.bff.config import BffSettings


def _as_iso(value: Optional[int]) -> Optional[str]:
    if not value:
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


class MlflowSummaryService:
    def __init__(self, settings: BffSettings) -> None:
        self.settings = settings

    def _load_project_params(self) -> Dict[str, Any]:
        if not self.settings.params_path.exists():
            return {}
        return yaml.safe_load(self.settings.params_path.read_text(encoding="utf-8"))

    def _load_registry_fallback(self) -> Dict[str, Any]:
        if not self.settings.registry_result_path.exists():
            return {}
        return json.loads(
            self.settings.registry_result_path.read_text(encoding="utf-8")
        )

    def _client(self) -> MlflowClient:
        return MlflowClient(tracking_uri=self.settings.mlflow_tracking_uri)

    def summary(self, limit: int = 20) -> Dict[str, Any]:
        params = self._load_project_params()
        experiment_name = (
            params.get("mlflow", {}).get("experiment_name")
            if isinstance(params.get("mlflow"), dict)
            else None
        )
        registry_name = (
            params.get("mlflow", {})
            .get("registry", {})
            .get("model_name")
            if isinstance(params.get("mlflow"), dict)
            else None
        )
        registry_alias = (
            params.get("mlflow", {})
            .get("registry", {})
            .get("alias")
            if isinstance(params.get("mlflow"), dict)
            else None
        )
        try:
            client = self._client()
            experiment = (
                client.get_experiment_by_name(experiment_name)
                if experiment_name
                else None
            )
            runs: List[Dict[str, Any]] = []
            if experiment is not None:
                search_results = client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["attribute.start_time DESC"],
                    max_results=limit,
                )
                for run in search_results:
                    start_time = _as_iso(run.info.start_time)
                    end_time = _as_iso(run.info.end_time)
                    duration = None
                    if run.info.start_time and run.info.end_time:
                        duration = round(
                            (run.info.end_time - run.info.start_time) / 1000, 2
                        )
                    runs.append(
                        {
                            "run_id": run.info.run_id,
                            "name": run.data.tags.get("mlflow.runName", run.info.run_id),
                            "status": run.info.status,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration_seconds": duration,
                            "metrics": {
                                key: float(value)
                                for key, value in run.data.metrics.items()
                            },
                            "params": {
                                key: str(value)
                                for key, value in run.data.params.items()
                            },
                            "tags": {
                                key: str(value) for key, value in run.data.tags.items()
                            },
                        }
                    )

            registry_summary = {
                "available": False,
                "model_name": registry_name,
                "alias": registry_alias,
                "alias_version": None,
                "stage": None,
                "tracking_uri": self.settings.mlflow_tracking_uri,
                "error": None,
            }
            if registry_name:
                try:
                    if registry_alias:
                        version = client.get_model_version_by_alias(
                            registry_name,
                            registry_alias,
                        )
                        registry_summary.update(
                            {
                                "available": True,
                                "alias_version": str(version.version),
                                "stage": version.current_stage,
                            }
                        )
                    else:
                        model = client.get_registered_model(registry_name)
                        registry_summary.update(
                            {
                                "available": True,
                                "alias_version": None,
                                "stage": None,
                                "model_name": model.name,
                            }
                        )
                except MlflowException:
                    fallback = self._load_registry_fallback()
                    if fallback:
                        registry_summary.update(
                            {
                                "available": True,
                                "model_name": fallback.get(
                                    "registered_model_name",
                                    registry_name,
                                ),
                                "alias": fallback.get("alias", registry_alias),
                                "alias_version": fallback.get("version"),
                                "stage": fallback.get("stage"),
                                "tracking_uri": fallback.get(
                                    "tracking_uri",
                                    self.settings.mlflow_tracking_uri,
                                ),
                            }
                        )

            return {
                "available": True,
                "tracking_uri": self.settings.mlflow_tracking_uri,
                "experiment_name": experiment_name,
                "experiment_id": (
                    experiment.experiment_id if experiment is not None else None
                ),
                "runs": runs,
                "registry": registry_summary,
                "error": None,
            }
        except Exception as exc:
            fallback = self._load_registry_fallback()
            return {
                "available": False,
                "tracking_uri": self.settings.mlflow_tracking_uri,
                "experiment_name": experiment_name,
                "experiment_id": None,
                "runs": [],
                "registry": {
                    "available": bool(fallback),
                    "model_name": fallback.get("registered_model_name")
                    if fallback
                    else registry_name,
                    "alias": fallback.get("alias") if fallback else registry_alias,
                    "alias_version": fallback.get("version") if fallback else None,
                    "stage": fallback.get("stage") if fallback else None,
                    "tracking_uri": fallback.get("tracking_uri")
                    if fallback
                    else self.settings.mlflow_tracking_uri,
                    "error": str(exc),
                },
                "error": str(exc),
            }
