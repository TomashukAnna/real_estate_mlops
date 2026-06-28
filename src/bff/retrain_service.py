from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Dict, Optional
from urllib import error, request

import yaml

from src.bff.config import BffSettings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_json(url: str) -> Dict[str, object]:
    http_request = request.Request(
        url=url,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class RetrainState:
    status: str = "idle"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    model_version: Optional[str] = None
    error: Optional[str] = None
    logs: str = ""


class RetrainManager:
    def __init__(self, settings: BffSettings) -> None:
        self.settings = settings
        self._lock = Lock()
        self._state = RetrainState()
        self._repo_root = Path(__file__).resolve().parents[2]

    def status(self) -> Dict[str, Optional[str]]:
        with self._lock:
            return {
                "status": self._state.status,
                "started_at": self._state.started_at,
                "finished_at": self._state.finished_at,
                "model_version": self._state.model_version,
                "error": self._state.error,
                "logs": self._state.logs,
            }

    def start(self) -> Dict[str, Optional[str]]:
        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("Retraining job is already running.")
            self._state = RetrainState(status="running", started_at=_utc_now())
        Thread(target=self._run, daemon=True).start()
        return self.status()

    def _load_params(self) -> Dict[str, object]:
        if not self.settings.params_path.exists():
            raise FileNotFoundError(
                f"params.yaml was not found at '{self.settings.params_path}'."
            )
        return yaml.safe_load(self.settings.params_path.read_text(encoding="utf-8"))

    def _run_command(self, command: list[str]) -> str:
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = self.settings.mlflow_tracking_uri
        env.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")
        completed = subprocess.run(
            command,
            cwd=self._repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        log = (
            f"$ {' '.join(command)}\n"
            f"{completed.stdout or ''}\n{completed.stderr or ''}"
        ).strip()
        if completed.returncode != 0:
            raise RuntimeError(
                f"Command exited with status {completed.returncode}:\n{log}"
            )
        return log

    def _run(self) -> None:
        accumulated_logs = ""
        try:
            params = self._load_params()
            mlflow_params = params["mlflow"]
            model_params = params["model"]["random_forest"]
            train_logs = self._run_command(
                [
                    sys.executable,
                    "src/models/train_dvc_model.py",
                    "--data-path",
                    "data/processed/cleaned_data.csv",
                    "--model-path",
                    "models/model/model.pkl",
                    "--metadata-path",
                    "models/model/metadata.json",
                    "--metrics-path",
                    "reports/train_metrics.json",
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                    "--experiment-name",
                    str(mlflow_params["experiment_name"]),
                    "--run-name",
                    str(mlflow_params["run_name"]),
                    "--n-estimators",
                    str(model_params["n_estimators"]),
                    "--max-depth",
                    str(model_params["max_depth"]),
                    "--random-state",
                    str(model_params["random_state"]),
                ]
            )
            accumulated_logs = train_logs
            register_logs = self._run_command(
                [
                    sys.executable,
                    "src/models/register_mlflow_model.py",
                    "--tracking-uri",
                    self.settings.mlflow_tracking_uri,
                    "--metadata-path",
                    "models/model/metadata.json",
                    "--registered-model-name",
                    str(mlflow_params["registry"]["model_name"]),
                    "--artifact-path",
                    "model",
                    "--alias",
                    str(mlflow_params["registry"]["alias"]),
                    "--stage",
                    str(mlflow_params["registry"]["stage"]),
                    "--output-path",
                    "reports/registry_result.json",
                ]
            )
            accumulated_logs = "\n\n".join([train_logs, register_logs])
            reload_response = _post_json(self.settings.reload_url)
            with self._lock:
                self._state.status = "succeeded"
                self._state.finished_at = _utc_now()
                self._state.model_version = str(
                    reload_response.get("model_version", "unknown")
                )
                self._state.logs = accumulated_logs[-8000:]
                self._state.error = None
        except Exception as exc:
            detail = f"{exc!s}\n\n{traceback.format_exc()}"
            with self._lock:
                self._state.status = "failed"
                self._state.finished_at = _utc_now()
                self._state.error = detail[:8000]
                self._state.logs = f"{accumulated_logs}\n\n{detail}".strip()[:8000]
