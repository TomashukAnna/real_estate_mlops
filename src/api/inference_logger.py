from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

DEFAULT_INFERENCE_LOG_PATH = Path("reports/inference/predictions.jsonl")


class InferenceLogger:
    """Append-only storage for prediction events."""

    def __init__(self) -> None:
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return Path(
            os.getenv("INFERENCE_LOG_PATH", str(DEFAULT_INFERENCE_LOG_PATH))
        )

    def log_prediction(
        self,
        payload: Dict[str, Any],
        prediction: float,
        model_version: str,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "prediction": float(prediction),
            **payload,
        }
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
