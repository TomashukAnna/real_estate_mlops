from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from dotenv import load_dotenv

DEFAULT_MODEL_PATH = Path("models/model/model.pkl")
DEFAULT_METADATA_PATH = Path("models/model/metadata.json")

# Загружаем значения по умолчанию из .env, если файл есть.
load_dotenv()


@dataclass
class LoadedModel:
    model: Any
    metadata: Dict[str, Any]
    model_path: Path


class ModelStore:
    """Ленивый контейнер для модели и метаданных."""

    def __init__(self) -> None:
        self._loaded: Optional[LoadedModel] = None
        self._error: Optional[str] = None

    @property
    def error(self) -> Optional[str]:
        return self._error

    def is_ready(self) -> bool:
        return self._loaded is not None

    def version(self) -> str:
        if not self._loaded:
            return "unavailable"
        return str(self._loaded.metadata.get("model_version", "unknown"))

    def load(self) -> None:
        model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        metadata_path = Path(
            os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH))
        )
        try:
            model = joblib.load(model_path)
            metadata = self._load_metadata(metadata_path)
        except Exception as exc:  # pragma: no cover - диагностика при старте
            self._loaded = None
            self._error = str(exc)
            return

        self._loaded = LoadedModel(
            model=model,
            metadata=metadata,
            model_path=model_path,
        )
        self._error = None

    def predict(self, payload: Dict[str, Any]) -> float:
        if not self._loaded:
            raise RuntimeError("Model is not loaded")
        frame = pd.DataFrame([payload])
        prediction = self._loaded.model.predict(frame)[0]
        return float(prediction)

    @staticmethod
    def _load_metadata(metadata_path: Path) -> Dict[str, Any]:
        if not metadata_path.exists():
            return {"model_version": "unknown"}
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
