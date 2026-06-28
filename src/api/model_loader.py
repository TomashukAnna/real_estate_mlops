from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from src.infrastructure.yandex_storage import YandexStorage

DEFAULT_MODEL_PATH = "models/model.pkl"           # путь на Яндекс Диске
DEFAULT_METADATA_PATH = "metadata/metadata.json"  # путь на Яндекс Диске

# Загружаем значения по умолчанию из .env
load_dotenv()


@dataclass
class LoadedModel:
    model: Any
    metadata: Dict[str, Any]
    model_path: str  # теперь это путь на Яндекс Диске


class ModelStore:
    """Ленивый контейнер для модели и метаданных (загрузка с Яндекс Диска)."""

    def __init__(self) -> None:
        self._loaded: Optional[LoadedModel] = None
        self._error: Optional[str] = None
        self._storage: Optional[YandexStorage] = None

        # Инициализируем хранилище Яндекс Диска
        try:
            self._storage = YandexStorage()
        except ValueError as e:
            self._error = f"Yandex Disk initialization failed: {e}"
            print(f"{self._error}")

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
        if self._storage is None:
            self._error = "Yandex Disk storage is not initialized"
            return

        # Читаем пути к файлам на Яндекс Диске из переменных окружения
        model_remote_path = os.getenv("YANDEX_MODEL_PATH", DEFAULT_MODEL_PATH)
        metadata_remote_path = os.getenv(
            "YANDEX_METADATA_PATH", DEFAULT_METADATA_PATH)

        print(f"Loading model from Yandex Disk: {model_remote_path}")
        print(f"Loading metadata from Yandex Disk: {metadata_remote_path}")

        try:
            # 1. Загружаем модель с Яндекс Диска
            model = self._storage.download_model(model_remote_path)
            if model is None:
                self._error = (
                                f"Model not found on Yandex Disk: "
                                f"{model_remote_path}"
                            )
                print(f"{self._error}")
                return

            # 2. Загружаем метаданные с Яндекс Диска
            metadata = self._storage.download_json(metadata_remote_path)
            if metadata is None:
                print(
                    f"Loading metadata from Yandex Disk: "
                    f"{metadata_remote_path}")
                metadata = {"model_version": "unknown"}

        except Exception as exc:
            self._loaded = None
            self._error = str(exc)
            print(f"Failed to load model: {exc}")
            return

        # 3. Сохраняем в память
        self._loaded = LoadedModel(
            model=model,
            metadata=metadata,
            model_path=model_remote_path,
        )
        self._error = None
        print(f"Model loaded from Yandex Disk. Version: {self.version()}")

    def predict(self, payload: Dict[str, Any]) -> float:
        if not self._loaded:
            raise RuntimeError("Model is not loaded")
        frame = pd.DataFrame([payload])
        prediction = self._loaded.model.predict(frame)[0]
        return float(prediction)
