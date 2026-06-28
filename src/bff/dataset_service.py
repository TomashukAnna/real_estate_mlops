from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import error, request

import pandas as pd

from src.bff.config import BffSettings, feature_columns
from src.infrastructure.yandex_storage import YandexStorage

MAP_COLUMNS = ["geo_lat", "geo_lon", "price_per_m2"]


def _read_json_response(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=30) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


class DatasetService:
    def __init__(self, settings: BffSettings) -> None:
        self.settings = settings
        self._feature_columns = feature_columns(settings.model_metadata_path)

        # Инициализируем хранилище Яндекс Диска
        try:
            self._storage = YandexStorage()
        except ValueError as e:
            print(f"❌ Yandex Disk initialization failed: {e}")
            self._storage = None

        # Загружаем датасет с Яндекс Диска
        self._dataset = self._load_dataset()
        self._actual_lookup = self._build_actual_lookup()
        self._row_lookup = self._build_row_lookup()

    @property
    def feature_names(self) -> List[str]:
        return self._feature_columns.copy()

    def total_rows(self) -> int:
        return int(len(self._dataset)) if self._dataset is not None else 0

    def _load_dataset_from_yandex(self) -> Optional[pd.DataFrame]:
        """Загружает датасет с Яндекс Диска."""
        if self._storage is None:
            print("❌ Yandex Storage not available")
            return None

        # Путь к датасету на Яндекс Диске
        remote_dataset_path = (
            self.settings.yandex_dataset_path
            if hasattr(self.settings, "yandex_dataset_path")
            else "data/processed/cleaned_data.csv"
        )

        print(f"Loading dataset from Yandex Disk:"
              f"{remote_dataset_path}")
        df = self._storage.download_dataframe(remote_dataset_path)

        if df is None:
            # Если обработанного датасета нет — пробуем загрузить сырой
            raw_remote_path = "data/raw/russia_real_estate.csv"
            print(
                f"Processed dataset not found, trying raw: {raw_remote_path}"
            )
            raw_df = self._storage.download_dataframe(raw_remote_path)
            if raw_df is not None:
                print("🔄 Processing raw dataset...")
                df = self._process_raw_data(raw_df)
                self._storage.upload_dataframe(df, remote_dataset_path)
            else:
                print(
                    f"Raw dataset not found on Yandex Disk: {raw_remote_path}"
                )
                return None

        return df

    def _load_dataset(self) -> pd.DataFrame:
        """Загружает датасет (с Яндекс Диска или локально как fallback)."""
        # Пытаемся загрузить с Яндекс Диска
        if self._storage is not None:
            df = self._load_dataset_from_yandex()
            if df is not None:
                use_columns = list(
                    dict.fromkeys(self._feature_columns + MAP_COLUMNS)
                )
                available_columns = [
                    col for col in use_columns
                    if col in df.columns
                ]
                if available_columns:
                    df = df[available_columns].copy()
                    df = df.dropna(subset=available_columns).copy()
                    df["listing_id"] = range(len(df))
                    print(
                        f"Dataset loaded from Yandex Disk: {len(df)} rows"
                    )
                    return df

        # Fallback: загружаем локально (если есть)
        dataset_path = self.settings.dataset_path
        if dataset_path.exists():
            print(f"Fallback: loading dataset from local path: {dataset_path}")
            use_columns = list(
                dict.fromkeys(self._feature_columns + MAP_COLUMNS)
            )
            frame = pd.read_csv(dataset_path, usecols=use_columns)
            frame = frame.dropna(
                subset=self._feature_columns + MAP_COLUMNS
            ).copy()
            frame["listing_id"] = range(len(frame))
            return frame

        raise FileNotFoundError(
            f"Dataset not found on Yandex Disk or locally at '{dataset_path}'."
        )

    def _process_raw_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Обработка сырых данных (аналог make_dataset.py)."""
        df = raw_df.copy()

        if "category" in df.columns:
            df = df[df["category"] == "flat sale"].copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["weekday_number"] = df["date"].dt.weekday

        if "price" in df.columns and "area" in df.columns:
            df["price_per_m2"] = df["price"] / df["area"]

        return df

    def _feature_signature(self, payload: Dict[str, Any]) -> tuple[Any, ...]:
        return tuple(payload.get(column) for column in self._feature_columns)

    def _build_actual_lookup(self) -> Dict[tuple[Any, ...], float]:
        if self._dataset is None:
            return {}
        grouped = (
            self._dataset.groupby(
                self._feature_columns, dropna=False
            )["price_per_m2"]
            .median()
            .reset_index()
        )
        return {
            self._feature_signature(record): float(record["price_per_m2"])
            for record in grouped.to_dict(orient="records")
        }

    def _build_row_lookup(self) -> Dict[tuple[Any, ...], Dict[str, Any]]:
        if self._dataset is None:
            return {}
        unique_rows = self._dataset.drop_duplicates(
            subset=self._feature_columns,
            keep="first",
        )
        lookup: Dict[tuple[Any, ...], Dict[str, Any]] = {}
        for record in unique_rows.to_dict(orient="records"):
            lookup[self._feature_signature(record)] = {
                "listing_id": int(record["listing_id"]),
                "lat": float(record["geo_lat"]),
                "lon": float(record["geo_lon"]),
            }
        return lookup

    def _resolve_actual_price(
        self, apartment: Dict[str, Any]
    ) -> Optional[float]:
        actual = _safe_float(apartment.get("actual_price_per_m2"))
        if actual is not None:
            return actual
        return self._actual_lookup.get(self._feature_signature(apartment))

    def _resolve_row(
        self, apartment: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return self._row_lookup.get(self._feature_signature(apartment))

    def _build_apartment_payload(self, row: pd.Series) -> Dict[str, Any]:
        payload = {
            "geo_lat": float(row["geo_lat"]),
            "geo_lon": float(row["geo_lon"]),
            "actual_price_per_m2": float(row["price_per_m2"]),
        }
        for column in self._feature_columns:
            value = row[column]
            payload[column] = value.item() if hasattr(value, "item") else value
        return payload

    def _call_predict(
        self,
        apartment: Dict[str, Any],
        *,
        log_event: bool,
    ) -> Dict[str, Any]:
        body = {
            key: apartment[key]
            for key in self._feature_columns + ["actual_price_per_m2"]
        }
        body["log_event"] = log_event
        return _read_json_response(self.settings.predict_url, body)

    def _predict_listing_for_sample(
        self,
        row: pd.Series,
    ) -> Tuple[int, Optional[float]]:
        apartment = self._build_apartment_payload(row)
        try:
            result = self._call_predict(apartment, log_event=False)
            return int(row["listing_id"]), float(result["prediction"])
        except (error.HTTPError,
                error.URLError,
                OSError,
                ValueError,
                KeyError):
            return int(row["listing_id"]), None

    def _compute_flags(
        self,
        prediction: Optional[float],
        actual: Optional[float],
    ) -> Dict[str, Any]:
        flags: List[str] = []
        marker_color = "slategray"
        relative_error = None
        if prediction is not None and actual is not None and actual > 0:
            relative_error = abs(prediction - actual) / actual
            if prediction < actual:
                marker_color = "#d14343"
                flags.append("underprediction")
            else:
                marker_color = "#1f9d55"
                flags.append("overprediction")
            if relative_error >= self.settings.high_error_threshold:
                flags.append("high_relative_error")
        return {
            "flags": flags,
            "marker_color": marker_color,
            "relative_error": relative_error,
        }

    def _row_to_marker(
        self,
        row: pd.Series,
        prediction: Optional[float] = None,
        actual: Optional[float] = None,
        added_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        computed = self._compute_flags(prediction, actual)
        apartment = self._build_apartment_payload(row)
        return {
            "listing_id": int(row["listing_id"]),
            "lat": float(row["geo_lat"]),
            "lon": float(row["geo_lon"]),
            "marker_color": computed["marker_color"],
            "apartment": apartment,
            "prediction": prediction,
            "actual_price_per_m2": actual,
            "relative_error": computed["relative_error"],
            "flags": computed["flags"],
            "added_at": added_at,
        }

    def sample_points(self, size: Optional[int] = None) -> Dict[str, Any]:
        sample_size = min(
            size or self.settings.map_sample_size,
            len(self._dataset)
        )
        sample = self._dataset.sample(n=sample_size).sort_values("listing_id")
        rows = [row for _, row in sample.iterrows()]
        predictions: Dict[int, Optional[float]] = {
            int(row["listing_id"]): None for row in rows
        }
        workers = min(8, max(1, len(rows)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._predict_listing_for_sample, row)
                for row in rows
            ]
            for future in as_completed(futures):
                listing_id, prediction = future.result()
                predictions[listing_id] = prediction
        points = []
        for row in rows:
            listing_id = int(row["listing_id"])
            actual = float(row["price_per_m2"])
            marker = self._row_to_marker(
                row,
                prediction=predictions.get(listing_id),
                actual=actual,
            )
            points.append(marker)
        return {
            "total_available": self.total_rows(),
            "points": points,
        }

    def add_random_point(self, excluded_ids: Iterable[int]) -> Dict[str, Any]:
        excluded = {int(value) for value in excluded_ids}
        available = self._dataset.loc[
            ~self._dataset["listing_id"].isin(excluded)
        ]
        if available.empty:
            raise ValueError("No more unseen apartments are available.")
        row = available.sample(n=1).iloc[0]
        apartment = self._build_apartment_payload(row)
        try:
            result = self._call_predict(apartment, log_event=True)
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8")
            raise RuntimeError(
                f"Inference API returned {exc.code}: {message}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Inference API is unavailable at {self.settings.predict_url}."
            ) from exc

        prediction = float(result["prediction"])
        actual = float(apartment["actual_price_per_m2"])
        return self._row_to_marker(
            row,
            prediction=prediction,
            actual=actual,
            added_at=result.get("timestamp")
            or datetime.now(timezone.utc).isoformat(),
        )

    def recent_predictions(self, limit: int = 20) -> List[Dict[str, Any]]:
        path = self.settings.inference_log_path
        if not path.exists():
            return []
        events = []
        with path.open("r", encoding="utf-8") as stream:
            lines = stream.readlines()[-limit:]
        for line in reversed(lines):
            payload = json.loads(line)
            prediction = float(payload["prediction"])
            apartment = {
                key: payload.get(key)
                for key in self._feature_columns
                if key in payload
            }
            if "area" in apartment and apartment["area"] is not None:
                apartment["area"] = float(apartment["area"])
            if "kitchen_area" in apartment and (
                apartment["kitchen_area"] is not None
            ):
                apartment["kitchen_area"] = float(apartment["kitchen_area"])
            actual = self._resolve_actual_price(
                {
                    **apartment,
                    "actual_price_per_m2": payload.get("actual_price_per_m2"),
                }
            )
            row_info = self._resolve_row(apartment)
            if row_info is None or actual is None:
                continue
            computed = self._compute_flags(prediction, actual)
            events.append(
                {
                    "timestamp": payload["timestamp"],
                    "model_version": str(
                        payload.get("model_version", "unknown")
                    ),
                    "listing_id": int(row_info["listing_id"]),
                    "lat": float(row_info["lat"]),
                    "lon": float(row_info["lon"]),
                    "marker_color": computed["marker_color"],
                    "flags": computed["flags"],
                    "prediction": prediction,
                    "actual_price_per_m2": actual,
                    "relative_error": computed["relative_error"],
                    "apartment": apartment,
                }
            )
        return events
