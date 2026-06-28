from __future__ import annotations

import os
from typing import Any, Dict, List

import folium
import pandas as pd
import requests
import streamlit as st
from branca.element import Element
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

BFF_BASE_URL = os.getenv("STREAMLIT_BFF_URL", "http://bff:8002").rstrip("/")
REQUEST_TIMEOUT = 60
DEFAULT_SAMPLE_SIZE = 10


def _request_json(
    path: str,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
) -> Any:
    url = f"{BFF_BASE_URL}{path}"
    if method == "POST":
        response = requests.post(url, json=payload or {}, timeout=REQUEST_TIMEOUT)
    else:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _format_currency(value: Any) -> str:
    if value is None or value == "":
        return "н/д"
    return f"{float(value):,.0f} RUB".replace(",", " ")


def _format_percent(value: Any) -> str:
    if value is None or value == "":
        return "н/д"
    return f"{float(value) * 100:.1f}%"


def _format_date(value: Any) -> str:
    if not value:
        return "н/д"
    return pd.to_datetime(value).strftime("%d.%m.%Y %H:%M:%S")


def _marker_color(color: str) -> str:
    if color == "#d14343":
        return "red"
    if color == "#1f9d55":
        return "green"
    return "gray"


def _popup_html(point: Dict[str, Any]) -> str:
    apartment = point.get("apartment", {})
    rows = "".join(
        f"<div><strong>{key}</strong>: {value}</div>"
        for key, value in apartment.items()
    )
    prediction = (
        f"<div><strong>prediction</strong>: "
        f"{_format_currency(point.get('prediction'))}</div>"
        if point.get("prediction") is not None
        else ""
    )
    actual = (
        f"<div><strong>actual_price_per_m2</strong>: "
        f"{_format_currency(point.get('actual_price_per_m2'))}</div>"
        if point.get("actual_price_per_m2") is not None
        else ""
    )
    relative_error = (
        f"<div><strong>relative_error</strong>: "
        f"{_format_percent(point.get('relative_error'))}</div>"
        if point.get("relative_error") is not None
        else ""
    )
    flags = ", ".join(point.get("flags", [])) or "нет"
    return (
        f"<div><h4>Квартира #{point.get('listing_id', 'н/д')}</h4>"
        f"{rows}{prediction}{actual}{relative_error}"
        f"<div><strong>флаги</strong>: {flags}</div></div>"
    )


def _merge_points(
    sample_points: List[Dict[str, Any]],
    recent_predictions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {
        int(point["listing_id"]): point for point in sample_points
    }
    for prediction in recent_predictions:
        listing_id = prediction.get("listing_id")
        lat = prediction.get("lat")
        lon = prediction.get("lon")
        if listing_id is None or lat is None or lon is None:
            continue
        merged[int(listing_id)] = {
            "listing_id": int(listing_id),
            "lat": float(lat),
            "lon": float(lon),
            "marker_color": prediction.get("marker_color", "gray"),
            "apartment": {
                **prediction.get("apartment", {}),
                "actual_price_per_m2": prediction.get("actual_price_per_m2"),
            },
            "prediction": prediction.get("prediction"),
            "actual_price_per_m2": prediction.get("actual_price_per_m2"),
            "relative_error": prediction.get("relative_error"),
            "flags": prediction.get("flags", []),
            "added_at": prediction.get("timestamp"),
        }
    return list(merged.values())


def _load_dashboard(force_new_sample: bool = False) -> None:
    if force_new_sample or not st.session_state.get("sample_points"):
        sample = _request_json(f"/map/sample?size={DEFAULT_SAMPLE_SIZE}")
        st.session_state["sample_points"] = sample["points"]
    st.session_state["predictions"] = _request_json("/predictions/recent?limit=25")
    st.session_state["drift"] = _request_json("/drift/summary")
    st.session_state["retrain_status"] = _request_json("/retrain/status")


def _load_experiments() -> None:
    st.session_state["experiments"] = _request_json("/experiments/summary?limit=20")


def _ensure_state() -> None:
    if "sample_points" not in st.session_state:
        st.session_state["sample_points"] = []
    if "predictions" not in st.session_state:
        st.session_state["predictions"] = []
    if "drift" not in st.session_state:
        st.session_state["drift"] = None
    if "retrain_status" not in st.session_state:
        st.session_state["retrain_status"] = None
    if "experiments" not in st.session_state:
        st.session_state["experiments"] = None
    if "loaded" not in st.session_state:
        _load_dashboard(force_new_sample=True)
        st.session_state["loaded"] = True


def _current_points() -> List[Dict[str, Any]]:
    return _merge_points(
        st.session_state["sample_points"],
        st.session_state["predictions"],
    )


def _build_map(points: List[Dict[str, Any]]) -> folium.Map:
    if points:
        center = [
            sum(point["lat"] for point in points) / len(points),
            sum(point["lon"] for point in points) / len(points),
        ]
    else:
        center = [59.9391, 30.3159]

    map_view = folium.Map(location=center, zoom_start=9, tiles="CartoDB positron")
    # Leaflet adds a flag image in the attribution bar; hide it inside the map
    # document so it applies under streamlit-folium's isolated HTML frame.
    map_view.get_root().header.add_child(
        Element(
            "<style>.leaflet-attribution-flag{display:none!important;}</style>"
        )
    )
    cluster = MarkerCluster().add_to(map_view)
    for point in points:
        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=folium.Popup(_popup_html(point), max_width=360),
            icon=folium.Icon(color=_marker_color(point["marker_color"])),
        ).add_to(cluster)
    return map_view


def _predictions_frame(predictions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in predictions:
        apartment = item.get("apartment", {})
        rows.append(
            {
                "Время": _format_date(item.get("timestamp")),
                "Прогноз": _format_currency(item.get("prediction")),
                "Факт": _format_currency(item.get("actual_price_per_m2")),
                "Относительная ошибка": _format_percent(item.get("relative_error")),
                "Флаги": ", ".join(item.get("flags", [])) or "нет",
                "Квартира": (
                    f"комнат {apartment.get('rooms', 'н/д')}, "
                    f"площадь {apartment.get('area', 'н/д')}, "
                    f"регион {apartment.get('region', 'н/д')}"
                ),
            }
        )
    return pd.DataFrame(rows)


def _experiments_frame(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "Название": run.get("name"),
                "Статус": run.get("status"),
                "Начало": _format_date(run.get("start_time")),
                "Длительность, с": run.get("duration_seconds"),
                "Метрики": "\n".join(
                    f"{key}: {value:.4f}"
                    for key, value in run.get("metrics", {}).items()
                ),
                "Параметры": "\n".join(
                    f"{key}: {value}" for key, value in run.get("params", {}).items()
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_inference_tab() -> None:
    points = _current_points()
    predictions = st.session_state["predictions"]
    drift = st.session_state["drift"] or {}
    retrain_status = st.session_state["retrain_status"] or {}

    controls = st.columns([1, 1, 1, 1])
    if controls[0].button("Обновить дашборд", use_container_width=True):
        _load_dashboard(force_new_sample=False)
        st.rerun()
    if controls[1].button("Обновить выборку на карте", use_container_width=True):
        _load_dashboard(force_new_sample=True)
        st.rerun()
    if controls[2].button("Добавить новую квартиру", use_container_width=True):
        excluded_ids = [int(point["listing_id"]) for point in points]
        _request_json(
            "/map/add-random",
            method="POST",
            payload={"excluded_ids": excluded_ids},
        )
        _load_dashboard(force_new_sample=False)
        st.rerun()
    if controls[3].button("Запустить переобучение", use_container_width=True):
        _request_json("/retrain", method="POST")
        _load_dashboard(force_new_sample=False)
        _load_experiments()
        st.rerun()

    st.subheader("Карта предсказаний")
    st.caption(
        "Маркеры используют результаты /predict: зелёный — если прогноз >= факта, "
        "красный — если ниже; последние сохранённые предсказания обновляют точку на карте."
    )
    st_folium(_build_map(points), width=None, height=620, returned_objects=[])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Точек на карте", len(points))
    metric_cols[1].metric("Предсказаний", len(predictions))
    metric_cols[2].metric(
        "Окно дрейфа",
        drift.get("observation_count", 0),
    )
    metric_cols[3].metric(
        "Размеченных наблюдений",
        drift.get("labeled_observation_count", 0),
    )

    st.subheader("Сводка по дрейфу")
    drift_cols = st.columns(4)
    drift_cols[0].metric(
        "Дрейф данных",
        f"{drift.get('data_drift', {}).get('score', 0.0):.3f}",
    )
    drift_cols[1].metric(
        "Дрейф предсказаний",
        f"{drift.get('prediction_drift', {}).get('score', 0.0):.3f}",
    )
    drift_cols[2].metric(
        "Дрейф таргета",
        f"{drift.get('target_drift', {}).get('score', 0.0):.3f}",
    )
    drift_cols[3].metric(
        "Концепт-дрейф",
        f"{drift.get('concept_drift', {}).get('score', 0.0):.3f}",
    )
    st.caption(
        "Выявленные признаки: "
        + ", ".join(drift.get("drifted_features", []))
        if drift.get("drifted_features")
        else "Выявленные признаки: нет"
    )

    st.subheader("Последние предсказания")
    st.dataframe(_predictions_frame(predictions), use_container_width=True)

    st.subheader("Переобучение")
    status_cols = st.columns(4)
    status_cols[0].metric("Статус", retrain_status.get("status", "неизвестно"))
    status_cols[1].metric("Запущено", _format_date(retrain_status.get("started_at")))
    status_cols[2].metric(
        "Завершено",
        _format_date(retrain_status.get("finished_at")),
    )
    status_cols[3].metric(
        "Версия модели",
        retrain_status.get("model_version", "н/д"),
    )
    if retrain_status.get("error"):
        st.error(retrain_status["error"])
    st.code(retrain_status.get("logs", "") or "Логи пока отсутствуют.", language="text")


def _render_experiments_tab() -> None:
    experiments = st.session_state["experiments"] or {}

    if st.button("Загрузить / обновить эксперименты", use_container_width=False):
        _load_experiments()
        st.rerun()

    if st.session_state["experiments"] is None:
        st.info("Эксперименты загружаются по запросу, чтобы интерфейс работал быстрее.")
        return

    st.subheader("Реестр моделей")
    registry = experiments.get("registry", {})
    registry_cols = st.columns(4)
    registry_cols[0].metric("Модель", registry.get("model_name", "н/д"))
    registry_cols[1].metric("Алиас", registry.get("alias", "н/д"))
    registry_cols[2].metric("Версия алиаса", registry.get("alias_version", "н/д"))
    registry_cols[3].metric("Стадия", registry.get("stage", "н/д"))

    st.caption(
        f"Tracking URI: {experiments.get('tracking_uri', 'н/д')} | "
        f"Эксперимент: {experiments.get('experiment_name', 'н/д')}"
    )

    if experiments.get("error"):
        st.warning(experiments["error"])

    st.subheader("Запуски")
    runs = experiments.get("runs", [])
    if runs:
        st.dataframe(_experiments_frame(runs), use_container_width=True)
    else:
        st.info("Запуски MLflow отсутствуют.")


def main() -> None:
    st.set_page_config(
        page_title="Real Estate MLOps",
        page_icon="🏠",
        layout="wide",
    )
    st.title("Real Estate MLOps")
    st.caption(
        "Интерфейс Streamlit поверх существующих BFF и сервиса инференса."
    )

    try:
        _ensure_state()
    except requests.RequestException as exc:
        st.error(f"Не удалось подключиться к BFF по адресу {BFF_BASE_URL}: {exc}")
        st.stop()

    inference_tab, experiments_tab = st.tabs(["Инференс", "Эксперименты"])
    with inference_tab:
        _render_inference_tab()
    with experiments_tab:
        _render_experiments_tab()


if __name__ == "__main__":
    main()
