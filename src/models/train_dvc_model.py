#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import math
import os
import sys
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# Allow running the script directly via DVC.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.drift_core import build_reference_profile  # noqa: E402

FEATURES = [
    "region",
    "building_type",
    "level",
    "levels",
    "year",
    "month",
    "rooms",
    "area",
    "kitchen_area",
    "object_type",
    "weekday_number",
]
TARGET = "price_per_m2"


def _check_mlflow_reachable(tracking_uri: str, timeout: float = 5.0) -> None:
    """Fail fast with a clear message when the tracking server is not reachable."""
    base = tracking_uri.rstrip("/")
    last_exc: BaseException | None = None
    for suffix in ("/version", "/health", "/"):
        target = base + suffix
        try:
            with urllib_request.urlopen(target, timeout=timeout):
                return
        except urllib_error.HTTPError as exc:
            if exc.code is not None:
                return
            last_exc = exc
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            continue
    raise RuntimeError(
        f"MLflow недоступен по адресу {tracking_uri!r} (проверка за {timeout}s). "
        "Запустите сервер с --host 0.0.0.0. Из Docker на Windows/Mac обычно нужен "
        "URI вида http://host.docker.internal:5000 (см. MLFLOW_TRACKING_URI в compose). "
        f"Последняя ошибка: {last_exc!r}"
    ) from last_exc


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(
        mean_absolute_percentage_error(y_true[mask], y_pred[mask])
    )


def _resolve_train_test_masks(
    dataset: pd.DataFrame,
    random_state: int,
) -> tuple[pd.Series, pd.Series, str, str]:
    """Use the latest year as holdout when possible; otherwise shuffle-split."""
    years = sorted(
        int(year)
        for year in pd.Series(dataset["year"]).dropna().unique().tolist()
    )
    if years:
        max_year = years[-1]
        train_mask = dataset["year"] < max_year
        test_mask = dataset["year"] == max_year
        if bool(train_mask.any()) and bool(test_mask.any()):
            return (
                train_mask,
                test_mask,
                f"year < {max_year}",
                f"year == {max_year}",
            )

    if len(dataset) < 2:
        raise ValueError(
            "Недостаточно строк для обучения после очистки (нужно минимум 2)."
        )

    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=random_state,
        shuffle=True,
    )
    train_mask = pd.Series(False, index=dataset.index)
    test_mask = pd.Series(False, index=dataset.index)
    train_mask.iloc[train_idx] = True
    test_mask.iloc[test_idx] = True
    return (
        train_mask,
        test_mask,
        "shuffle_split_train_0.8",
        "shuffle_split_test_0.2",
    )


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение baseline-модели для DVC пайплайна."
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "120")

    data_path = Path(args.data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Файл датасета не найден: {data_path.resolve()}. "
            "Проверьте том ./data в docker-compose и наличие cleaned_data.csv."
        )

    _check_mlflow_reachable(args.tracking_uri)

    df = pd.read_csv(args.data_path)
    dataset = df[FEATURES + [TARGET]].dropna().copy()

    train_mask, test_mask, train_rule, test_rule = _resolve_train_test_masks(
        dataset,
        args.random_state,
    )

    x_train = dataset.loc[train_mask, FEATURES]
    y_train = dataset.loc[train_mask, TARGET]
    x_test = dataset.loc[test_mask, FEATURES]
    y_test = dataset.loc[test_mask, TARGET]

    if x_train.empty or x_test.empty:
        raise ValueError(
            "Пустой train или test после разбиения. Проверьте входные данные."
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    try:
        mlflow.set_experiment(args.experiment_name)
    except MlflowException as exc:
        err_text = str(exc)
        if "Invalid Host header" in err_text or "DNS rebinding" in err_text:
            raise RuntimeError(
                "MLflow отклонил заголовок Host (защита от DNS rebinding в MLflow 3.5+). "
                "Запустите tracking server с явным списком хостов, например:\n"
                "  mlflow server --host 0.0.0.0 --port 5000 "
                '--allowed-hosts "host.docker.internal:5000,host.docker.internal,'
                'localhost:*,127.0.0.1:*" ...\n'
                "Только для локальной разработки: флаг --disable-security-middleware."
            ) from exc
        raise

    model_path = ensure_parent(args.model_path)
    metadata_path = ensure_parent(args.metadata_path)
    metrics_path = ensure_parent(args.metrics_path)

    with mlflow.start_run(run_name=args.run_name) as run_info:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mape = _safe_mape(
            np.asarray(y_test),
            np.asarray(y_pred),
        )
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }
        if math.isfinite(mape):
            metrics["mape"] = mape
        reference_profile = build_reference_profile(
            x_reference=x_train,
            y_reference=y_train,
            y_prediction=y_pred,
            baseline_metrics=metrics,
        )

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rule", train_rule)
        mlflow.log_param("test_rule", test_rule)
        mlflow.log_param("features", json.dumps(FEATURES, ensure_ascii=False))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        joblib.dump(model, model_path)
        metadata = {
            "model_version": run_info.info.run_id,
            "model_name": "random_forest",
            "run_id": run_info.info.run_id,
            "tracking_uri": args.tracking_uri,
            "experiment_name": args.experiment_name,
            "features": FEATURES,
            "target": TARGET,
            "train_rule": train_rule,
            "test_rule": test_rule,
            "metrics": metrics,
            "reference_profile": reference_profile,
            "params": {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "random_state": args.random_state,
            },
        }

        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    try:
        run(parse_args())
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
