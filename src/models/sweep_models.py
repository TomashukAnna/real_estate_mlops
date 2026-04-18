#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from itertools import product
from typing import Dict, Iterable, List, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


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


def build_param_grid(
    grid: Dict[str, List[object]],
) -> Iterable[Dict[str, object]]:
    """Простой генератор комбинаций гиперпараметров."""
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def load_dataset(data_path: str) -> pd.DataFrame:
    """Читаем данные и делаем базовую валидацию колонок."""
    df = pd.read_csv(data_path)
    required_columns = set(FEATURES + [TARGET])
    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"В данных не хватает колонок: {missing}")

    # На этом этапе нам важнее стабильный прогон, чем агрессивная очистка.
    return df[FEATURES + [TARGET]].dropna().copy()


def split_by_year(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Держим тот же split по времени, чтобы результаты были сопоставимы."""
    train_mask = df["year"] <= 2024
    test_mask = df["year"] == 2025

    x_train = df.loc[train_mask, FEATURES]
    y_train = df.loc[train_mask, TARGET]
    x_test = df.loc[test_mask, FEATURES]
    y_test = df.loc[test_mask, TARGET]

    if x_train.empty or x_test.empty:
        raise ValueError(
            "После split по годам train/test оказался пустым. "
            "Проверьте годовые данные."
        )

    return x_train, y_train, x_test, y_test


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Считаем ключевые метрики качества для регрессии."""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "mape": mape, "rmse": rmse, "r2": r2}


def get_candidates():
    return [
        (
            "random_forest",
            RandomForestRegressor,
            {
                "n_estimators": [120],
                "max_depth": [10],
                "random_state": [42],
                "n_jobs": [-1],
            },
        ),
        (
            "extra_trees",
            ExtraTreesRegressor,
            {
                "n_estimators": [120],
                "max_depth": [10],
                "random_state": [42],
                "n_jobs": [-1],
            },
        ),
        (
            "gradient_boosting",
            GradientBoostingRegressor,
            {
                "n_estimators": [100],
                "learning_rate": [0.1],
                "max_depth": [3],
                "random_state": [42],
            },
        ),
        (
            "ridge",
            Ridge,
            {
                "alpha": [1.0, 10.0],
                "random_state": [42],
            },
        ),
    ]


def format_run_name(model_name: str, params: Dict[str, object]) -> str:
    """Делаем читаемое имя run, чтобы проще фильтровать в MLflow UI."""
    pieces = [f"{key}={value}" for key, value in params.items()]
    return f"{model_name} | " + ", ".join(pieces)


def run_sweep(args: argparse.Namespace) -> None:
    """Запускаем перебор моделей и складываем всё в MLflow."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    df = load_dataset(args.data_path)
    x_train, y_train, x_test, y_test = split_by_year(df)

    print(f"Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Experiment: {args.experiment_name}")

    best_result = None
    total_runs = 0

    with mlflow.start_run(run_name=args.parent_run_name):
        # Логируем общий контекст один раз, чтобы потом не вспоминать вручную.
        mlflow.log_param("data_path", args.data_path)
        mlflow.log_param("target", TARGET)
        mlflow.log_param("features", json.dumps(FEATURES, ensure_ascii=False))
        mlflow.log_param("train_rule", "year <= 2024")
        mlflow.log_param("test_rule", "year == 2025")
        mlflow.log_param("models_count", len(get_candidates()))

        for model_name, model_cls, param_grid in get_candidates():
            for params in build_param_grid(param_grid):
                total_runs += 1
                run_name = format_run_name(model_name, params)

                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tag("stage", "model_sweep")
                    mlflow.set_tag("model_family", model_name)
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_params(params)

                    model = model_cls(**params)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)

                    metrics = evaluate_model(y_test, y_pred)
                    mlflow.log_metrics(metrics)
                    mlflow.sklearn.log_model(model, artifact_path="model")

                    if best_result is None or (
                        metrics["rmse"] < best_result["metrics"]["rmse"]
                    ):
                        best_result = {
                            "model_name": model_name,
                            "params": params,
                            "metrics": metrics,
                        }

                    print(
                        f"[{total_runs}] {model_name} "
                        f"RMSE={metrics['rmse']:.2f}, R2={metrics['r2']:.4f}"
                    )

        if best_result is None:
            raise RuntimeError("Перебор завершился без результатов.")

        # Сохраняем победителя и в родительский run для быстрого чтения.
        mlflow.log_param("best_model_name", best_result["model_name"])
        mlflow.log_param(
            "best_model_params",
            json.dumps(best_result["params"], ensure_ascii=False),
        )
        mlflow.log_metric("best_mae", best_result["metrics"]["mae"])
        mlflow.log_metric("best_mape", best_result["metrics"]["mape"])
        mlflow.log_metric("best_rmse", best_result["metrics"]["rmse"])
        mlflow.log_metric("best_r2", best_result["metrics"]["r2"])

    print("\nЛучший результат:")
    print(
        f"Модель: {best_result['model_name']}\n"
        f"Параметры: {best_result['params']}\n"
        f"MAE: {best_result['metrics']['mae']:.2f}\n"
        f"MAPE: {best_result['metrics']['mape']:.4f}\n"
        f"RMSE: {best_result['metrics']['rmse']:.2f}\n"
        f"R2: {best_result['metrics']['r2']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Перебор моделей недвижимости с логированием в MLflow."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/cleaned_data.csv",
        help="Путь к подготовленному датасету.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="Адрес MLflow Tracking Server.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="real_estate_model_sweep",
        help="Имя эксперимента в MLflow.",
    )
    parser.add_argument(
        "--parent-run-name",
        type=str,
        default="housing_price_sweep_v1",
        help="Имя родительского run, внутри которого будут вложенные прогоны.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_sweep(parse_args())
