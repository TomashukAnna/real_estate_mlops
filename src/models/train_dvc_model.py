#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    df = pd.read_csv(args.data_path)
    dataset = df[FEATURES + [TARGET]].dropna().copy()

    train_mask = dataset["year"] <= 2024
    test_mask = dataset["year"] == 2025

    x_train = dataset.loc[train_mask, FEATURES]
    y_train = dataset.loc[train_mask, TARGET]
    x_test = dataset.loc[test_mask, FEATURES]
    y_test = dataset.loc[test_mask, TARGET]

    if x_train.empty or x_test.empty:
        raise ValueError(
            "Пустой train/test после temporal split. Проверьте входные данные."
        )

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

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

        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("train_rule", "year <= 2024")
        mlflow.log_param("test_rule", "year == 2025")
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
            "train_rule": "year <= 2024",
            "test_rule": "year == 2025",
            "metrics": metrics,
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
    run(parse_args())
