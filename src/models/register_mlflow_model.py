#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Регистрация модели в MLflow Registry по run_id из metadata."
    )
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--registered-model-name", required=True)
    parser.add_argument("--artifact-path", default="model")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--stage", default="Staging")
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def ensure_parent(path_str: str) -> Path:
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_run_id(metadata_path: str) -> str:
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    run_id = payload.get("run_id")
    if not run_id:
        raise ValueError("В metadata.json отсутствует поле 'run_id'.")
    return str(run_id)


def find_existing_version(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> Optional[ModelVersion]:
    versions = client.search_model_versions(f"name = '{model_name}'")
    for version in versions:
        if version.run_id == run_id:
            return version
    return None


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)


def run(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    run_id = get_run_id(args.metadata_path)
    ensure_registered_model(client, args.registered_model_name)

    existing = find_existing_version(client, args.registered_model_name, run_id)
    if existing:
        model_version = existing
    else:
        source = f"runs:/{run_id}/{args.artifact_path}"
        model_version = client.create_model_version(
            name=args.registered_model_name,
            source=source,
            run_id=run_id,
        )

    if args.alias:
        client.set_registered_model_alias(
            name=args.registered_model_name,
            alias=args.alias,
            version=model_version.version,
        )

    if args.stage:
        client.transition_model_version_stage(
            name=args.registered_model_name,
            version=model_version.version,
            stage=args.stage,
            archive_existing_versions=True,
        )

    result: Dict[str, Any] = {
        "registered_model_name": args.registered_model_name,
        "run_id": run_id,
        "version": str(model_version.version),
        "stage": args.stage,
        "alias": args.alias,
        "tracking_uri": args.tracking_uri,
        "status": "ok",
    }
    output_path = ensure_parent(args.output_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    run(parse_args())
