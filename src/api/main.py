from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from src.api.model_loader import ModelStore
from src.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Real Estate Inference API", version="0.1.0")
model_store = ModelStore()


@app.on_event("startup")
def startup_event() -> None:
    model_store.load()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model_store.is_ready() else "degraded",
        "model_ready": model_store.is_ready(),
        "model_version": model_store.version(),
        "error": model_store.error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is unavailable: "
                f"{model_store.error or 'not loaded'}"
            ),
        )
    prediction = model_store.predict(request.model_dump())
    return PredictionResponse.from_prediction(
        prediction,
        model_store.version(),
    )
