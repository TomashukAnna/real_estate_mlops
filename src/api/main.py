from time import perf_counter
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram
from prometheus_client import generate_latest

from src.api.model_loader import ModelStore
from src.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Real Estate Inference API", version="0.1.0")
model_store = ModelStore()

REQUEST_COUNT = Counter(
    "real_estate_api_requests_total",
    "Total HTTP requests processed by the API.",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "real_estate_api_request_latency_seconds",
    "Request latency in seconds.",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "real_estate_api_predictions_total",
    "Total prediction requests handled by the API.",
    ["model_version", "status"],
)
MODEL_READY = Gauge(
    "real_estate_api_model_ready",
    "Whether the model is loaded and ready for predictions.",
)


@app.on_event("startup")
def startup_event() -> None:
    model_store.load()
    MODEL_READY.set(1 if model_store.is_ready() else 0)


@app.middleware("http")
async def prometheus_middleware(
    request: Request,
    call_next,
) -> Response:
    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        REQUEST_COUNT.labels(
            method=request.method,
            path=path,
            status_code=str(status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            path=path,
        ).observe(perf_counter() - started_at)


@app.get("/health")
def health() -> Dict[str, Any]:
    MODEL_READY.set(1 if model_store.is_ready() else 0)
    return {
        "status": "ok" if model_store.is_ready() else "degraded",
        "model_ready": model_store.is_ready(),
        "model_version": model_store.version(),
        "error": model_store.error,
    }


@app.get("/metrics")
def metrics() -> Response:
    MODEL_READY.set(1 if model_store.is_ready() else 0)
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if not model_store.is_ready():
        PREDICTION_COUNT.labels(
            model_version=model_store.version(),
            status="error",
        ).inc()
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is unavailable: "
                f"{model_store.error or 'not loaded'}"
            ),
        )
    prediction = model_store.predict(request.model_dump())
    PREDICTION_COUNT.labels(
        model_version=model_store.version(),
        status="success",
    ).inc()
    return PredictionResponse.from_prediction(
        prediction,
        model_store.version(),
    )
