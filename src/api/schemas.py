from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    region: int = Field(..., description="Region id")
    building_type: int = Field(..., description="Building type id")
    level: int = Field(..., ge=0)
    levels: int = Field(..., ge=0)
    year: int = Field(..., ge=1900, le=2100)
    month: int = Field(..., ge=1, le=12)
    rooms: int = Field(..., ge=0)
    area: float = Field(..., gt=0)
    kitchen_area: float = Field(..., ge=0)
    object_type: int = Field(..., description="Object type id")
    weekday_number: int = Field(..., ge=0, le=6)
    actual_price_per_m2: float = Field(
        -1,
        description=(
            "Observed target value. Use -1 when the factual price "
            "is unknown."
        ),
    )
    log_event: bool = Field(
        True,
        description="Error, the prediction's not appended.",
    )

    @field_validator("actual_price_per_m2")
    @classmethod
    def validate_actual_price_per_m2(cls, value: float) -> float:
        if value == -1 or value > 0:
            return value
        raise ValueError("actual_price_per_m2 must be -1 or a positive value")


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    timestamp: datetime

    @classmethod
    def from_prediction(
        cls,
        prediction: float,
        model_version: str,
    ) -> "PredictionResponse":
        return cls(
            prediction=prediction,
            model_version=model_version,
            timestamp=datetime.now(timezone.utc),
        )
