from pydantic import BaseModel
from typing import (
    List,
    Dict,
    Optional,
    Literal,
    Union
)


class ModelConfig(BaseModel):
    id: str
    ml_model_type: Literal[
        'auto_arima', 'holt_winters'
    ]
    hyperparameters: Optional[
        Dict[str, Union[str, bool, int, float]]
    ] = None


class FitRequest(BaseModel):
    data: List[float]
    n_periods: int
    config: ModelConfig


class FitResponse(BaseModel):
    message: str


class PredictRequest(BaseModel):
    id: str
    n_periods: int


class PredictResponse(BaseModel):
    forecast: List[float]


class ModelListResponse(BaseModel):
    models: List[Dict[str, str]]


class RemoveResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    response: Dict[str, str]


class LoadRequest(BaseModel):
    id: str


class LoadResponse(BaseModel):
    message: str
