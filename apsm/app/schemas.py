from enum import Enum
from typing import (
    List,
    Dict,
    Optional,
    Union,
    Any
)

from pydantic import BaseModel


class ModelType(str, Enum):
    auto_arima = 'auto_arima'
    holt_winters = 'holt_winters'
    catboost = 'catboost'


class ModelConfig(BaseModel):
    id: str
    ml_model_type: ModelType
    hyperparameters: Optional[
        Dict[str, Union[str, bool, int, float]]
    ] = None


class FitRequest(BaseModel):
    data: List[float]
    config: ModelConfig


class FitResponse(BaseModel):
    message: str


class PredictRequest(BaseModel):
    n_periods: int
    future_forecast: bool = True


class PredictResponse(BaseModel):
    forecast: List[float]


class ModelListResponse(BaseModel):
    models: List[Dict[str, str]]


class RemoveResponse(BaseModel):
    message: str


class StatusResponse(BaseModel):
    response: Dict[str, str]


class SetRequest(BaseModel):
    id: str


class SetResponse(BaseModel):
    message: str


class LoadRequest(BaseModel):
    id: str


class LoadResponse(BaseModel):
    message: str
