import os
import pickle as pkl
import pandas as pd
import numpy as np

from typing_extensions import Annotated
from http import HTTPStatus
from fastapi import FastAPI, APIRouter, HTTPException
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from apsm.app.schemas import (
    ModelConfig,
    FitRequest,
    FitResponse,
    PredictRequest,
    PredictResponse,
    ModelListResponse,
    RemoveResponse
)
from apsm.utils import setup_logger


model_router = APIRouter()

logger = setup_logger(
    name='model',
    log_file=os.getenv('PYTHONPATH') + '/logs/model_api.log'
)

models = {}


@model_router.post(
    '/fit',
    response_model=FitResponse,
    status_code=HTTPStatus.CREATED
)
async def fit(
    request: Annotated[FitRequest, '...']
) -> FitResponse:
    global models

    model_id = request.config.id
    model_type = request.config.ml_model_type
    data = request.data
    config = request.config.hyperparameters or {}

    if model_id in models:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_id}' уже существует."
        )

    try:
        if model_type == 'auto_arima':
            model = auto_arima(data)

        elif model_type == 'holt_winters':
            required_params = ['trend', 'seasonal', 'seasonal_periods']

            for param in required_params:
                if param not in config:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Параметр '{param}' обязателен."
                    )

            data_series = pd.Series(
                data=data,
                index=pd.date_range(
                    start=config.get('start_date', '2023-01-01'),
                    periods=len(data),
                    freq=config.get('freq', 'D')
                )
            )
            model = ExponentialSmoothing(
                data_series,
                trend=config['trend'],
                seasonal=config['seasonal'],
                seasonal_periods=config['seasonal_periods']
            )
            model = model.fit()

        else:
            raise HTTPException(
                status_code=400,
                detail='Неподдерживаемый тип модели.'
            )

        models[model_id] = pickle.dumps(model)

        return FitResponse(
            message=f"Модель '{model_id}' успешно обучена и сохранена."
        )

    except Exception as e:
        logger.error(f'Ошибка: {str(e)}')

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обучения модели '{model_id}': {str(e)}"
        )


@model_router.post(
    '/predict',
    response_model=PredictResponse,
    status_code=HTTPStatus.OK
)
async def predict_model(
    request: Annotated[PredictRequest, '']
) -> PredictResponse:
    global models

    model_id = request.id
    n_periods = request.n_periods

    if model_id not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_id}' не найдена."
        )

    try:
        model = pickle.loads(models[model_id])

        if hasattr(model, 'predict'):
            forecast = model.predict(n_periods=n_periods)
        elif hasattr(model, 'forecast'):
            forecast = model.forecast(steps=n_periods)
        else:
            raise HTTPException(
                status_code=400,
                detail='Неподдерживаемый тип модели.'
            )

        return PredictResponse(forecast=forecast.tolist())

    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка предсказания: {str(e)}"
        )


@model_router.get(
    '/list_models',
    response_model=ModelListResponse,
    status_code=HTTPStatus.OK
)
async def list_models() -> ModelListResponse:
    model_list = [{'id': model_id} for model_id in models.keys()]

    return ModelListResponse(models=model_list)


@model_router.delete(
    '/remove_all',
    response_model=RemoveResponse,
)
async def remove_all_models() -> RemoveResponse:
    global models
    models.clear()

    return RemoveResponse(message='Все модели успешно удалены.')
