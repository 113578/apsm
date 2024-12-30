import os
import shutil
from typing import Optional
import joblib
import pandas as pd
from typing_extensions import Annotated
from http import HTTPStatus
from fastapi import FastAPI, APIRouter, HTTPException
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima, StepwiseContext
from apsm.app.schemas import *
from apsm.utils import setup_logger

model_router = APIRouter()

logger = setup_logger(
    name='model',
    log_file='../logs/model.log'
)

model = None


@model_router.post(
    '/fit',
    response_model=FitResponse,
    status_code=HTTPStatus.CREATED
)
async def fit(
        request: Annotated[FitRequest, '...']
) -> FitResponse:
    global model

    model_id = request.config.id
    model_type = request.config.ml_model_type
    data = request.data
    config = request.config.hyperparameters or {}
    model_path = get_model_path(model_id)

    if model_path:
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_id}' уже существует."
        )

    try:
        if model_type == ModelType.auto_arima:
            model = auto_arima(data)
        elif model_type == ModelType.holt_winters:
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

        os.makedirs(f"models/{model_type.value}", exist_ok=True)
        joblib.dump(model, f"models/{model_type.value}/{model_id}.joblib")
        return FitResponse(
            message=f"Модель '{model_id}' успешно обучена и сохранена."
        )

    except Exception as e:
        logger.error(f'Ошибка: {str(e)}')

        raise HTTPException(
            status_code=500,
            detail=f"Ошибка обучения модели '{model_id}': {str(e)}"
        )


def get_model_path(model_id: str) -> Optional[str]:
    auto_arima_path = os.path.join("models", "auto_arima",
                                   model_id + ".joblib")
    holt_winters_path = os.path.join("models", "holt_winters",
                                     model_id + ".joblib")
    if os.path.exists(auto_arima_path):
        return auto_arima_path
    if os.path.exists(holt_winters_path):
        return holt_winters_path
    return None


@model_router.post(
    '/predict',
    response_model=PredictResponse,
    status_code=HTTPStatus.OK
)
async def predict_model(
        request: Annotated[PredictRequest, '']
) -> PredictResponse:
    global model

    n_periods = request.n_periods
    if not model:
        raise HTTPException(
            status_code=404,
            detail=f"Модель не активна."
        )
    try:
        if hasattr(model, 'forecast'):
            forecast = model.forecast(steps=n_periods)
        elif hasattr(model, 'predict'):
            forecast = model.predict(n_periods=n_periods)
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
    '/list',
    response_model=ModelListResponse,
    status_code=HTTPStatus.OK
)
async def list_models() -> ModelListResponse:
    models_list = [{"id": model_id.removesuffix(".joblib"),
                    "type": model_type} for model_type in ModelType if
                   os.path.exists(f"models/{model_type.value}") for model_id in
                   os.listdir(f"models/{model_type.value}")]
    return ModelListResponse(models=models_list)


@model_router.post(
    '/set',
    response_model=SetResponse,
)
async def set_active_model(
        request: Annotated[SetRequest, '']
) -> SetResponse:
    global model
    model_id = request.id
    model_path = get_model_path(model_id)

    if not model_path:
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_id}' не найдена."
        )

    model = joblib.load(model_path)

    return SetResponse(message=f'Модель {model_id} активна.')


@model_router.get(
    '/remove_all',
    response_model=RemoveResponse,
    status_code=HTTPStatus.OK
)
async def remove_all() -> RemoveResponse:
    global model
    model = None
    if os.path.exists("models"):
        shutil.rmtree("models")
    return RemoveResponse(message="Все модели удалены!")
