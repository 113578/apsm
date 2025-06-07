import os
import traceback
import pickle as pkl
from http import HTTPStatus

import joblib
import pandas as pd
import lightgbm
from typing_extensions import Annotated
from fastapi import APIRouter, HTTPException
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from apsm.utils import setup_logger, get_model_path
from apsm.app.schemas import (
    ModelType,
    FitRequest,
    FitResponse,
    PredictRequest,
    PredictResponse,
    ModelListResponse,
    RemoveResponse,
    SetRequest,
    SetResponse,
    ModelManager
)


logger = setup_logger(
    name='api',
    log_file=os.getenv('PYTHONPATH') + '/logs/api.log'
)

model_router = APIRouter()


@model_router.post(
    '/fit',
    response_model=FitResponse,
    status_code=HTTPStatus.CREATED
)
async def fit(
    request: Annotated[
        FitRequest,
        'Схема запроса для обучения модели.'
    ]
) -> FitResponse:
    """
    Обучение модели на основе предоставленных данных.

    Parameters
    ----------
    request : FitRequest
        Запрос на обучение, включающий данные и конфигурацию модели.

    Returns
    -------
    FitResponse
        Сообщение об успешном обучении модели или ошибка в случае сбоя.
    """
    model_id = request.config.id
    model_type = request.config.ml_model_type
    data = request.data
    config = request.config.hyperparameters or {}
    model_path = get_model_path(model_id)

    if model_path:
        logger.error("Модель '%s' уже существует.", model_id)
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_id}' уже существует."
        )

    try:
        if model_type == ModelType.auto_arima:
            ModelManager.model = auto_arima(data)
        elif model_type == ModelType.holt_winters:
            required_params = ['trend', 'seasonal', 'seasonal_periods']
            for param in required_params:
                if param not in config:
                    logger.error("Параметра '%s' обязателен.", param)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Параметр '{param}' обязателен."
                    )
        elif model_type == ModelType.pretrained:
            with open('', 'rb') as file:
                model = pkl.load(file)

            trend = config['trend']
            seasonal = config['seasonal']
            seasonal_periods = config['seasonal_periods']

            if trend not in ['add', 'mul', None]:
                logger.error("Параметр 'trend' должен быть 'add', 'mul' или None.")
                raise HTTPException(
                    status_code=400,
                    detail="Параметр 'trend' должен быть 'add', 'mul' или None."
                )
            if seasonal not in ['add', 'mul', None]:
                logger.error("Параметр 'seasonal' должен быть 'add', 'mul' или None.")
                raise HTTPException(
                    status_code=400,
                    detail="Параметр 'seasonal' должен быть 'add', 'mul' или None."
                )

            try:
                seasonal_periods = int(seasonal_periods)
                if seasonal_periods <= 0:
                    logger.error("Параметр 'seasonal_periods' должен быть положительным целым числом.")
                    raise ValueError("Параметр 'seasonal_periods' должен быть положительным целым числом.")
            except (ValueError, TypeError) as exc:
                logger.error("Параметр 'seasonal_periods' должен быть положительным целым числом.")
                raise HTTPException(
                    status_code=400,
                    detail="Параметр 'seasonal_periods' должен быть положительным целым числом."
                ) from exc

            if seasonal == 'mul' and any(x <= 0 for x in data):
                logger.error("Все значения данных должны быть положительными для мультипликативной сезонности.")
                raise HTTPException(
                    status_code=400,
                    detail="Все значения данных должны быть положительными для мультипликативной сезонности."
                )
            if len(data) < seasonal_periods:
                logger.error("Количество данных должно быть больше или равно 'seasonal_periods'.")
                raise HTTPException(
                    status_code=400,
                    detail="Количество данных должно быть больше или равно 'seasonal_periods'."
                )

            data_series = pd.Series(
                data=data,
                index=pd.date_range(
                    start=config.get('start_date', '2023-01-01'),
                    periods=len(data),
                    freq=config.get('freq', 'D')
                )
            )
            ModelManager.model = ExponentialSmoothing(
                data_series,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            ).fit()

        else:
            logger.error('Неподдерживаемый тип модели.')
            raise HTTPException(
                status_code=400,
                detail='Неподдерживаемый тип модели.'
            )

        os.makedirs('models/auto_arima', exist_ok=True)
        os.makedirs('models/holt_winters', exist_ok=True)
        joblib.dump(ModelManager.model, f'models/{model_type.value}/{model_id}.joblib')

        logger.info("Модель '%s' успешно обучена и сохранена.", model_id)
        return FitResponse(
            message=f"Модель '{model_id}' успешно обучена и сохранена."
        )

    except Exception as e:
        traceback_text = traceback.format_exc()
        logger.error("Ошибка обучения модели '%s': %s", model_id, traceback_text)
        raise HTTPException(
            status_code=500,
            detail={
                'message': f"Ошибка обучения модели '{model_id}'",
                'error': str(e),
                'traceback': traceback_text
            }
        ) from e


@model_router.post(
    '/predict',
    response_model=PredictResponse,
    status_code=HTTPStatus.OK
)
async def predict_model(
    request: Annotated[
        PredictRequest,
        'Схема запроса для предсказаний моделью.'
    ]
) -> PredictResponse:
    """
    Выполнение предсказания с использованием активной модели.

    Parameters
    ----------
    request : PredictRequest
        Запрос на предсказание, включающий количество периодов для прогноза.

    Returns
    -------
    PredictResponse
        Предсказания в виде списка значений.
    """
    n_periods, future_forecast = int(request.n_periods), bool(request.future_forecast)

    if not ModelManager.model:
        logger.error('Модель не активна.')
        raise HTTPException(
            status_code=404,
            detail='Модель не активна.'
        )

    try:
        if future_forecast:
            if hasattr(ModelManager.model, 'forecast'):
                forecast = ModelManager.model.forecast(steps=n_periods)
            elif hasattr(ModelManager.model, 'predict'):
                forecast = ModelManager.model.predict(n_periods=n_periods)
            else:
                logger.error('Неподдерживаемый тип модели.')
                raise HTTPException(
                    status_code=400,
                    detail='Неподдерживаемый тип модели.'
                )
        else:
            if hasattr(ModelManager.model, 'predict'):
                forecast = ModelManager.model.predict(start=0, end=n_periods - 1)
            else:
                logger.error('Неподдерживаемый тип модели.')
                raise HTTPException(
                    status_code=400,
                    detail='Неподдерживаемый тип модели.'
                )

        return PredictResponse(forecast=forecast.tolist())

    except Exception as e:
        logger.error('Ошибка предсказания: %s', str(e))
        raise HTTPException(
            status_code=500,
            detail=f'Ошибка предсказания: {str(e)}'
        ) from e


@model_router.get(
    '/list',
    response_model=ModelListResponse,
    status_code=HTTPStatus.OK
)
async def list_models() -> ModelListResponse:
    """
    Получение списка всех сохраненных моделей.

    Returns
    -------
    ModelListResponse
        Список доступных моделей, включающий их идентификаторы и типы.
    """
    models_list = [
        {
            'id': model_id.removesuffix('.joblib'),
            'type': model_type
        }
        for model_type in ModelType
        if os.path.exists(f'models/{model_type.value}')
        for model_id in os.listdir(f'models/{model_type.value}')
    ]

    return ModelListResponse(models=models_list)


@model_router.post(
    '/set',
    response_model=SetResponse,
)
async def set_active_model(
        request: Annotated[
            SetRequest,
            'Схема запроса к установке модели.'
        ]
) -> SetResponse:
    """
    Установка активной модели для выполнения операций.

    Parameters
    ----------
    request : SetRequest
        Запрос, содержащий идентификатор модели для активации.

    Returns
    -------
    SetResponse
        Сообщение об успешной активации модели.
    """
    model_id = request.id
    model_path = get_model_path(model_id)

    if not model_path:
        logger.error("Модель '%s' не найдена.", model_id)
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_id}' не найдена."
        )

    ModelManager.model = joblib.load(model_path)

    logger.info('Модель %s активна.', model_id)

    return SetResponse(message=f'Модель {model_id} активна.')


@model_router.get(
    '/remove_all',
    response_model=RemoveResponse,
    status_code=HTTPStatus.OK
)
async def remove_all() -> RemoveResponse:
    """
    Удаление всех сохраненных моделей.

    Returns
    -------
    RemoveResponse
        Сообщение об успешном удалении всех моделей.
    """
    ModelManager.model = None

    for folder in ['/models/auto_arima', '/models/holt_winters']:
        path = os.getenv('PYTHONPATH') + folder
        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    logger.info('Все модели удалены!')

    return RemoveResponse(message='Все модели удалены!')
