import os
import traceback
import pickle as pkl
from http import HTTPStatus

import joblib
import pandas as pd
from typing_extensions import Annotated
from fastapi import APIRouter, HTTPException
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from catboost import CatBoostRegressor
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
    SetResponse
)
from apsm.app.data_preprocessing import preprocess_time_series, extract_time_series_features

logger = setup_logger(
    name='api',
    log_file=os.getenv('PYTHONPATH') + '/logs/api.log'
)

model_router = APIRouter()


# Глобальный менеджер модели для хранения активной ML-модели
class ModelManager:
    model = None


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
    model_id = request.config.id
    model_type = request.config.ml_model_type
    data = request.data
    config = request.config.hyperparameters or {}
    # Для catboost определяем тип данных
    data_type = 'currency' if 'currency' in model_id else 'stock'
    model_path = get_model_path(model_id, model_type.value, data_type)

    if model_path and model_type != ModelType.catboost:
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
            trend = config['trend']
            seasonal = config['seasonal']
            seasonal_periods = config['seasonal_periods']
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
        elif model_type == ModelType.catboost:
            # Для catboost используем предобученную модель, обучение не требуется
            catboost_path = get_model_path(model_id, 'catboost', data_type)
            if not catboost_path or not os.path.exists(catboost_path):
                logger.error("CatBoost модель не найдена по пути %s", catboost_path)
                raise HTTPException(
                    status_code=404,
                    detail=f"CatBoost модель не найдена по пути {catboost_path}"
                )
            ModelManager.model = CatBoostRegressor()
            ModelManager.model.load_model(catboost_path)
        else:
            logger.error('Неподдерживаемый тип модели.')
            raise HTTPException(
                status_code=400,
                detail='Неподдерживаемый тип модели.'
            )

        if model_type == ModelType.auto_arima:
            os.makedirs('models/auto_arima', exist_ok=True)
            joblib.dump(ModelManager.model, f'models/auto_arima/{model_id}.joblib')
        elif model_type == ModelType.holt_winters:
            os.makedirs('models/holt_winters', exist_ok=True)
            joblib.dump(ModelManager.model, f'models/holt_winters/{model_id}.joblib')
        # Для catboost не сохраняем модель, она уже предобучена

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
    n_periods = int(request.n_periods)
    future_forecast = bool(getattr(request, 'future_forecast', True))

    if not ModelManager.model:
        logger.error('Модель не активна.')
        raise HTTPException(
            status_code=404,
            detail='Модель не активна.'
        )

    try:
        if hasattr(ModelManager.model, 'predict') and hasattr(ModelManager.model, 'load_model'):
            # CatBoost: ожидаем, что данные для прогноза будут в request.data
            data = getattr(request, 'data', None)
            if data is None:
                logger.error('Для CatBoost необходимо передать данные для предсказания.')
                raise HTTPException(
                    status_code=400,
                    detail='Для CatBoost необходимо передать данные для предсказания.'
                )
            # Предобработка данных для catboost
            import numpy as np
            df = pd.DataFrame({'value': data})
            df['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
            features = extract_time_series_features(df[['Date', 'value']])
            X = features.values[-len(data):, 1:]  # последние n значений
            forecast = ModelManager.model.predict(X)[:n_periods]
            return PredictResponse(forecast=forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast))
        elif future_forecast:
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
        return PredictResponse(forecast=forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast))

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
            'type': model_type.value
        }
        for model_type in ModelType
        if os.path.exists(f'models/{model_type.value}')
        for model_id in os.listdir(f'models/{model_type.value}')
    ]
    # Добавим catboost, если файл существует
    catboost_path = os.path.join('models', 'pretrained', 'classic', 'currency', 'cb.pkl')
    if os.path.exists(catboost_path):
        models_list.append({'id': 'catboost_pretrained', 'type': 'catboost'})
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
    # Определяем тип модели по id
    if model_id == 'catboost_pretrained':
        model_type = 'catboost'
        # Попробовать оба типа данных
        model_path = get_model_path(model_id, model_type, 'currency')
        if not model_path:
            model_path = get_model_path(model_id, model_type, 'stock')
    else:
        model_type = None
        model_path = get_model_path(model_id, model_type)

    if not model_path:
        logger.error("Модель '%s' не найдена.", model_id)
        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_id}' не найдена."
        )

    if model_type == 'catboost':
        # Открываем модель через pickle (joblib/pickle)
        import pickle
        with open(model_path, 'rb') as f:
            ModelManager.model = pickle.load(f)
    else:
        ModelManager.model = joblib.load(model_path)

    logger.info('Модель %s активна.', model_id)
    return SetResponse(message=f'Модель {model_id} активна.')


@model_router.get(
    '/remove_all',
    response_model=RemoveResponse,
    status_code=HTTPStatus.OK
)
async def remove_all_models() -> RemoveResponse:
    """
    Удаление всех сохраненных моделей.

    Returns
    -------
    RemoveResponse
        Сообщение об успешном удалении всех моделей.
    """
    import shutil
    removed = []
    for folder in ['models/auto_arima', 'models/holt_winters']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            removed.append(folder)
    # CatBoost предобученную модель не удаляем
    ModelManager.model = None
    logger.info('Все модели удалены: %s', removed)
    return RemoveResponse(message='Все модели удалены (кроме предобученной CatBoost).')
