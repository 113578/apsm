import os
import traceback
import pickle as pkl
import shutil
from http import HTTPStatus

import joblib
import pandas as pd
from typing_extensions import Annotated
from fastapi import APIRouter, HTTPException
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from apsm.utils import (
    setup_logger,
    get_model_path,
    get_transformer_path
)
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
from apsm.app.data_preprocessing import (
    preprocess_time_series,
    extract_time_series_features,
    inverse_preprocess_time_series
)


logger = setup_logger(
    name='api',
    log_file=os.getenv('PYTHONPATH') + '/logs/api.log'
)

model_router = APIRouter()


# Глобальный менеджер модели для хранения активной ML-модели.
class ModelManager:
    model = None
    data_type: str


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
    Обучает модель на предоставленных данных и сохраняет её. Если модель уже существует, возвращает ошибку.

    Parameters
    ----------
    request : FitRequest
        Запрос, содержащий конфигурацию модели, данные и гиперпараметры.

    Returns
    -------
    FitResponse
        Сообщение об успешном обучении и сохранении модели.

    Raises
    ------
    HTTPException
        Если модель уже существует, отсутствуют обязательные параметры или возникает ошибка обучения.
    """
    model_id = request.config.id
    model_type = request.config.ml_model_type
    data = request.data
    config = request.config.hyperparameters or {}
    data_type = request.config.data_type

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
            ModelManager.data_type = data_type

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
            ModelManager.data_type = data_type

        elif model_type == ModelType.catboost:
            catboost_path = get_model_path(model_id, 'catboost', data_type)

            if not catboost_path or not os.path.exists(catboost_path):
                logger.error("CatBoost модель не найдена по пути %s", catboost_path)
                raise HTTPException(
                    status_code=404,
                    detail=f"CatBoost модель не найдена по пути {catboost_path}"
                )

            with open(catboost_path, 'rb') as file:
                ModelManager.model = pkl.load(file)
            ModelManager.data_type = data_type

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
    Выполняет предсказание с помощью активной модели.

    Parameters
    ----------
    request : PredictRequest
        Запрос, содержащий параметры для предсказания (количество периодов, данные и др.).

    Returns
    -------
    PredictResponse
        Результаты предсказания.

    Raises
    ------
    HTTPException
        Если модель не активна или возникает ошибка предсказания.
    """
    n_periods = int(request.n_periods)
    future_forecast = bool(getattr(request, 'future_forecast', True))
    ticker = request.ticker

    if not ModelManager.model:
        logger.error('Модель не активна.')
        raise HTTPException(
            status_code=404,
            detail='Модель не активна.'
        )

    try:
        if hasattr(ModelManager.model, 'predict') and hasattr(ModelManager.model, 'shrink'):
            data = getattr(request, 'data', None)
            data_type = ModelManager.data_type
            transformers = get_transformer_path(data_type=data_type)

            with open(transformers, 'rb') as file:
                transformers = pkl.load(file)

            if data is None:
                logger.error('Для CatBoost необходимо передать данные для предсказания.')
                raise HTTPException(
                    status_code=400,
                    detail='Для CatBoost необходимо передать данные для предсказания.'
                )

            df = pd.DataFrame({'value': data})
            df['Date'] = pd.date_range(start='2022-01-01', periods=len(data), freq='D')

            features = extract_time_series_features(df[['Date', 'value']])
            features['ticker'] = ticker
            features, _ = preprocess_time_series(df=features, target='target', transformers=transformers)

            max_periods = features.shape[0]
            n_periods = min(n_periods, max_periods)

            X = features.values[-n_periods:, 1:]

            forecast = ModelManager.model.predict(X)
            forecast = inverse_preprocess_time_series(ts_transformed=forecast, transformers=transformers)

            return PredictResponse(
                forecast=forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
            )

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

        return PredictResponse(
            forecast=forecast.tolist() if hasattr(forecast, 'tolist') else list(forecast)
        )

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
    Получение списка всех сохранённых моделей.

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
    Устанавливает активную модель для последующих операций.

    Parameters
    ----------
    request : SetRequest
        Запрос с идентификатором и типом данных модели для активации.

    Returns
    -------
    SetResponse
        Сообщение об успешной активации модели.

    Raises
    ------
    HTTPException
        Если модель не найдена.
    """
    model_id = request.id
    data_type = request.data_type

    if model_id == 'catboost_pretrained':
        model_type = 'catboost'
        model_path = get_model_path(model_id, model_type, data_type)

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
        with open(model_path, 'rb') as f:
            ModelManager.model = pkl.load(f)
            ModelManager.data_type = data_type

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
    Удаляет все сохранённые модели (кроме предобученной CatBoost).

    Returns
    -------
    RemoveResponse
        Сообщение об успешном удалении всех моделей.
    """
    removed = []
    for folder in ['models/auto_arima', 'models/holt_winters']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            removed.append(folder)

    ModelManager.model = None

    logger.info('Все модели удалены: %s', removed)

    return RemoveResponse(
        message='Все модели удалены (кроме предобученной CatBoost).'
    )
