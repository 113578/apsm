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
from catboost import CatBoostRegressor
from apsm.app.model_manager import ModelManager
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
    ticker = request.ticker 
    config = request.config.hyperparameters or {}
    data_type = request.config.data_type

    model_path = get_model_path(model_id, ticker, model_type.value, data_type)

    if model_path:
        logger.error("Модель '%s' уже существует.", model_id)
        raise HTTPException(
            status_code=400,
            detail=f"Модель '{model_id}' уже существует."
        )

    try:
        ModelManager.data_type = data_type        
        
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
            features = ModelManager.transform_data(data, ticker)
            X, y = features.drop("target", axis = 1), features["target"]
            
            catboost_path = get_model_path(model_id, ticker, 'catboost', data_type)
            ModelManager.model = CatBoostRegressor(
                thread_count=-1,
                verbose=0
            ).fit(X, y)
            
        else:
            logger.error('Неподдерживаемый тип модели.')
            raise HTTPException(
                status_code=400,
                detail='Неподдерживаемый тип модели.'
            )

        os.makedirs(f'models/{model_type.value}', exist_ok=True)
        joblib.dump(
            {
                "model": ModelManager.model,
                "transformers": ModelManager.transformers if model_type == ModelType.catboost else None
            },
            f'models/{model_type.value}/{model_id}_{ticker}.joblib'
        )

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
            
            if data is None:
                logger.error('Для CatBoost необходимо передать данные для предсказания.')
                raise HTTPException(
                    status_code=400,
                    detail='Для CatBoost необходимо передать данные для предсказания.'
                )
    
            features = ModelManager.transform_data(data, ticker, is_train=False).drop(["target"], axis = 1)
            
            max_periods = features.shape[0]
            n_periods = min(n_periods, max_periods)

            X = features.values[-n_periods:]
            
            forecast = ModelManager.model.predict(X)
            forecast = inverse_preprocess_time_series(ts_transformed=forecast, transformers=ModelManager.transformers)

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
            'id': "".join(info.split('_')[:-1]),
            'type': model_type.value,
            'ticker': info.split('_')[-1].removesuffix('.joblib')
        }
        for model_type in ModelType
        if os.path.exists(f'models/{model_type.value}')
        for info in os.listdir(f'models/{model_type.value}')
    ]

    catboost_path = os.path.join('models', 'pretrained', 'classic', 'currency', 'cb.pkl')

    if os.path.exists(catboost_path):
        models_list.append({'id': 'catboost_pretrained', 'type': 'catboost', 'ticker':'unknown'})

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
    model_id, ticker = request.id, request.ticker
    ModelManager.data_type = request.data_type

    if model_id == 'catboost_pretrained':
        model_type = 'catboost'
        model_path = get_model_path(model_id, ticker, model_type, ModelManager.data_type)

    else:
        model_type = None
        model_path = get_model_path(model_id, ticker, model_type)

    if not model_path:
        logger.error("Модель '%s' не найдена.", model_id)

        raise HTTPException(
            status_code=404,
            detail=f"Модель '{model_id}' не найдена."
        )

    if model_type == 'catboost':
        with open(model_path, 'rb') as f:
            ModelManager.model = pkl.load(f)
            ModelManager.load_transformers()

    else:
        obj = joblib.load(model_path)
        ModelManager.model, ModelManager.transformers = obj.values()

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
    for folder in ['models/auto_arima', 'models/holt_winters', 'models/catboost']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            removed.append(folder)

    ModelManager.model = None

    logger.info('Все модели удалены: %s', removed)

    return RemoveResponse(
        message='Все модели удалены (кроме предобученной CatBoost).'
    )
