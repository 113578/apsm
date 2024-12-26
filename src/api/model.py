import sys
sys.path.insert(0, '/home/boyarskikhae/Documents/Магистратура/1 курс/apsm')

import os
import pandas as pd

from typing_extensions import Annotated
from fastapi import FastAPI, APIRouter, HTTPException
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from config.logging.logger import logger
from config.dataclasses import AutoARIMAPredictRequest, HoltWintersPredictRequest


model_router = APIRouter()


@model_router.post('/predict/auto_arima', response_model=dict)
async def predict_auto_arima(request: Annotated[AutoARIMAPredictRequest, AutoARIMAPredictRequest]) -> dict:
    try:
        logger.info('Received request for auto_arima prediction')
        
        X_train = request.data
        n_periods = request.n_periods

        model = auto_arima(
            X_train,
            start_p=1,
            start_q=1,
            test='adf',
            max_p=3,
            max_q=3,
            seasonal=True,
            d=None,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        forecast = model.predict(n_periods=n_periods)
        
        logger.info('Successfully generated auto_arima forecast')
        
        return {'forecast': forecast.tolist()}

    except Exception as e:
        logger.error(f'Error in auto_arima prediction: {str(e)}')
        
        raise HTTPException(status_code=500, detail=str(e))


@model_router.post('/predict/holt_winters', response_model=dict)
async def predict_holt_winters(request: Annotated[HoltWintersPredictRequest, HoltWintersPredictRequest]) -> dict:
    try:
        logger.info('Received request for Holt-Winters prediction')
        
        data = request.data
        n_periods = request.n_periods
        trend = request.trend
        seasonal = request.seasonal

        data_series = pd.Series(data, index=pd.date_range(start='2023-01-01', periods=len(data), freq='D'))

        model = ExponentialSmoothing(
            endog=data_series,
            trend=trend,
            seasonal=seasonal,
            freq='D'
        ).fit()

        forecast = model.forecast(steps=n_periods)
        
        logger.info('Successfully generated Holt-Winters forecast')
        
        return {'forecast': forecast.tolist()}

    except Exception as e:
        logger.error(f'Error in Holt-Winters prediction: {str(e)}')
        
        raise HTTPException(status_code=500, detail=str(e))
        