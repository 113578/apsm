import os
import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from typing_extensions import Annotated
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima

# Настройка логирования
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/app.log"
LOG_DIR = os.path.dirname(LOG_FILE)

# Создаем папку для логов, если она не существует
os.makedirs(LOG_DIR, exist_ok=True)

# Создаем обработчик с ротацией логов
handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
handler.setFormatter(logging.Formatter(LOG_FORMAT))
handler.setLevel(logging.INFO)

# Конфигурируем корневой логгер
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger("FastAPI")

app = FastAPI()


class AutoARIMAPredictRequest(BaseModel):
    data: List[float]
    n_periods: int


class HoltWintersPredictRequest(BaseModel):
    data: List[float]
    n_periods: int
    trend: Optional[str] = None  # 'additive' or 'multiplicative'
    seasonal: Optional[str] = None  # 'additive', 'multiplicative', or None


@app.post("/predict/auto_arima", response_model=dict)
async def predict_auto_arima(
    request: Annotated[AutoARIMAPredictRequest, AutoARIMAPredictRequest]
) -> dict:
    try:
        logger.info("Received request for auto_arima prediction")
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
        logger.info("Successfully generated auto_arima forecast")
        return {"forecast": forecast.tolist()}

    except Exception as e:
        logger.error(f"Error in auto_arima prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/holt_winters", response_model=dict)
async def predict_holt_winters(
    request: Annotated[HoltWintersPredictRequest, HoltWintersPredictRequest]
) -> dict:
    try:
        logger.info("Received request for Holt-Winters prediction")
        data = request.data
        n_periods = request.n_periods
        trend = request.trend
        seasonal = request.seasonal

        data_series = pd.Series(data, index=pd.date_range(start="2023-01-01", periods=len(data), freq="D"))

        model = ExponentialSmoothing(
            endog=data_series,
            trend=trend,
            seasonal=seasonal,
            freq="D"
        ).fit()

        forecast = model.forecast(steps=n_periods)
        logger.info("Successfully generated Holt-Winters forecast")
        return {"forecast": forecast.tolist()}

    except Exception as e:
        logger.error(f"Error in Holt-Winters prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))