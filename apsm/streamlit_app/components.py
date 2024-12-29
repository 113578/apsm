import asyncio

import aiohttp
import streamlit as st
import pandas as pd

from typing import Literal
import plotly.express as px
from pygments.lexers import go


# from apsm.streamlit_app.main import logger

base_url = "http://0.0.0.0:8000"


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred in {func.__name__}: {e}")
            # logger.error(f"An error occurred in {func.__name__}: {e}")

    return wrapper


def async_exception_handler(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            st.error(f"An error occurred in {func.__name__}: {e}")
            # logger.error(f"An error occurred in {func.__name__}: {e}")

    return wrapper


@exception_handler
def get_analytics(df, template_type, selected_option):
    """
    Получение аналитики по загруженному файлу.

    Parameters
    ----------
    df : pd.DataFrame
        Загруженный файл.

    Returns
    -------
    analytics : pd.DataFrame
        Кадр данных, содержащий аналитику.
    """

    fig = get_chart(df, selected_option)
    st.plotly_chart(fig)


@exception_handler
def train_model():
    pass


@exception_handler
def compare_experiments(selected_experiments):
    """
    Сравнение нескольких экспериментов с отображением кривых обучения.

    Parameters
    ----------
    selected_experiments : list
        Список выбранных экспериментов для сравнения.

    Returns
    -------
    None
    """

    # Simulated data for demonstration purposes
    fig = go.Figure()

    for experiment in selected_experiments:
        x = list(range(10))  # Epochs
        y = [val * (0.9 + 0.1 * (hash(experiment) % 3)) for val in range(10)]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=experiment))

    fig.update_layout(
        title="Кривые обучения",
        xaxis_title="Эпоха",
        yaxis_title="Значение метрики",
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)


@exception_handler
def get_chart(df, ticker):
    fig = px.line(df, x=df.index, y=f"{ticker}", title=f"{ticker}")

    fig.update_layout(
        xaxis_title="Date", yaxis_title="Price", legend_title_text=f"{ticker}"
    )
    return fig


@async_exception_handler
async def inference_model(df, ticker, model, hyperparameters):
    url = f"{base_url}/predict/auto_arima"
    payload = {
        "data": df[ticker].values.tolist(),
        "n_periods": hyperparameters["period"],
    }
    if model == "Holt Winters":
        url = f"{base_url}/predict/holt_winters"
        payload["trend"] = hyperparameters["trend"]
        payload["seasonal"] = hyperparameters["seasonal"]

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                predictions = await response.json()
                st.write("Результаты предсказания:", predictions)
                fig = get_chart(df, ticker)
                fig.add_scatter(
                    x=pd.date_range(
                        start=df.index[-1] + pd.DateOffset(days=1),
                        periods=hyperparameters["period"],
                        freq="D",
                    ),
                    y=predictions["forecast"],
                    mode="lines",
                    name="Predictions",
                )
                st.plotly_chart(fig)

            else:
                error_message = await response.text()
                st.error(f"Ошибка при отправке запроса: {error_message}")


@exception_handler
def clean_data(df, template_type):
    if template_type == "Котировки валют":
        df.dropna(inplace=True)
        df = df.filter(regex="Close", axis=1)
        df.columns = (col[: col.find("=")] for col in df.columns)
        cleaned_df = df.loc[:, (df == 0).sum() < 4][:-3]
    else:
        cleaned_df = df.loc[:, (df.isnull()).sum() < 115]
        cleaned_df.dropna(inplace=True)
    return cleaned_df


@exception_handler
def upload_file(template_type):
    if template_type == "Котировки валют":
        st.header("Загрузка котировок валют🔻")
    else:
        st.header("Загрузка котировок акций🔻")

    uploaded_file = st.file_uploader(label="Загрузите файл", key=template_type)
    if uploaded_file:
        df = pd.read_parquet(uploaded_file, engine="pyarrow")
        cleaned_df = clean_data(df, template_type)
        return cleaned_df, True
    return None, False


@exception_handler
def select_ticker(df, template_type):
    st.sidebar.header("Выбор тикера")
    options = df.columns
    search_term = st.sidebar.text_input(
        "Поиск:",
        placeholder=f"Введите тикер {
            'валютной пары' if template_type == 'Котировки валют'
            else 'акции'}",
    )

    filtered_options = [
        option for option in options if search_term.lower() in option.lower()
    ]

    selected_option = st.sidebar.selectbox(
        f"Выберите "
        f"{'валютную пару'if template_type == 'Котировки валют' else 'акцию'}",
        filtered_options,
    )

    if selected_option:
        st.sidebar.write(f"Ваш выбор: {selected_option}")
        return selected_option
    return None


@exception_handler
def select_model_and_hyperparameters():
    st.sidebar.header("Выбор модели и гиперпараметров")
    options = ["Auto ARIMA", "Holt Winters"]

    selected_model = st.sidebar.selectbox(f"Выберите модель:",
                                          options, index=None)
    selected_period, selected_trend, selected_seasonal = None, None, None
    if selected_model:
        selected_period = int(
            st.sidebar.text_input("Период:",
                                  placeholder="Введите период предсказания:")
        )
        if selected_model == "Holt Winters":
            selected_trend = st.sidebar.selectbox(
                f"Выберите тип трендовой компоненты:",
                ["additive", "multiplicative"],
                index=None,
            )
            selected_seasonal = st.sidebar.selectbox(
                f"Выберите тип сезонной компоненты:",
                ["additive", "multiplicative"],
                index=None,
            )

    return selected_model, {
        "period": selected_period,
        "trend": selected_trend,
        "seasonal": selected_seasonal,
    }


@exception_handler
def create_template(
    is_uploaded: bool, template_type: Literal["Котировки валют", "Акции"]
) -> None:
    """
    Создание шаблона приложения.

    Parameters
    ----------
    is_uploaded : bool
        Состояние загружаемого файла.
    template_type : str
        Тип шаблона
    """

    df, is_uploaded = upload_file(template_type)

    if is_uploaded:
        st.write(df)
        selected_ticker = select_ticker(df, template_type)

        if selected_ticker:
            st.header("Аналитика файла 📊")
            analytics = get_analytics(df, template_type, selected_ticker)

            model, hyperparameters = select_model_and_hyperparameters()
            if model and hyperparameters["period"]:
                st.header("Обучение модели 🔧")
                train_model()

                st.header("Инференс модели 🔥")
                asyncio.run(
                    inference_model(
                        df, selected_ticker, model, hyperparameters
                    )
                )
