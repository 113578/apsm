import asyncio
import os
from enum import Enum
from http import HTTPStatus

import httpx
import streamlit as st
import pandas as pd
import plotly.express as px

from typing import Literal
from pygments.lexers import go


# from apsm.utils import setup_logger


# logger = setup_logger(
#     name='streamlit',
#     log_file=os.getenv('PYTHONPATH') + '/logs/streamlit.log'
# )

class ModelType(str, Enum):
    auto_arima = "auto_arima"
    holt_winters = "holt_winters"


base_url = 'http://127.0.0.1:8000'


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f'Про {func.__name__}: {e}')
            # logger.error(f'An error occurred in {func.__name__}: {e}')

    return wrapper


def async_exception_handler(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            st.error(f'An error occurred in {func.__name__}: {e}')
            # logger.error(f'An error occurred in {func.__name__}: {e}')

    return wrapper


@exception_handler
def get_analytics(df, template_type, selected_option):
    '''
    Получение аналитики по загруженному файлу.

    Parameters
    ----------
    df : pd.DataFrame
        Загруженный файл.

    Returns
    -------
    analytics : pd.DataFrame
        Кадр данных, содержащий аналитику.
    '''

    fig = get_figure(df, selected_option)
    st.plotly_chart(fig)


@exception_handler
async def train_model(df, model_id, selected_model,
                      trend, seasonal, seasonal_periods):
    url = f'{base_url}/fit'
    payload = {
        'data': df.values.tolist(),
        'config': {
            'id': model_id,
            'ml_model_type': selected_model
        }
    }

    if selected_model == 'holt_winters':
        payload["config"]['hyperparameters'] = {
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': int(seasonal_periods)
        }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

        if response.status_code == 201:
            message = response.json()["message"]
            st.write(message)
        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')


@exception_handler
def compare_experiments():
    pass


@exception_handler
def get_figure(df, ticker):
    fig = px.line(df, x=df.index, y=f'{ticker}', title=f'{ticker}')

    fig.update_layout(
        xaxis_title='Date', yaxis_title='Price', legend_title_text=f'{ticker}'
    )
    return fig


@async_exception_handler
async def inference_model(df, ticker, period):
    url = f'{base_url}/predict'
    payload = {
        'n_periods': period
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

        if response.status_code == 200:
            predictions = response.json()
            st.subheader('Результаты предсказания')
            fig = get_figure(df, ticker)
            fig.add_scatter(
                x=pd.date_range(
                    start=df.index[-1] + pd.DateOffset(days=1),
                    periods=period,
                    freq='D',
                ),
                y=predictions['forecast'],
                mode='lines',
                name='Predictions',
            )
            st.plotly_chart(fig)

        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')


@async_exception_handler
async def get_list_models():
    url = f'{base_url}/list'

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

        if response.status_code == 200:
            models = response.json()["models"]
            df = pd.DataFrame(models)
            st.table(df)
            return df
        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')


@async_exception_handler
async def delete_models():
    url = f'{base_url}/remove_all'

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

        if response.status_code == 200:
            st.write(response.json()["message"])
        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')


@exception_handler
def clean_data(df, template_type):
    if template_type == 'Котировки валют':
        df.dropna(inplace=True)
        df = df.filter(regex='Close', axis=1)
        df.columns = (col[: col.find('=')] for col in df.columns)
        cleaned_df = df.loc[:, (df == 0).sum() < 4][:-3]
    else:
        cleaned_df = df.loc[:, (df.isnull()).sum() < 115]
        cleaned_df.dropna(inplace=True)
    return cleaned_df


@exception_handler
def upload_file(template_type):
    if template_type == 'Котировки валют':
        st.header('Загрузка котировок валют🔻')
    else:
        st.header('Загрузка котировок акций🔻')

    uploaded_file = st.file_uploader(label='Загрузите файл', key=template_type)
    if uploaded_file:
        df = pd.read_parquet(uploaded_file, engine='pyarrow')
        cleaned_df = clean_data(df, template_type)
        return cleaned_df, True
    return None, False


@exception_handler
def select_ticker(df, template_type):
    st.sidebar.header('Выбор тикера')
    options = df.columns
    search_term = st.sidebar.text_input(
        'Поиск:',
        placeholder=f'Введите тикер {
            'валютной пары' if template_type == 'Котировки валют'
            else 'акции'}')

    filtered_options = [
        option for option in options if search_term.lower() in option.lower()
    ]

    selected_option = st.sidebar.selectbox(
        f'Выберите {'валютную пару' if template_type == 'Котировки валют'
                    else 'акцию'}', filtered_options
    )

    if selected_option:
        st.write(f'Ваш выбор: {selected_option}')
        return selected_option
    return None


@async_exception_handler
async def set_active_model(model_id):
    url = f'{base_url}/set'
    payload = {
        'id': model_id
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

        if response.status_code == 200:
            st.write(response.json()["message"])
        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')


@async_exception_handler
async def fit_or_predict(template_type, df, ticker=None):
    if template_type == "fit":
        st.header('Обучение модели 🔧')
        selected_model = st.selectbox(f'Выберите модель:',
                                      ModelType)
        model_id = st.text_input('Id:',
                                 placeholder='Введите id модели:')

        seasonal_periods = None
        selected_period, selected_trend, selected_seasonal = None, None, None

        if selected_model == ModelType.holt_winters:
            selected_trend = st.selectbox(
                f'Выберите тип трендовой компоненты:',
                ['additive', 'multiplicative'],
                index=None,
            )
            selected_seasonal = st.selectbox(
                f'Выберите тип сезонной компоненты:',
                ['additive', 'multiplicative'],
                index=None,
            )
            seasonal_periods = st.text_input('Сезонный период:',
                                             placeholder='Введите длину'
                                                         ' сезонного цикла:')
        if model_id:
            if st.button('Обучить модель!'):
                await train_model(df, model_id, selected_model,
                                  selected_trend, selected_seasonal,
                                  seasonal_periods
                                  )
    else:
        st.header('Инференс модели 🔥')
        list_models = await get_list_models()
        if st.button('Удалить все модели'):
            await delete_models()
        selected_model = st.selectbox(f'Выберите модель:', list_models)
        if selected_model:
            selected_period = (
                st.text_input('Период:',
                              placeholder='Введите период предсказания:')
            )

            if st.button('Предсказать!') and selected_period:
                await set_active_model(selected_model)
                await inference_model(df, ticker, int(selected_period))


@exception_handler
async def create_template(
        is_uploaded: bool, template_type: Literal['Котировки валют', 'Акции']
) -> None:
    '''
    Создание шаблона приложения.

    Parameters
    ----------
    is_uploaded : bool
        Состояние загружаемого файла.
    template_type : str
        Тип шаблона
    '''

    df, is_uploaded = upload_file(template_type)

    if is_uploaded:
        st.write(df)
        selected_ticker = select_ticker(df, template_type)

        if selected_ticker:
            st.header('Аналитика файла 📊')
            analytics = get_analytics(df, template_type, selected_ticker)
            tab_fit, tab_predict = st.tabs(tabs=['Обучение',
                                                 'Прогнозирование'])

            with tab_fit:
                await fit_or_predict(
                    template_type='fit',
                    df=df[selected_ticker]
                )

            with tab_predict:
                await fit_or_predict(
                    template_type='predict',
                    df=df,
                    ticker=selected_ticker
                )
