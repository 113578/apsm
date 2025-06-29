import os
from typing import Literal, Union, Tuple

import httpx
import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from apsm.app.schemas import ModelType
from apsm.utils import setup_logger
import plotly.graph_objects as go

logger = setup_logger(
    name='streamlit',
    log_file=os.getenv('PYTHONPATH') + '/logs/streamlit.log'
)

base_url = os.getenv('STREAMLIT_BASE_URL', 'http://fastapi:8000')


def get_analytics(
    df: pd.DataFrame,
    selected_ticker: str
) -> None:
    '''
    Отображает статистику и график по выбранному тикеру из загруженного файла в интерфейсе Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        Загруженный файл с данными.
    selected_ticker : str
        Выбранный тикер для анализа.

    Returns
    -------
    None
        Отображает аналитику в Streamlit.
    '''
    st.subheader('Статистика кадра данных')
    a, b, c = st.columns(3)
    a.metric(label='Объём данных', value=df.shape[0])
    b.metric(label='Кол-во тикеров', value=df.shape[1])
    c.metric(
        label='Уникальных значений',
        value=df.nunique().sum()
    )

    st.subheader(f'{selected_ticker}')
    st.dataframe(df[selected_ticker].describe())

    st.subheader('Распределение тикера')
    fig = get_figure(
        df=df,
        ticker=selected_ticker
    )
    st.plotly_chart(fig)


async def train_model(
    df: pd.DataFrame,
    model_id: str,
    selected_model: str,
    data_type: str,
    ticker: str,
    trend: str = None,
    seasonal: str = None,
    seasonal_periods: int = None
) -> None:
    """
    Запуск обучения модели прогнозирования с заданными параметрами.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для обучения.
    model_id : str
        Уникальный идентификатор обучаемой модели.
    selected_model : str
        Тип модели, которая будет обучена (например, 'holt_winters').
    trend : str
        Трендовая компонента для модели Holt-Winters.
    seasonal : str
        Сезонная компонента для модели Holt-Winters.
    seasonal_periods : int
        Длина сезонного цикла.

    Returns
    -------
    None
        Отображает сообщения об успехе или ошибке в интерфейсе Streamlit.
    """
    url = f'{base_url}/fit'
    payload = {
        'data': df.values.tolist(),
        'config': {
            'id': model_id,
            'ml_model_type': selected_model,
            'data_type': data_type
        },
        'ticker': ticker
    }

    if selected_model == 'holt_winters':
        payload['config']['hyperparameters'] = {
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': int(seasonal_periods)
        }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, timeout=60.0)

        if response.status_code == 201:
            message = response.json()['message']
            st.write(message)
            logger.info(message)

        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')
            logger.error(
                'Ошибка при отправке запроса: %s',
                error_message
            )


def get_figure(
    df: pd.DataFrame,
    ticker: str,
    y_title: str = 'Price'
) -> plotly.graph_objs.Figure:
    """
    Генерирует график временного ряда по тикеру.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для построения графика.
    ticker : str
        Название тикера для отображения.
    y_title: str
        Название оси y.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        Объект графика для отображения.
    """
    df = df.iloc[:-1]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[ticker],
            name='Actual', 
            mode='lines'
        )
    )   
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=y_title
    )

    return fig


async def inference_model(
    df: pd.DataFrame,
    ticker: str,
    period: int
) -> None:
    """
    Проведение прогнозирования с использованием обученной модели.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для предсказания.
    ticker : str
        Название тикера для анализа.
    period : int
        Период прогнозирования в днях.

    Returns
    -------
    None
        Отображает график предсказаний или сообщение об ошибке в интерфейсе Streamlit.
    """
    url = f'{base_url}/predict'
    payload = {
        'n_periods': period,
        'data': df[ticker].values.tolist(),
        'ticker': ticker
    }

    start = df.index[-1] + pd.DateOffset(days=1)

    async with httpx.AsyncClient() as client:
        predictions = None
        for i in range(2):
            if i == 1:
                period = df.shape[0]
                start = df.index[0]
                payload['n_periods'] = period
                payload['future_forecast'] = True
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                predictions = response.json()['forecast']
                st.subheader('Результаты предсказания' + (" на обучающей выборке" if i == 1 else ""))
                
                
                df_cut = df[df.shape[0] - len(predictions):] if i == 1 else df
                df_index = df_cut.index if i == 1 else pd.date_range(start=start, periods=period, freq='D')
                    
                fig = get_figure(df_cut, ticker)
                
                fig.add_scatter(
                    x=df_index,
                    y=predictions,
                    mode='lines',
                    name='Predictions',
                )
                st.plotly_chart(fig)
            else:
                error_message = response.text
                st.error(f'Ошибка при отправке запроса: {error_message}')
                logger.error(
                    'Ошибка при отправке запроса: %s',
                    error_message
                )

        if predictions is not None:
            st.subheader('График остатков')
            actual = df[ticker]
            min_len = min(len(actual), len(predictions))
            actual = actual.iloc[-min_len:]
            predictions = np.array(predictions)[-min_len:]
            residuals = actual - predictions
            residuals = pd.DataFrame(residuals)
            fig = get_figure(residuals, ticker, 'residuals')
            st.plotly_chart(fig)


async def get_list_models() -> pd.DataFrame:
    """
    Получение списка доступных моделей.

    Returns
    -------
    pd.DataFrame
        Таблица с информацией о моделях.
    """
    url = f'{base_url}/list'

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

        if response.status_code == 200:
            models = response.json()['models']
            df = pd.DataFrame(models)
            st.table(df)

            return df

        error_message = response.text
        st.error(f'Ошибка при отправке запроса: {error_message}')
        logger.error(
            'Ошибка при отправке запроса: %s',
            error_message
        )


async def delete_models() -> None:
    """
    Удаляет все сохраненные модели на сервере.

    Returns
    -------
    None
        Выводит результат операции в Streamlit.
    """
    url = f'{base_url}/remove_all'

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

        if response.status_code == 200:
            st.write(response.json()['message'])
        else:
            error_message = response.text
            st.error(f'Ошибка при отправке запроса: {error_message}')
            logger.error(
                'Ошибка при отправке запроса: %s',
                error_message
            )


def clean_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Очищает DataFrame: удаляет столбцы и строки с большим количеством пропусков.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для очистки.

    Returns
    -------
    pd.DataFrame
        Очищенные данные.
    """
    half_cols = len(df) // 2
    cols_to_drop = [col for col in df.columns if df[col].isna().sum() > half_cols]
    df = df.drop(columns=cols_to_drop)

    half_rows = len(df.columns) // 2
    rows_to_drop = df[df.isna().sum(axis=1) > half_rows].index
    return df.drop(index=rows_to_drop)


def upload_file(
    template_type: str
) -> Tuple[Union[pd.DataFrame, None], bool]:
    """
    Загружает parquet-файл через Streamlit и очищает его.

    Parameters
    ----------
    template_type : str
        Тип шаблона ('Котировки валют' или другой тип).

    Returns
    -------
    Tuple[Union[pd.DataFrame, None], bool]
        Очищенные данные и флаг успешной загрузки. Если файл не загружен, возвращает (None, False).
    """
    if template_type == 'Котировки валют':
        st.header('Загрузка котировок валют🔻')
    else:
        st.header('Загрузка котировок акций🔻')

    uploaded_file = st.file_uploader(label='Загрузите файл', key=template_type)
    if uploaded_file:
        df = pd.read_parquet(uploaded_file, engine='pyarrow')
        cleaned_df = clean_data(df)
        return cleaned_df, True

    return None, False


def select_ticker(
    df: pd.DataFrame,
    template_type: str
) -> Union[str, None]:
    """
    Позволяет выбрать тикер из DataFrame через боковую панель Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для выбора тикера.
    template_type : str
        Тип шаблона ('Котировки валют' или другой тип).

    Returns
    -------
    Union[str, None]
        Выбранный тикер или None, если не выбран.
    """
    is_currency = int(template_type == 'Котировки валют')
    st.sidebar.header('Выбор тикера')
    options = df.columns
    search_term = st.sidebar.text_input(
        'Поиск:',
        placeholder=f"""
            Введите тикер {
                'валютной пары' if template_type == 'Котировки валют' else 'акции'
            }
        """,
        key=f"search_term{is_currency}"
    )

    filtered_options = [
        option for option in options
        if search_term.lower() in option.lower()
    ]

    selected_option = st.sidebar.selectbox(
        f"""
            Выберите {
                'валютную пару' if template_type == 'Котировки валют' else 'акцию'
            }
        """,
        filtered_options,
        key=f"selected_option{is_currency}"
    )

    if selected_option:
        st.write(f'Ваш выбор: {selected_option}')

        return selected_option

    return None


async def set_active_model(model_id: str, ticker: str, data_type: str) -> None:
    """
    Установка активной модели на сервере.

    Parameters
    ----------
    model_id : str
        Идентификатор модели, которая будет установлена как активная.

    Returns
    -------
    None
        Отображает сообщение об успехе или ошибке в интерфейсе Streamlit.
    """
    url = f'{base_url}/set'
    payload = {
        'id': model_id,
        'ticker': ticker,
        'data_type': data_type
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

        if response.status_code == 200:
            message = response.json()['message']

            st.write(message)
            logger.info(message)
        else:
            error_message = response.text

            st.error(f'Ошибка при отправке запроса: {error_message}')
            logger.error(
                'Ошибка при отправке запроса: %s',
                error_message
            )


async def fit_or_predict(
    template_type: str,
    df: pd.DataFrame,
    ticker: str = None
):
    """
    В зависимости от типа шаблона запускает обучение модели или прогнозирование через Streamlit-интерфейс.

    Parameters
    ----------
    template_type : str
        Тип операции ('fit_...' для обучения, 'predict_...' для прогнозирования).
    df : pd.DataFrame
        Данные для обработки.
    ticker : str, optional
        Название тикера для анализа (по умолчанию None).

    Returns
    -------
    None
        Отображает результаты в Streamlit.
    """
    data_type = template_type.split('_')[1]
    template_type, is_currency = template_type.split('_')[0], int(template_type.split('_')[1] == 'Котировки валют')

    if template_type == 'fit':
        st.header('Обучение модели 🔧')
        selected_model = st.selectbox(
            'Выберите модель:',
            ModelType,
            key=f"selected_model{is_currency}"
        )
        model_id = st.text_input(
            'ID:',
            placeholder='Введите ID модели:',
            key=f"model_id{is_currency}"
        )

        seasonal_periods = None
        selected_period = selected_trend = selected_seasonal = None

        if selected_model == ModelType.holt_winters:
            selected_trend = st.selectbox(
                'Выберите тип трендовой компоненты:',
                ['add', 'mul'],
                index=None,
                key=f"selected_trend{is_currency}"
            )
            selected_seasonal = st.selectbox(
                'Выберите тип сезонной компоненты:',
                ['add', 'mul'],
                index=None,
                key=f"selected_seasonal{is_currency}"
            )
            seasonal_periods = st.text_input(
                'Сезонный период:',
                placeholder='Введите длину сезонного цикла:',
                key=f"seasonal_periods{is_currency}"
            )

        if model_id:
            if st.button(
                'Обучить модель!',
                key=f"fit_model{is_currency}"
            ):
                await train_model(
                    df=df[:int(len(df)*0.7)],
                    model_id=model_id,
                    data_type=data_type,
                    selected_model=selected_model,
                    trend=selected_trend,
                    seasonal=selected_seasonal,
                    seasonal_periods=seasonal_periods,
                    ticker=ticker
                )
    else:
        st.header('Инференс модели 🔥')
        list_models = await get_list_models()

        if st.button(
            'Удалить все модели',
            key=f"delete_models{is_currency}"
        ):
            await delete_models()
            st.rerun()

        selected_model = st.selectbox(
            'Выберите модель:',
            list_models,
            key=f"select_model{is_currency}"
        )
        if selected_model:
            selected_period = (
                st.text_input(
                    'Период (дни):',
                    placeholder='Введите период предсказания:',
                    key=f"period{is_currency}"
                )
            )
            
            if st.button(
                'Предсказать!',
                key=f"predict{is_currency}"
            ) and selected_period:
                await set_active_model(selected_model, ticker, data_type)
                await inference_model(
                    df=df[int(len(df)*0.7):] if selected_model != "catboost_pretrained" else df,
                    ticker=ticker,
                    period=int(selected_period)
                )


async def create_template(
    is_uploaded: bool,
    template_type: Literal['Котировки валют', 'Акции']
) -> None:
    """
    Управляет загрузкой файла, выбором тикера и отображением аналитики, обучения и прогнозирования в Streamlit.

    Parameters
    ----------
    is_uploaded : bool
        Состояние загружаемого файла.
    template_type : Literal['Котировки валют', 'Акции']
        Тип шаблона ('Котировки валют' или 'Акции').

    Returns
    -------
    None
        Управляет логикой пользовательского интерфейса Streamlit.
    """
    df, is_uploaded = upload_file(template_type=template_type)

    if is_uploaded:
        st.write(df)
        selected_ticker = select_ticker(
            df=df,
            template_type=template_type
        )

        if selected_ticker:
            st.header('Аналитика файла 📊')
            get_analytics(
                df=df,
                selected_ticker=selected_ticker
            )

            tab_fit, tab_predict = st.tabs(
                tabs=['Обучение', 'Прогнозирование']
            )

            with tab_fit:
                await fit_or_predict(
                    template_type=f'fit_{template_type}',
                    df=df[selected_ticker],
                    ticker=selected_ticker
                )

            with tab_predict:
                await fit_or_predict(
                    template_type=f'predict_{template_type}',
                    df=df,
                    ticker=selected_ticker
                )
