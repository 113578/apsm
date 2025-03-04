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


logger = setup_logger(
    name='streamlit',
    log_file=os.getenv('PYTHONPATH') + '/logs/streamlit.log'
)

base_url = os.getenv('STREAMLIT_BASE_URL', 'http://fastapi:8000')


def get_analytics(
    df: pd.DataFrame,
    template_type: str,
    selected_ticker: str
) -> None:
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
    trend: str,
    seasonal: str,
    seasonal_periods: int
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
            'ml_model_type': selected_model
        }
    }

    if selected_model == 'holt_winters':
        payload['config']['hyperparameters'] = {
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': int(seasonal_periods)
        }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)

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
    ticker: str
) -> plotly.graph_objs.Figure:
    """
    Генерация графика на основе данных.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для построения графика.
    ticker : str
        Название тикера для отображения.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        Объект графика для отображения.
    """
    fig = px.line(df, x=df.index, y=f'{ticker}', title=f'{ticker}')

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title_text=f'{ticker}'
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
            logger.error(
                'Ошибка при отправке запроса: %s',
                error_message
            )


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

        else:
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
    template_type: str
) -> pd.DataFrame:
    """
    Очистка загруженных данных в зависимости от типа шаблона.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для очистки.
    template_type : str
        Тип шаблона ('Котировки валют' или другой тип).

    Returns
    -------
    pd.DataFrame
        Очищенные данные.
    """
    if template_type == 'Котировки валют':
        df.dropna(inplace=True)
        df = df.filter(regex='Close', axis=1)
        df.columns = (col[: col.find('=')] for col in df.columns)
        cleaned_df = df.loc[:, (df == 0).sum() < 4][:-3]
    else:
        cleaned_df = df.loc[:, (df.isnull()).sum() < 115]
        cleaned_df.dropna(inplace=True)

    return cleaned_df


def upload_file(
    template_type: str
) -> Tuple[Union[pd.DataFrame, None], bool]:
    """
    Загрузка файла через интерфейс Streamlit и его обработка.

    Parameters
    ----------
    template_type : str
        Тип шаблона ('Котировки валют' или другой тип).

    Returns
    -------
    Tuple[Union[pd.DataFrame, None], bool]
        Возвращает очищенные данные и флаг успешной загрузки.
        Если файл не загружен, возвращает (None, False).
    """
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


def select_ticker(
    df: pd.DataFrame,
    template_type: str
) -> Union[str, None]:
    """
    Выбор тикера из предоставленных данных через боковую панель Streamlit.

    Parameters
    ----------
    df : pd.DataFrame
        Данные для выбора тикера.
    template_type : str
        Тип шаблона ('Котировки валют' или другой тип).

    Returns
    -------
    Union[str, None]
        Выбранный тикер, если он выбран, иначе None.
    """
    st.sidebar.header('Выбор тикера')
    options = df.columns

    search_term = st.sidebar.text_input(
        'Поиск:',
        placeholder=f"""
            Введите тикер {
                'валютной пары' if template_type == 'Котировки валют' else 'акции'
            }
        """
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
        filtered_options
    )

    if selected_option:
        st.write(f'Ваш выбор: {selected_option}')

        return selected_option

    return None


async def set_active_model(model_id: str) -> None:
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
        'id': model_id
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
    Выполнение обучения модели или прогнозирования в зависимости от типа шаблона.

    Parameters
    ----------
    template_type : str
        Тип операции ('fit' для обучения, 'predict' для прогнозирования).
    df : pd.DataFrame
        Данные для обработки.
    ticker : str, optional
        Название тикера для анализа (по умолчанию None).

    Returns
    -------
    None
        Выполняет задачу обучения или прогнозирования и отображает результаты.
    """
    if template_type == 'fit':
        st.header('Обучение модели 🔧')
        selected_model = st.selectbox(
            'Выберите модель:',
            ModelType,
            key=np.random.randint(10_000)
        )
        model_id = st.text_input(
            'ID:',
            placeholder='Введите ID модели:',
            key=np.random.randint(10_000)
        )

        seasonal_periods = None
        selected_period = selected_trend = selected_seasonal = None

        if selected_model == ModelType.holt_winters:
            selected_trend = st.selectbox(
                'Выберите тип трендовой компоненты:',
                ['add', 'mul'],
                index=None,
                key=np.random.randint(10_000)
            )
            selected_seasonal = st.selectbox(
                'Выберите тип сезонной компоненты:',
                ['add', 'mul'],
                index=None,
                key=np.random.randint(10_000)
            )
            seasonal_periods = st.text_input(
                'Сезонный период:',
                placeholder='Введите длину сезонного цикла:',
                key=np.random.randint(10_000)
            )

        if model_id:
            if st.button(
                'Обучить модель!',
                key=np.random.randint(10_000)
            ):
                await train_model(
                    df=df,
                    model_id=model_id,
                    selected_model=selected_model,
                    trend=selected_trend,
                    seasonal=selected_seasonal,
                    seasonal_periods=seasonal_periods
                    )
    else:
        st.header('Инференс модели 🔥')
        list_models = await get_list_models()

        if st.button(
            'Удалить все модели',
            key=np.random.randint(10_000)
        ):
            await delete_models()

        selected_model = st.selectbox(
            'Выберите модель:',
            list_models,
            key=np.random.randint(10_000)
        )
        if selected_model:
            selected_period = (
                st.text_input(
                    'Период (дни):',
                    placeholder='Введите период предсказания:',
                    key=np.random.randint(10_000)
                )
            )

            if st.button(
                'Предсказать!',
                key=np.random.randint(10_000)
            ) and selected_period:
                await set_active_model(model_id=selected_model)
                await inference_model(
                    df=df,
                    ticker=ticker,
                    period=int(selected_period)
                )


async def create_template(
    is_uploaded: bool,
    template_type: Literal['Котировки валют', 'Акции']
) -> None:
    """
    Создание шаблона приложения.

    Parameters
    ----------
    is_uploaded : bool
        Состояние загружаемого файла.
    template_type : Literal['Котировки валют', 'Акции']
        Тип шаблона ('Котировки валют' или 'Акции').

    Returns
    -------
    None
        Управляет загрузкой файла, выбором тикера и последующими действиями
        (обучение модели или прогнозирование).
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
                template_type=template_type,
                selected_ticker=selected_ticker
            )

            tab_fit, tab_predict = st.tabs(
                tabs=['Обучение', 'Прогнозирование']
            )

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
