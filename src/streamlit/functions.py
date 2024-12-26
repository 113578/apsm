import streamlit as st
import pandas as pd

from typing import Literal


def get_analytics():
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
    pass


def train_model():
    pass


def inference_model():
    pass


def create_template(
    is_uploaded: bool,
    template_type: Literal['Котировки валют', 'Акции']
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
    st.header('Загрузка данных 🔻')

    uploaded_file = st.file_uploader(
        label='Загрузите файл',
        key=template_type
    )
    if uploaded_file is not None:
        # df = pd.read_csv(uploaded_file)
        is_uploaded = True

    if is_uploaded:
        st.header('Аналитика файла 📊')
        analytics = get_analytics()

        st.header('Обучение модели 🔧')
        train_model()

        st.header('Инференс модели 🔥')
        inference_model()
