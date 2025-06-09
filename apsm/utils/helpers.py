import os
import sys

from typing import Optional


def create_project_path() -> None:
    """
    Создаёт переменную окружения PYTHONPATH и добавляет путь проекта в sys.path.
    Не возвращает значения, влияет на окружение процесса.
    """
    project_path = os.path.abspath(os.path.dirname(__file__) + '/../..')
    os.environ['PYTHONPATH'] = project_path
    sys.path.append(project_path)


def get_model_path(model_id: str, model_type: Optional[str] = None, data_type: Optional[str] = None) -> Optional[str]:
    """
    Возвращает путь к файлу модели по id и типу модели/данных, если файл существует.

    Parameters
    ----------
    model_id : str
        Идентификатор модели.
    model_type : Optional[str]
        Тип модели ('catboost', 'auto_arima', 'holt_winters').
    data_type : Optional[str]
        Тип данных ('Котировки валют' или 'Акции').

    Returns
    -------
    Optional[str]
        Путь к файлу модели или None, если не найден.
    """
    auto_arima_path = os.path.join('models', 'auto_arima', model_id + '.joblib')
    holt_winters_path = os.path.join('models', 'holt_winters', model_id + '.joblib')
    catboost_path = os.path.join('models', 'catboost', model_id + '.joblib')
    
    if model_type == 'catboost' and model_id == 'catboost_pretrained':
        catboost_path = None
        if data_type == 'Котировки валют':
            catboost_path = os.path.join('models', 'pretrained', 'classic', 'currency', 'cb.pkl')

        elif data_type == 'Акции':
            catboost_path = os.path.join('models', 'pretrained', 'classic', 'stock', 'cb.pkl')

        if catboost_path and os.path.exists(catboost_path):
            return catboost_path

        return None

    if os.path.exists(auto_arima_path):
        return auto_arima_path

    elif os.path.exists(holt_winters_path):
        return holt_winters_path
    
    elif os.path.exists(catboost_path):
        return catboost_path

    return None


def get_transformer_path(data_type: str) -> Optional[str]:
    """
    Возвращает путь к файлу трансформера по типу данных, если файл существует.

    Parameters
    ----------
    data_type : str
        Тип данных ('Котировки валют' или 'Акции').

    Returns
    -------
    Optional[str]
        Путь к файлу трансформера или None, если не найден.
    """
    base_path = os.path.join('models', 'transformers')
    if data_type == 'Котировки валют':
        transformer_path = os.path.join(base_path, 'currency.pkl')
    elif data_type == 'Акции':
        transformer_path = os.path.join(base_path, 'stock.pkl')
    else:
        return None

    if os.path.exists(transformer_path):
        return transformer_path

    return None
