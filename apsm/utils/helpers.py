import os
import sys

from typing import Optional


def create_project_path() -> None:
    """
    Создание директории проекта.
    """
    project_path = os.path.abspath(os.path.dirname(__file__) + '/../..')
    os.environ['PYTHONPATH'] = project_path
    sys.path.append(project_path)


def get_model_path(model_id: str, model_type: Optional[str] = None, data_type: Optional[str] = None) -> Optional[str]:
    """
    Получение пути к модели с учетом типа данных (currency/stock) для catboost.
    """
    auto_arima_path = os.path.join('models', 'auto_arima', model_id + '.joblib')
    holt_winters_path = os.path.join('models', 'holt_winters', model_id + '.joblib')

    if model_type == 'catboost':
        catboost_path = None
        if data_type == 'currency':
            catboost_path = os.path.join('models', 'pretrained', 'classic', 'currency', 'cb.pkl')

        elif data_type == 'stock':
            catboost_path = os.path.join('models', 'pretrained', 'classic', 'stock', 'cb.pkl')

        if catboost_path and os.path.exists(catboost_path):
            return catboost_path

        return None

    if os.path.exists(auto_arima_path):
        return auto_arima_path

    if os.path.exists(holt_winters_path):
        return holt_winters_path

    return None
