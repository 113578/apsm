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


def get_model_path(model_id: str) -> Optional[str]:
    """
    Получения директории моделей.
    """
    auto_arima_path = os.path.join(
        'models', 'auto_arima', model_id + '.joblib'
    )
    holt_winters_path = os.path.join(
        'models', 'holt_winters', model_id + '.joblib'
    )

    if os.path.exists(auto_arima_path):
        return auto_arima_path

    if os.path.exists(holt_winters_path):
        return holt_winters_path

    return None
