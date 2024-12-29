import os
import sys


def create_project_path() -> None:
    """
    Создание директории проекта.
    """
    project_path = os.path.abspath(os.path.dirname(__file__) + '/../..')
    os.environ['PYTHONPATH'] = project_path
    sys.path.append(project_path)
