import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str):
    """
    Создаёт и настраивает логгер с именем name, который пишет в файл log_file и в консоль.

    Parameters
    ----------
    name : str
        Имя логгера.
    log_file : str
        Путь к файлу для логов.

    Returns
    -------
    logging.Logger
        Настроенный логгер.
    """
    os.makedirs(
        os.getenv('PYTHONPATH') + '/logs',
        exist_ok=True
    )

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,
        backupCount=5
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
