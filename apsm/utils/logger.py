import logging
from logging.handlers import RotatingFileHandler


def setup_logger(name: str, log_file: str):
    """
    Установка логгера с именем и названием файла с логами.

    Parameters
    ----------
    name : str
        Имя логгера.
    log_file : str
        Название файла с логами.
    """
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
