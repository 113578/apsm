import os
import logging
import logging.config

from logging.handlers import RotatingFileHandler


logging.config.fileConfig(os.getenv('PYTHONPATH') + '/config/logging/config')

logger = logging.getLogger()
