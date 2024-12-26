import os
import logging
import logging.config


logging.config.fileConfig(os.getenv('PYTHONPATH') + '/config/logging/config')

logger = logging.getLogger()
