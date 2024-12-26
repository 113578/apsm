import os
import logging
import logging.config

path = os.getenv('PYTHONPATH') + '/config/logging/config.conf'
logging.config.fileConfig(path)

logger = logging.getLogger()
