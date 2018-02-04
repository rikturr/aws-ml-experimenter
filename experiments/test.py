import importlib
import logging
import sys

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
logger.addHandler(console_handler)

config_module = sys.argv[1]
logger.info('Using config: {}'.format(config_module))
config = importlib.import_module(config_module)

logger.info('Myvar = {}'.format(config.myvar))
