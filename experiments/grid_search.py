from experiments.helpers import *
import logging
import os
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import GridSearchCV
import importlib
import sys
import pandas as pd
from datetime import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger('root')
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

RESULTS_STAGING = '/tmp/results'

# load config
config_module = sys.argv[1]
logger.warning('Using config: {}'.format(config_module))

config = importlib.import_module(config_module)
path = config.base_path
local = config.local
name = config.name
bucket = config.bucket
sparse = hasattr(config, 'sparse') and config.sparse
param_grid = config.param_grid
pipeline = config.pipeline
n_jobs = config.n_jobs
folds = config.folds

# load files
logger.warning('Loading data from: {}'.format(config.data_path))
x_train_file = config.x_train_file if local else get_s3(config.x_train_file, bucket=bucket)
if sparse:
    x_train = sp.load_npz(x_train_file)
else:
    x_train = np.load(x_train_file)

y_train_file = config.y_train_file if local else get_s3(config.y_train_file, bucket=bucket)
y_train = np.load(y_train_file)

# run experiment
logger.warning('Starting grid search')
grid = GridSearchCV(pipeline, param_grid, cv=folds, n_jobs=n_jobs, scoring='roc_auc', return_train_score=True, error_score=-1, verbose=2)
grid.fit(x_train, y_train)
results = pd.DataFrame(grid.cv_results_)

timestamp = datetime.now().isoformat().replace(":", "_").replace('.', '_')
results_path = os.path.join(path, 'grid_search', '{}_{}.csv'.format(name, timestamp))
logger.warning('Writing results file: {}'.format(results_path))
if local:
    results.to_csv(results_path, index=False)
else:
    to_csv_s3(results, results_path, bucket=bucket)
