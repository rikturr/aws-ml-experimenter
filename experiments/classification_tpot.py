import logging
import scipy.sparse as sp
import pandas as pd
import importlib
import argparse
from multiprocessing import cpu_count

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_module", help="Config module")
    parser.add_argument("--experiment-id", help="Assign experiment ID")
    parser.add_argument("--bucket", help="Name of S3 bucket")

    return parser.parse_args()

# load config
args = parse_args()
config_module = args.config_module
logger.info('Using config: {}'.format(config_module))

config = importlib.import_module(config_module)
func = importlib.import_module('run_experiment')
bucket = args.bucket
experiment_id = args.experiment_id
random_state = config.random_state

# load files
logger.info('Loading ID/lookup file: {}'.format(config.labels_file))
labels_file = func.get_s3(bucket, config.labels_file)
labels = pd.read_csv(labels_file)

# get feature instances
logger.info('Loading feature instances: {}'.format(config.features_file))
x_file = func.get_s3(bucket, config.features_file)
X = sp.load_npz(x_file)

x = pd.DataFrame(X.toarray())
y = labels[config.label_col]

# run experiment
pipeline = config.classifier
pipeline.n_jobs = cpu_count()

logger.info('Starting TPOT optimizer')
pipeline.fit(x, y)

logger.info('TPOT finished, uploading best pipeline')
pipeline.export('{}.py'.format(experiment_id))
func.copy_s3('{}.py'.format(experiment_id), bucket, 'tpot/{}.py'.format(experiment_id))
