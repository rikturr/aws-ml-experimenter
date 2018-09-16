from experiments.helpers import *
import logging
import os
import scipy.sparse as sp
import numpy as np
from sklearn.externals import joblib
import importlib
import argparse
from imblearn.pipeline import make_pipeline
import uuid
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_module", help="Config module")
    parser.add_argument("config_num", type=int, help="Which iteration of the config to use")
    parser.add_argument("--experiment-id", help="Assign experiment ID")

    return parser.parse_args()


args = parse_args()
config_module = args.config_module
logger.warning('Using config: {}'.format(config_module))
config = importlib.import_module(config_module)
config_num = args.config_num
config_name = config.configs[config_num]['name']
ml_config = config.configs[config_num]['ml_config']

path = config.base_path
local = config.local
name = config.name
n_jobs = config.n_jobs
bucket = config.bucket
sparse = hasattr(config, 'sparse') and config.sparse
save_model = hasattr(config, 'save_model') and config.save_model
metrics = hasattr(config, 'metrics') and config.metrics
runs = config.runs if hasattr(config, 'runs') else 1
folds = config.folds if hasattr(config, 'folds') else 5
experiment_id = args.experiment_id or uuid.uuid4()

instance_id = None
instance_type = None
if local:
    logger.warning('Local mode - not recording EC2 instance-id')
else:
    try:
        instance_id = subprocess.check_output('curl -m 30 http://169.254.169.254/latest/meta-data/instance-id', shell=True).decode('utf-8')
        instance_type = subprocess.check_output('curl -m 30 http://169.254.169.254/latest/meta-data/instance-type', shell=True).decode('utf-8')
    except Exception:
        logger.warning('Could not get EC2 instance-id or instance-type')

# load files
logger.warning('Loading data from: {}'.format(config.data_path))
x_train_file = config.x_train_file if local else get_s3(config.x_train_file, bucket=bucket)
if sparse:
    x_train = sp.load_npz(x_train_file)
else:
    x_train = np.load(x_train_file)

y_train_file = config.y_train_file if local else get_s3(config.y_train_file, bucket=bucket)
y_train = np.load(y_train_file)

# prepare pre-sampled sets
x_train_neg = None
y_train_neg = None
x_train_pos = {}
y_train_pos = {}
if 'reduce_pos_samples' in ml_config and ml_config['reduce_pos_samples']:
    if 'reduce_pos_runs' in ml_config and ml_config['reduce_pos_runs']:
        if len(ml_config['reduce_pos_runs']) > 1:
            raise ValueError('Only support one setting of reduce_pos_runs or reduce_pos_samples')

        neg_idx = np.nonzero(y_train == 0)[0]
        x_train_neg = x_train[neg_idx, :]
        y_train_neg = y_train[neg_idx]

        for num_pos in ml_config['reduce_pos_samples']:
            logger.warning('Starting pre-sampling for {}'.format(num_pos))
            x_train_pos[num_pos] = []
            y_train_pos[num_pos] = []
            for x in range(ml_config['reduce_pos_runs'][0]):
                logger.warning('Pre-sampling dataset {}'.format(x))
                np.random.seed(config.random_state + x)
                pos_idx = np.random.choice(np.nonzero(y_train == 1)[0], num_pos, replace=False)
                x_train_pos[num_pos].append(x_train[pos_idx, :])
                y_train_pos[num_pos].append(y_train[pos_idx])

    else:
        raise ValueError('Need to specify both reduce_pos_sample and reduce_pos_runs')

# run experiment
for c in dict_product(ml_config):
    metadata = {k: get_object_name(v) for k, v in c.items()}
    metadata['experiment_id'] = experiment_id
    metadata['model'] = get_object_name(c['model'][-1])
    model_uuid = uuid.uuid4()
    metadata['model_uuid'] = model_uuid
    metadata['config'] = '{}.{}'.format(config_module, config_name)
    metadata['instance_id'] = instance_id
    metadata['instance_type'] = instance_type
    metadata['dataset'] = name
    metadata['folds'] = folds
    logger.warning('Running: {}'.format(metadata))

    pipeline = []

    if 'oversampling' in c and c['oversampling']:
        if 'over_pos_samples' in c and c['over_pos_samples']:
            oversampling = c['oversampling']
            pipeline += [oversampling(c['over_pos_samples'], random_state=config.random_state)]
        else:
            raise ValueError('Need over_pos_samples if setting oversampling')

    if 'sampling_ratio' in c and c['sampling_ratio']:
        pipeline += [RatioRandomUnderSampler(c['sampling_ratio'], random_state=config.random_state)]

    pipeline.extend(c['model'])
    model = make_pipeline(*pipeline)
    print(model)

    if save_model:
        model.fit(x_train[:, config.feat_start_idx:], y_train)

        logger.warning('Saving model: {}'.format(model_uuid))
        dump_path = os.path.join(path, 'models', '{}.pkl'.format(model_uuid))
        if local:
            joblib.dump(model, dump_path)
        else:
            joblib_dump_s3(model, dump_path, bucket=bucket)

    # cross-validation
    logger.warning('Running cross-validation')

    results = None
    # pre sampling of positive instances (for class imbalance testing)
    if 'reduce_pos_samples' in c and c['reduce_pos_samples']:
        if 'reduce_pos_runs' in c and c['reduce_pos_runs']:
            reduce_results = []
            for x in range(c['reduce_pos_runs']):
                metadata['reduce_pos_run'] = x
                if sparse:
                    x_samp = sp.vstack([x_train_neg, x_train_pos[c['reduce_pos_samples']][x]])
                else:
                    x_samp = np.vstack([x_train_neg, x_train_pos[c['reduce_pos_samples']][x]])
                y_samp = np.append(y_train_neg, y_train_pos[c['reduce_pos_samples']][x])

                reduce_results.append(cross_validate_repeat(model, x_samp, y=y_samp, scoring=metrics, runs=1, n_jobs=n_jobs, folds=folds,
                                                            random_state=config.random_state, metadata=metadata))
            results = pd.concat(reduce_results)
        else:
            raise ValueError('Need to specify both reduce_pos_sample and reduce_pos_runs')
    else:
        results = cross_validate_repeat(model, x_train, y=y_train, scoring=metrics, runs=runs, n_jobs=n_jobs, folds=folds,
                                        random_state=config.random_state, metadata=metadata)

    timestamp = datetime.now()
    results_path = os.path.join(path, 'cv_results', '{}_{}.csv'.format(name, timestamp.isoformat().replace(":", "_").replace('.', '_')))
    logger.warning('Saving results: {}'.format(results_path))
    if local:
        results.to_csv(results_path, index=False)
    else:
        to_csv_s3(results, results_path, bucket=bucket)

    # save to combined CSV
    combined_results_path = os.path.join(path, 'cv_results_combined', '{}.csv'.format(name))
    if file_exists(combined_results_path, bucket=bucket, local=local):
        current_results = pd.read_csv(combined_results_path if local else get_s3(combined_results_path, bucket=bucket))
        new_results = pd.concat([current_results, results])
    else:
        new_results = results

    if local:
        new_results.to_csv(combined_results_path, index=False)
    else:
        to_csv_s3(new_results, combined_results_path, bucket=bucket)
