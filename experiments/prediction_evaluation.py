from experiments.helpers import *
import logging
import os
from sklearn.externals import joblib
import importlib
import argparse
import uuid

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
x_test_file = config.x_test_file
y_test_file = config.y_test_file
local = config.local
name = config.name
bucket = config.bucket
class_col = config.class_col if hasattr(config, 'class_col') else None
experiment_id = args.experiment_id or uuid.uuid4()
threshold_metrics = config.threshold_metrics
metrics = config.metrics
temp_path = config.temp_path if hasattr(config, 'temp_path') else None

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

# load data
logger.warning('Loading data files from {}, {}'.format(x_test_file, y_test_file))
x_file = x_test_file if local else get_s3(x_test_file, bucket=bucket)
x = np.load(x_file)

y_file = y_test_file if local else get_s3(y_test_file, bucket=bucket)
y = np.load(y_file)


def make_predict(model, x, actual, threshold=0.5):
    pred = model.predict_proba(x)
    pred_df = pd.DataFrame(pred)
    pred_df['actual'] = actual
    pred_df['predicted'] = (pred_df[1] > threshold).astype(int)
    return pred_df


def calculate_metrics(pred_df):
    results = {}
    for m in metrics:
        results[m.__name__] = m(pred_df['actual'], pred_df[1])

    for m in threshold_metrics:
        results[m.__name__] = m(pred_df['actual'], pred_df['predicted'])

    return results


# run experiment
for c in dict_product(ml_config):
    metadata = {k: get_object_name(v) for k, v in c.items()}
    metadata['experiment_id'] = experiment_id
    model_uuid = uuid.uuid4()
    metadata['config'] = config_module
    metadata['instance_id'] = instance_id
    metadata['instance_type'] = instance_type
    logger.warning('Running: {}'.format(metadata))

    model_path = 'size={s}/pos_ratio={pr}/{name}_{clf}_{s}_{pr}_{r}.pkl'.format(name=name, clf=metadata['model'], s=c['size'], pr=c['pos_ratio'], r=c['run'])
    if local:
        model_file = os.path.join(path, 'models', model_path)
    else:
        s3_path = '{}/models/{}'.format(path, model_path)
        logger.warning('Loading model from: {}'.format(s3_path))
        model_file = get_s3(s3_path, bucket=bucket)
    model = joblib.load(model_file)
    logger.warning('Loaded model {}'.format(model))

    # predict
    pred_df = make_predict(model, x, y)
    logger.warning('Got predictions')

    # metrics
    results_metrics = calculate_metrics(pred_df)
    metadata.update(results_metrics)
    results = pd.DataFrame(metadata, index=[0])
    logger.warning('Got metrics: {}'.format(results_metrics))

    # save to combined CSV
    combined_results_path = os.path.join(path, 'test_results', '{}.csv'.format(name))
    if file_exists(combined_results_path, bucket=bucket, local=local):
        current_results = pd.read_csv(combined_results_path if local else get_s3(combined_results_path, bucket=bucket))
        new_results = pd.concat([current_results, results])
    else:
        new_results = results

    if local:
        new_results.to_csv(combined_results_path, index=False)
    else:
        to_csv_s3(new_results, combined_results_path, bucket=bucket)

