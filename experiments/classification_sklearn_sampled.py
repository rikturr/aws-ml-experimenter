from experiments.helpers import *
import logging
import os
import numpy as np
from sklearn.externals import joblib
import importlib
import argparse
from sklearn.pipeline import make_pipeline
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
data_path = config.data_path
x_path = config.x_path if hasattr(config, 'x_path') else None
y_path = config.y_path if hasattr(config, 'y_path') else None
local = config.local
name = config.name
bucket = config.bucket
sparse = hasattr(config, 'sparse') and config.sparse
save_model = hasattr(config, 'save_model') and config.save_model
metrics = hasattr(config, 'metrics') and config.metrics
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
    logger.warning('Running: {}'.format(metadata))

    if x_path:
        x_file_path = x_path
    else:
        x_file_path = 'size={s}/pos_ratio={pr}/{name}_train_X_{s}_{pr}_{r}.npy'.format(name=name, s=c['size'], pr=c['pos_ratio'], r=c['run'])
    x_file = os.path.join(data_path, x_file_path) if local else get_s3('{}/{}'.format(data_path, x_file_path), bucket=bucket)
    x = np.load(x_file)

    if y_path:
        y_file_path = y_path
    else:
        y_file_path = 'size={s}/pos_ratio={pr}/{name}_train_Y_{s}_{pr}_{r}.npy'.format(name=name, s=c['size'], pr=c['pos_ratio'], r=c['run'])
    y_file = os.path.join(data_path, y_file_path) if local else get_s3('{}/{}'.format(data_path, y_file_path), bucket=bucket)
    y = np.load(y_file)
    logger.warning('Loaded data files')

    model = make_pipeline(*c['model'])
    model.fit(x, y)

    logger.warning('Saving model: {}'.format(model_uuid))
    dump_path = 'size={s}/pos_ratio={pr}/ecbdl_{clf}_{s}_{pr}_{r}.pkl'.format(clf=metadata['model'], s=c['size'], pr=c['pos_ratio'], r=c['run'])
    if local:
        joblib.dump(model, os.path.join(path, 'models', dump_path))
    else:
        joblib_dump_s3(model, '{}/models/{}'.format(path, dump_path), bucket=bucket)
