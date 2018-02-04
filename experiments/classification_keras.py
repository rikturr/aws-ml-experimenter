import logging
import scipy.sparse as sp
import pandas as pd
import importlib
import argparse
from keras.utils import to_categorical
from keras.callbacks import *
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

logger = logging.getLogger('root')
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
logger.addHandler(console_handler)

CHECKPOINT_PATH = '/tmp/checkpoints'


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

if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)

# load files
logger.info('Loading ID/lookup file: {}'.format(config.labels_file))
labels_file = func.get_s3(bucket, config.labels_file)
labels = pd.read_csv(labels_file)

# get feature instances
logger.info('Loading feature instances: {}'.format(config.features_file))
x_file = func.get_s3(bucket, config.features_file)
X = sp.load_npz(x_file)

Y = to_categorical(labels[config.label_col].astype('category').cat.codes)

pipeline = Pipeline([('scale', MaxAbsScaler()), ('zero_var', VarianceThreshold(0))])
preprocessed = pipeline.fit_transform(X)
x_train, x_val, y_train, y_val = train_test_split(preprocessed, Y, test_size=0.1, random_state=random_state)

# run experiment
model = config.model(X.shape[1], Y.shape[1])
model.compile(optimizer=config.optimizer, loss=config.loss, metrics=config.metrics)

# setup callbacks
checkpointer = func.S3Checkpoint(filepath=os.path.join(CHECKPOINT_PATH, '{}_'.format(experiment_id) + '{epoch}.h5'),
                                 s3_resource=func.s3,
                                 bucket=bucket,
                                 s3_folder='keras_checkpoints')
history_logger = func.S3HistoryLogger(
    s3_resource=func.s3,
    bucket=bucket,
    model_id=experiment_id,
    history_folder='keras_history'
)
tensorboard = TensorBoard(log_dir='/tmp/tensorboard/{}'.format(experiment_id))
callbacks = [checkpointer, tensorboard, history_logger]

# fit model
model.fit_generator(func.sparse_generator(x_train, y_train, config.batch_size),
                    epochs=config.epochs,
                    steps_per_epoch=-0 - - x_train.shape[0] / config.batch_size,
                    validation_data=func.sparse_generator(x_val, y_val, config.batch_size),
                    validation_steps=-0 - - x_val.shape[0] / config.batch_size,
                    callbacks=callbacks)

logger.info('DONE!!')
