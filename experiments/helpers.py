import boto3
import io
from sklearn.externals import joblib
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import keras
import time
import json

s3 = boto3.resource('s3')


def sparse_generator(x, y=None, batch_size=32):
    index = np.arange(x.shape[0])
    start = 0
    while True:
        if start == 0 and y is not None:
            np.random.shuffle(index)

        batch = index[start:start + batch_size]

        if y is not None:
            yield x[batch].toarray(), y[batch].toarray()
        else:
            yield x[batch].toarray()

        start += batch_size
        if start >= x.shape[0]:
            start = 0


class S3Checkpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, s3_resource, bucket, s3_folder,
                 monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(S3Checkpoint, self).__init__(filepath, monitor=monitor, verbose=verbose,
                                           save_best_only=save_best_only, save_weights_only=save_weights_only,
                                           mode=mode, period=period)
        self.s3_resource = s3_resource
        self.bucket = s3_resource.Bucket(bucket)
        self.s3_folder = s3_folder

    def on_epoch_end(self, epoch, logs=None):
        super(S3Checkpoint, self).on_epoch_end(epoch, logs)
        if self.epochs_since_last_save == 0:
            local_filepath = self.filepath.format(epoch=epoch + 1, **logs)
            self.bucket.upload_file(local_filepath, os.path.join(self.s3_folder, os.path.basename(local_filepath)))


class S3HistoryLogger(keras.callbacks.Callback):
    def __init__(self, s3_resource, bucket, model_id, history_folder):
        super(S3HistoryLogger, self).__init__()
        self.s3_resource = s3_resource
        self.bucket = bucket
        self.model_id = model_id
        self.history_folder = history_folder

    def to_csv_s3(self, df, key, index=False):
        buf = io.StringIO()
        df.to_csv(buf, index=index)
        self.s3_resource.Object(self.bucket, key).put(Body=buf.getvalue())

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}
        self.time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.epoch_start_time

        logs = logs or {}
        self.epoch.append(epoch)

        # get history - see keras.callbacks.History
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        # add extra stuff
        self.history.setdefault('epoch', []).append(epoch)
        self.history.setdefault('elapsed_time', []).append(elapsed_time)
        self.history.setdefault('model_id', []).append(self.model_id)

        # save to s3
        self.to_csv_s3(pd.DataFrame(self.history), os.path.join(self.history_folder, '{}.csv'.format(self.model_id)))


def get_s3(bucket, key):
    obj = s3.Object(bucket, key)
    return io.BytesIO(obj.get()['Body'].read())


def delete_recursive_s3(bucket, key):
    objects_to_delete = s3.meta.client.list_objects(Bucket=bucket, Prefix=key)

    delete_keys = {'Objects': [{'Key': k} for k in [obj['Key'] for obj in objects_to_delete.get('Contents', [])]]}

    if delete_keys['Objects']:
        s3.meta.client.delete_objects(Bucket=bucket, Delete=delete_keys)


def to_csv_s3(df, bucket, key, index=False):
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def np_save_s3(x, bucket, key):
    buf = io.StringIO()
    np.save(buf, x)
    s3.Object(bucket, '{}.npy'.format(key)).put(Body=buf.getvalue())


def sp_save_s3(x, bucket, key):
    buf = io.StringIO()
    sp.save_npz(buf, x)
    s3.Object(bucket, '{}.npz'.format(key)).put(Body=buf.getvalue())


def joblib_dump_s3(obj, bucket, key):
    buf = io.StringIO()
    joblib.dump(obj, buf)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def json_dump_s3(obj, bucket, key):
    buf = io.StringIO()
    json.dump(obj, buf, indent=4)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def copy_dir_s3(path, bucket, key):
    for f in os.listdir(path):
        s3.Bucket(bucket).upload_file(os.path.join(path, f), os.path.join(key, f))
