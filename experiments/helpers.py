import itertools
import boto3
import io
from sklearn.externals import joblib
import subprocess
import numpy as np
import scipy.sparse as sp
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import os
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, cross_validate
from sklearn.feature_selection.univariate_selection import SelectKBest
import json
import warnings
import time
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_selection import chi2
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import sparsefuncs as spf


warnings.simplefilter(action='ignore', category=FutureWarning)
import keras

warnings.resetwarnings()

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')


def get_object_name(x):
    if x is None:
        return None
    if callable(x):
        return x.__name__
    if hasattr(x, '__dict__'):
        return type(x).__name__
    return x


def dict_product(dicts):
    return list(dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


class FeatureSelectionConfig:
    def __init__(self, score_func, size):
        self.score_func = score_func
        self.size = size

    def __repr__(self):
        return '{}_{}'.format(get_object_name(self.score_func), self.size)


def feature_selection_configs(score_funcs, sizes):
    return map(lambda x: FeatureSelectionConfig(x[0], x[1]), [(None, None)] + list(itertools.product(score_funcs, sizes)))


class RatioRandomUnderSampler(RandomUnderSampler):
    def __init__(self, pos_ratio, random_state=0):
        self.pos_ratio = pos_ratio
        self.ratio_sampler = None
        super(RatioRandomUnderSampler, self).__init__(random_state=random_state)

    def fit(self, X, y):
        pos = len(y[y == 1])
        neg = int(pos * ((1 - self.pos_ratio) / self.pos_ratio))
        self.ratio_sampler = RandomUnderSampler(random_state=self.random_state, ratio={0: neg, 1: pos})
        self.ratio_sampler.fit(X, y)
        return self

    def sample(self, X, y):
        return self.ratio_sampler.sample(X, y)


class ModifiedRandomOverSampler(RandomOverSampler):
    def __init__(self, pos_samples, random_state=0):
        self.pos_samples = pos_samples
        self.ratio_sampler = None
        super(ModifiedRandomOverSampler, self).__init__(random_state=random_state)

    def fit(self, X, y):
        pos = self.pos_samples
        neg = len(y[y == 0])
        self.ratio_sampler = RandomOverSampler(random_state=self.random_state, ratio={0: neg, 1: pos})
        self.ratio_sampler.fit(X, y)
        return self

    def sample(self, X, y):
        return self.ratio_sampler.sample(X, y)


class ModifiedSMOTE(SMOTE):
    def __init__(self, pos_samples, random_state=0):
        self.pos_samples = pos_samples
        self.ratio_sampler = None
        super(ModifiedSMOTE, self).__init__(random_state=random_state)

    def fit(self, X, y):
        pos = self.pos_samples
        neg = len(y[y == 0])
        self.ratio_sampler = SMOTE(random_state=self.random_state, ratio={0: neg, 1: pos})
        self.ratio_sampler.fit(X, y)
        return self

    def sample(self, X, y):
        return self.ratio_sampler.sample(X, y)


class ModifiedSelectKBest(SelectKBest):
    """Modified SelectKBest to default to all features if k > n_features
    """

    def _check_params(self, X, y):
        if self.k != 'all' and self.k > X.shape[1]:
            warnings.warn('k > n_features (%r, %r), setting to all' % (self.k, X.shape[1]))
            self.k = 'all'
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError("k should be >=0, <= n_features; got %r."
                             "Use k='all' to return all features."
                             % self.k)


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


def monte_carlo(pipeline, x, y, n_runs, random_state, df=True):
    sss = StratifiedShuffleSplit(n_splits=n_runs, test_size=0.3, random_state=random_state)
    out = []
    for train_index, test_index in sss.split(x, y):
        if df:
            x_train, y_train = x.iloc[train_index], y.iloc[train_index]
            x_test, y_test = x.iloc[test_index], y.iloc[test_index]
        else:
            x_train, y_train = x[train_index, :], y[train_index]
            x_test, y_test = x[test_index, :], y[test_index]
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict_proba(x_test)
        predicted = predicted[:, 1] if len(predicted.shape) > 1 else predicted
        out.append(pd.DataFrame({'predicted': predicted, 'actual': y_test, 'run': [len(out)] * x_test.shape[0]}))
    return pd.concat(out)


def sparse_relu(x):
    x.data = np.where(x.data < 0, np.zeros(x.data.shape), x.data)
    return x


def get_s3(key, bucket):
    obj = s3.Object(bucket, key)
    return io.BytesIO(obj.get()['Body'].read())


def delete_recursive_s3(key, bucket):
    objects_to_delete = s3.meta.client.list_objects(Bucket=bucket, Prefix=key)

    delete_keys = {'Objects': [{'Key': k} for k in [obj['Key'] for obj in objects_to_delete.get('Contents', [])]]}

    if delete_keys['Objects']:
        s3.meta.client.delete_objects(Bucket=bucket, Delete=delete_keys)


def to_csv_s3(df, key, bucket, index=False):
    buf = io.StringIO()
    df.to_csv(buf, index=index)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def np_save_s3(x, key, bucket):
    buf = io.BytesIO()
    np.save(buf, x)
    s3.Object(bucket, '{}.npy'.format(key)).put(Body=buf.getvalue())


def sp_save_s3(x, key, bucket):
    buf = io.BytesIO()
    sp.save_npz(buf, x)
    s3.Object(bucket, '{}.npz'.format(key)).put(Body=buf.getvalue())


def joblib_dump_s3(obj, key, bucket):
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def json_dump_s3(obj, key, bucket):
    buf = io.BytesIO()
    json.dump(obj, buf, indent=4)
    s3.Object(bucket, key).put(Body=buf.getvalue())


def copy_dir_s3(path, key, bucket):
    for f in os.listdir(path):
        s3.Bucket(bucket).upload_file(os.path.join(path, f), os.path.join(key, f))


def file_exists(path, bucket=None, local=False):
    if local:
        return os.path.exists(path)
    else:
        obj_status = s3_client.list_objects(Bucket=bucket, Prefix=path)
        if obj_status.get('Contents'):
            return True
        else:
            return False


def run_command(cmd, return_stdout=False):
    """from http://blog.kagesenshi.org/2008/02/teeing-python-subprocesspopen-output.html
    python 3 fix: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    """
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = []
    while True:
        line = p.stdout.readline().decode()
        stdout.append(line)
        print(line, end='')
        if line == '' and not p.poll() is None:
            break
    if return_stdout:
        return ''.join(stdout)


def cross_validate_repeat(estimator, X, y=None, scoring=None, n_jobs=1, verbose=0, random_state=0, return_train_score=True, runs=5, folds=5, metadata=None):
    if not scoring:
        scoring = ['roc_auc']
    results = pd.DataFrame()
    for i in range(random_state, random_state + runs):
        np.random.seed(i)
        cv = StratifiedKFold(n_splits=folds, shuffle=True)
        scores = cross_validate(estimator=estimator, X=X, y=y, scoring=scoring, cv=cv, n_jobs=n_jobs, verbose=verbose, return_train_score=return_train_score)
        result = pd.DataFrame(scores)
        for m in metadata:
            result[m] = metadata[m]
        results = pd.concat([results, result])
    return results


class DatasetStats(object):

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def agg_column_stats(x):
        funcs = [np.amax, np.amin, np.mean, np.median, np.std]
        return pd.Series(x).apply(funcs).to_dict()

    # Table stats

    def num_instances(self):
        return self._X.shape[0]

    def num_positive(self):
        return np.count_nonzero(self._Y == 1)

    def num_negative(self):
        return np.count_nonzero(self._Y == 0)

    def positive_ratio(self):
        return self.num_positive() / self.num_instances()

    def num_attributes(self):
        return self._X.shape[1]

    def density(self):
        if self._sparse:
            nonzero = self._X.count_nonzero()
        else:
            nonzero = np.count_nonzero(self._X)
        return nonzero / (self.num_instances() * self.num_attributes())

    def _run_kmeans(self, n_clusters, random_state):
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=1000)
        kmeans.fit(self._X)
        rmsd = np.sqrt(kmeans.inertia_ / self._X.shape[0])
        return rmsd

    def _kmeans_rmsd(self, n_clusters, runs=5):
        rmsds = []
        for i in range(runs):
            rmsds.append(self._run_kmeans(n_clusters, self._random_state + i))
        return np.mean(rmsds)

    def kmeans_rmsd1(self):
        return self._kmeans_rmsd(1)

    def kmeans_rmsd2(self):
        return self._kmeans_rmsd(2)

    def kmeans_rmsd3(self):
        return self._kmeans_rmsd(3)

    def kmeans_rmsd4(self):
        return self._kmeans_rmsd(4)

    def kmeans_rmsd5(self):
        return self._kmeans_rmsd(5)

    def kmeans_rmsd6(self):
        return self._kmeans_rmsd(6)

    def kmeans_rmsd7(self):
        return self._kmeans_rmsd(7)

    def kmeans_rmsd8(self):
        return self._kmeans_rmsd(8)

    def kmeans_rmsd9(self):
        return self._kmeans_rmsd(9)

    def kmeans_rmsd10(self):
        return self._kmeans_rmsd(10)

    # Column stats

    def amax(self):
        if self._sparse:
            return spf.min_max_axis(self._X, 0)[1]
        else:
            return np.apply_along_axis(np.amax, 0, self._X)

    def amin(self):
        if self._sparse:
            return spf.min_max_axis(self._X, 0)[0]
        else:
            return np.apply_along_axis(np.amin, 0, self._X)

    def mean(self):
        if self._sparse:
            return spf.mean_variance_axis(self._X, 0)[0]
        else:
            return np.apply_along_axis(np.mean, 0, self._X)

    def median(self):
        if self._sparse:
            return spf.csc_median_axis_0(self._X.tocsc())
        else:
            return np.apply_along_axis(np.median, 0, self._X)

    def std(self):
        if self._sparse:
            return np.sqrt(spf.mean_variance_axis(self._X, 0)[1])
        else:
            return np.apply_along_axis(np.std, 0, self._X)

    def chi2(self):
        ft = FunctionTransformer(np.abs, accept_sparse=True)
        x_abs = ft.fit_transform(self._X)
        chi2_score, pval = chi2(x_abs, self._Y)
        return chi2_score

    def __init__(self, X, Y, sparse=False, random_state=42):
        self.X = X
        self.Y = Y
        # these are used for "current" dataset to run
        self._X = X
        self._Y = Y
        self._sparse = sparse
        self._random_state = random_state
        self.table_metrics = [
            self.num_instances,
            self.num_positive,
            self.num_negative,
            self.positive_ratio,
            self.num_attributes,
            self.density,
            self.kmeans_rmsd1,
            self.kmeans_rmsd2,
            self.kmeans_rmsd3,
            self.kmeans_rmsd4,
            self.kmeans_rmsd5,
            self.kmeans_rmsd6,
            self.kmeans_rmsd7,
            self.kmeans_rmsd8,
            self.kmeans_rmsd9,
            self.kmeans_rmsd10,
        ]
        self.column_metrics = [
            self.chi2,
            self.amax,
            self.amin,
            self.mean,
            self.median,
            self.std,
        ]
        self._table_metrics = None
        self._column_metrics = None
        self.results = []
        self.results_df = None

    def include_all(self):
        self._table_metrics = self.table_metrics
        self._column_metrics = self.column_metrics
        return self

    def exclude(self, *args):
        self._table_metrics = [m for m in self.table_metrics if m.__name__ not in args]
        self._column_metrics = [m for m in self.column_metrics if m.__name__ not in args]
        return self

    def include(self, *args):
        self._table_metrics = [m for m in self.table_metrics if m.__name__ in args]
        self._column_metrics = [m for m in self.column_metrics if m.__name__ in args]
        return self

    def run_data(self, data):
        for m in self._table_metrics:
            start = time.time()
            out_val = m()
            end = time.time() - start

            out = {
                'metric': f'{data}_{m.__name__}',
                'result': out_val,
                'time': end,
                'group': m.__name__,
                'data': data
            }
            self.results.append(out)

        for m in self._column_metrics:
            start = time.time()
            out_all = m()
            end = time.time() - start

            aggregated = self.agg_column_stats(out_all)
            for k, v in aggregated.items():
                out = {
                    'metric': f'{data}_{m.__name__}_{k}',
                    'result': v,
                    'time': end,
                    'group': m.__name__,
                    'data': data,
                }
                self.results.append(out)

    def run(self):
        if self._table_metrics or self._column_metrics:
            print('Running for full data')
            self.run_data('all')

            print('Running for positive only')
            pos_idx = self.Y == 1
            self._X = self.X[pos_idx]
            self._Y = self.Y[pos_idx]
            self.run_data('positive')

            print('Running for negative only')
            neg_idx = self.Y == 0
            self._X = self.X[neg_idx]
            self._Y = self.Y[neg_idx]
            self.run_data('negative')

            self.results_df = pd.DataFrame(self.results)
            return self.results_df
        else:
            raise ValueError('Must select metrics using include_all(), include(), or exclude()')
