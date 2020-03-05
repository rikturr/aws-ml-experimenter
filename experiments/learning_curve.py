from experiments.helpers import *
import logging
from datetime import datetime
import importlib
import argparse
import uuid
import random
from multiprocessing.dummy import Pool as ThreadPool
from imblearn.pipeline import make_pipeline

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=PendingDeprecationWarning)

logger = logging.getLogger('root')
logFormatter = logging.Formatter('%(asctime)s %(process)d [%(funcName)s] %(levelname)s: %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(logFormatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_module", help="Config module")
    parser.add_argument("config_num", type=int, help="Which iteration of the config to use")
    parser.add_argument("--experiment-id", help="Assign experiment ID")

    return parser.parse_args()


def pseudo_label(pipeline, x_lab, y_lab, x_unlab, y_unlab, threshold=None):
    model = make_pipeline(*pipeline)
    model.fit(x_lab, y_lab)

    pseudo_lab = pd.DataFrame({
        'actual': y_unlab,
        'predict_proba': model.predict_proba(x_unlab)[:, 1]
    })
    if threshold:
        results = threshold_metrics(pseudo_lab['actual'], pseudo_lab['predict_proba'], threshold=threshold)
    else:
        results = threshold_metrics(pseudo_lab['actual'], pseudo_lab['predict_proba'], rank_best='lab_gmean')
    pseudo_lab['predicted'] = (pseudo_lab['predict_proba'] > results['lab_threshold']).astype(int)

    y_pseudo = pseudo_lab['predicted'].values
    results['lab_num_pos'] = np.sum(y_pseudo)
    results['lab_num_neg'] = y_pseudo.shape[0] - results['lab_num_pos']

    return y_pseudo, results


def model_eval(pipeline, local, x, y, pos_ratio, results_path, bucket, rus, random_state, pos_size, run, extra=None):
    start = datetime.now()

    error = False
    error_message = ''
    if pos_size:
        total = int(pos_size / pos_ratio)
        neg_size = total - pos_size

        if np.sum(y) < pos_size:
            error = True
            error_message += f' Not enough positive samples: {np.sum(y)}'
        if np.count_nonzero(y == 0) * 2 < neg_size:
            error = True
            error_message += f' Not enough negative samples: {np.count_nonzero(y == 0)}'
    else:
        total = y.shape[0]
        neg_size = total - np.sum(y)
    logger.warning(f"Starting RUN={run} Size={pos_size} RUS={rus}")

    if error:
        o = {'error': [error_message]}
        logger.warning(f"RUN={run} Size={pos_size} RUS={rus} {error_message}")
    else:
        if pos_size:
            x_train, y_train = sample_data(x, y, pos_size, neg_size, random_state + run)
        else:
            x_train = x
            y_train = y

        model = make_pipeline(*pipeline)
        o = cross_validate(model, x_train, y_train, cv=5, scoring=['roc_auc'])

        logger.warning(f"RUN={run} Size={pos_size} RUS={rus} AUC={round(np.mean(o['test_roc_auc']), 4)} Time: {datetime.now() - start}")

    o['num_instances'] = total
    o['num_pos'] = pos_size
    o['num_neg'] = neg_size
    o['rus'] = rus
    o['run'] = run
    if extra:
        for k, v in extra.items():
            o[k] = v

    filename = f'{results_path}/pos_size={pos_size}_rus={rus}_run={run}.csv'
    # if file_exists(filename, bucket=bucket, local=local):
    #     current_results = pd.read_csv(filename if local else get_s3(filename, bucket=bucket))
    #     new_results = pd.concat([current_results, pd.DataFrame(o)])
    # else:
    #     new_results = pd.DataFrame(o)
    new_results = pd.DataFrame(o)

    if local:
        new_results.to_csv(filename, index=False)
    else:
        to_csv_s3(new_results, filename, bucket=bucket)


def main():
    args = parse_args()
    config_module = args.config_module
    logger.warning('Using config: {}'.format(config_module))
    config = importlib.import_module(config_module)

    path = config.path
    x_path = config.x_path
    y_path = config.y_path
    local = config.local
    bucket = config.bucket
    experiment_id = args.experiment_id or uuid.uuid4()
    sparse = hasattr(config, 'sparse') and config.sparse
    random_state = config.random_state

    config_num = args.config_num
    exp_config = config.configs[config_num]
    name = exp_config['name']
    rus = exp_config['rus']
    pipeline = exp_config['pipeline']
    sizes = exp_config['sizes']
    pseudo_size = exp_config['pseudo_size'] if 'pseudo_size' in exp_config else None
    runs = exp_config['runs']
    threshold = exp_config['threshold'] if 'threshold' in exp_config else None
    nthreads = exp_config['nthreads'] if 'nthreads' in exp_config else config.nthreads

    logger.warning(f'Creative curve for: {name}')

    logger.warning(f'Loading data from: {x_path} {y_path}')
    if local:
        x_file = x_path
        y_file = y_path
    else:
        x_file = get_s3(x_path, bucket)
        y_file = get_s3(y_path, bucket)

    if sparse:
        X = sp.load_npz(x_file)
    else:
        X = np.load(x_file)
    Y = np.load(y_file)
    check_data(X, Y)

    logger.warning(f'Sizes to run: {sizes}')

    pos_ratio = np.sum(Y) / Y.shape[0]
    logger.warning(f'Positive class ratio: {pos_ratio}')

    if rus:
        pipeline = [RatioRandomUnderSampler(rus, random_state=random_state)] + pipeline

    y_pseudo = {}
    pseudo_results = {}
    if pseudo_size:
        logger.warning(f'Pseudo size: {pseudo_size}')
        total = int(pseudo_size / pos_ratio)
        neg_size = total - pseudo_size

        for r in runs:
            x_lab, y_lab = sample_data(X, Y, pseudo_size, neg_size, random_state + r)
            pseudo_lab, results = pseudo_label(pipeline, x_lab, y_lab, X, Y, threshold=threshold)
            y_pseudo[r] = pseudo_lab

            results['pseudo_size'] = pseudo_size
            pseudo_results[r] = results

    run_args = [(pipeline,
                 local,
                 X,
                 y_pseudo[r] if y_pseudo else Y,
                 pos_ratio,
                 f'{path}/results/{name}',
                 bucket,
                 rus,
                 random_state,
                 s,
                 r,
                 pseudo_results[r] if pseudo_results else None) for s in sizes for r in runs]
    if nthreads > 1:
        random.seed(random_state)
        # random.shuffle(run_args)

    with ThreadPool(nthreads) as pool:
        pool.starmap(model_eval, run_args)


if __name__ == "__main__":
    main()
