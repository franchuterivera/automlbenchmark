import argparse
import pathlib
import pickle
import pprint
import tempfile
from collections import namedtuple

from autogluonbenchmarking.autogluon_utils.benchmarking.baselines.process_data import processData
from autogluonbenchmarking.autogluon_utils.configs.kaggle.constants import (
    EVAL_METRIC,
    FETCH_PROCESSOR,
    INDEX_COL,
    LABEL_COLUMN,
    NAME,
    PRE_PROCESSOR,
    PROBLEM_TYPE,
)
from autogluonbenchmarking.autogluon_utils.configs.kaggle.kaggle_competitions import (
    KAGGLE_COMPETITIONS
)
from autogluonbenchmarking.autogluon_utils.kaggle_tools.kaggle_utils import fetch_kaggle_files

from frameworks.AutoGluon_020.exec import run as autogluon_run
from frameworks.autosklearn_memefficient_metalearning.exec import run as autosklearn_run
from frameworks.autosklearnEnsembleIntensification_noniterative_parallel.exec import run as autosklearn_ensembleintensification_run
from frameworks.autosklearn_autogluonstrategy.exec import run as autosklearn_autogluonstrategy_run

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype


def get_task_setup(task_name):
    # {
    #     NAME: 'ieee-fraud-detection',
    #     LABEL_COLUMN: 'isFraud',
    #     EVAL_METRIC: 'roc_auc',
    #     PROBLEM_TYPE: BINARY,
    #     INDEX_COL: 'TransactionID',
    #     PRE_PROCESSOR: IeeeFraudDetectionPreProcessor(),
    #     POST_PROCESSOR: StandardPostProcessor(),
    #     FITTING_TIME_LIMITS: NO_TIME_LIMITS,
    # }
    task = [task_ for task_ in KAGGLE_COMPETITIONS if task_[NAME] == task_name][-1]
    task['output_dir'] = pathlib.Path.home() / pathlib.Path('kaggle')
    task['output_dir'].mkdir(parents=True, exist_ok=True)
    task['train_path'] = pathlib.Path.home().joinpath(
        'kaggle', task[NAME], 'train.preprocessed.csv')
    task['test_path'] = pathlib.Path.home().joinpath(
        'kaggle', task[NAME], 'test.preprocessed.csv')
    return task


def fetch_kaggle_dataset(args):
    # Autogluon fetches the data once as described in fetch() of
    # autogluon_utils/benchmarking/kaggle/fetch_datasets.py

    task = get_task_setup(args.task)
    print(f"task={task}")

    # Generate the file once
    if task['train_path'].exists():
        return task

    # Fetch the file as done here autogluon_utils/kaggle_tools/kaggle_utils.py
    files = fetch_kaggle_files(competition=task[NAME], outdir=task['output_dir'])
    print('Retrieved files:')
    pprint.pprint(files)
    fetch_processor = task.get(FETCH_PROCESSOR)
    if fetch_processor:
        files = fetch_processor(files)
    df_train, df_test = task[PRE_PROCESSOR].preprocess(
        file_to_location={
            'train.csv': task['output_dir'].joinpath(task[NAME], 'train.csv'),
            'train_identity.csv': task['output_dir'].joinpath(task[NAME], 'train_identity.csv'),
            'train_transaction.csv': task['output_dir'].joinpath(task[NAME], 'train_transaction.csv'),
            'test_identity.csv': task['output_dir'].joinpath(task[NAME], 'test_identity.csv'),
            'test_transaction.csv': task['output_dir'].joinpath(task[NAME], 'test_transaction.csv'),
            'test.csv': task['output_dir'].joinpath(task[NAME], 'test.csv'),
        },
        competition_meta=task,
    )
    df_train.to_csv(task['train_path'], index=task[INDEX_COL] is not None)
    df_test.to_csv(task['test_path'], index=task[INDEX_COL] is not None)
    return task


config = namedtuple('config', [
    'framework_params', 'type', 'max_runtime_seconds', 'output_predictions_file',
    'output_dir', 'name', 'fold', 'cores', 'max_mem_size_mb', 'seed', 'metric'])


def framework2params(framework):
    if framework == 'AutoGluon':
        return {'presets': 'best_quality'}
    elif framework == 'AutoSklearn':
        return {'resampling_strategy': 'cv', 'k_folds': 5}
    elif framework == 'AutoSklearnEnsembleIntensification':
        return {
            'k_folds': [3, 5, 10, 5, 3],
            'repeats': 5,
            'max_stacking_level': 2,
            'stacking_strategy': 'instances_anyasbase',
            'ensemble_folds': 'highest_repeat_trusted',
            'max_ensemble_members': 10,
            'min_challengers': 10,
            'stack_at_most': 20,
            'min_prune_members': 10,
            'fidelities_as_individual_models': True,
            'enable_median_rule_prunning': True,
            'train_all_repeat_together': False,
            'fast_track_performance_criteria': 'common_instances',
            'stack_based_on_log_loss': False,
            'stack_tiebreak_w_log_loss': True,
            'resampling_strategy': 'intensifier-cv',
            'test_loss_in_autosklearn': False,
        }
    elif framework == 'AutoSklearnAutoGluonStrategy':
        return {
            'k_folds': [3, 5, 10, 5, 3, 3, 5, 10, 5, 3, 3, 5, 10, 5, 3, 3, 5, 10, 5, 3],
            'repeats': 20,
            'max_stacking_level': 3,
            'stacking_strategy': 'instances_anyasbase',
            'ensemble_folds': 'highest_repeat_trusted',
            'max_ensemble_members': 10,
            'min_challengers': 10,
            'stack_at_most': 20,
            'min_prune_members': 10,
            'fidelities_as_individual_models': False,
            'enable_median_rule_prunning': False,
            'train_all_repeat_together': False,
            'fast_track_performance_criteria': 'common_instances',
            'stack_based_on_log_loss': False,
            'stack_tiebreak_w_log_loss': False,
            'resampling_strategy': 'partial-iterative-intensifier-cv',
            'test_loss_in_autosklearn': False,
            'only_intensify_members_repetitions': True,
            'test_loss_in_autosklearn': False,
        }
    else:
        raise NotImplementedError


def str2mem(memory):
    return {'12G': 12288, '32G': 32768, '8G': 4096}[memory]


metric_map = {
    'roc_auc': 'auc',
    'mean_absolute_error': 'mae',
    'mean_squared_error': 'mse',
    'log_loss': 'logloss',
}


def translate_task_2_automlbenchmark(task, args):
    return config(
        metric=metric_map[task[EVAL_METRIC]],
        framework_params=framework2params(args.framework),
        type='classification',
        max_runtime_seconds=args.runtime,
        output_predictions_file=f"{args.framework}.{args.task}.{args.seed}.{args.runtime}.csv",
        output_dir=tempfile.TemporaryDirectory(),
        name=args.framework,
        fold=0,
        cores=args.cores,
        max_mem_size_mb=str2mem(args.memory),
        seed=args.seed,
    )


data = namedtuple('data', ['data'])
column = namedtuple('column', ['name'])
dataset = namedtuple('dataset', [
    'columns', 'train', 'test', 'target', 'problem_type', 'predictors_type'])


def generate_autogluon_dataset(X_train, y_train, X_test, y_test, task):
    X_train[task[LABEL_COLUMN]] = y_train

    # we don't have y_test :( so mimic with y_train
    X_test[task[LABEL_COLUMN]] = y_train[:X_test.shape[0]]
    X_test[task[LABEL_COLUMN]].fillna(value=0, inplace=True)
    # keep as object everything that is not numerical
    columns=[(col, ('object' if not is_numeric_dtype(X_train[col]) else 'int' if ('int' in str(X_train[col].dtype).lower()) else 'float')) for col in X_train.columns]
    return dataset(
        columns=columns,
        target=column(name=task[LABEL_COLUMN]),
        problem_type=task[PROBLEM_TYPE],
        train=data(data=X_train),
        test=data(data=X_test),
        predictors_type=None,
    )


enc = namedtuple('enc', ['X_enc', 'y_enc'])

def generate_autosklearn_dataset(X_train, y_train, X_test, y_test, task):
    predictors_type = ['Numerical' if is_numeric_dtype(X_train[col]) else 'Categorical' for col in X_train.columns]

    y_test = np.zeros((X_test.shape[0], y_train.shape[1]) if len(y_train.shape) > 1 else (X_test.shape[0]))
    y_test[:X_test.shape[0]] = y_train[:X_test.shape[0]]

    # we don't have y_test :( so mimic with y_train
    return dataset(
        columns=None,
        target=None,
        problem_type=None,
        train=enc(X_enc=X_train.to_numpy(), y_enc=y_train.to_numpy()),
        test=enc(X_enc=X_test.to_numpy(), y_enc=y_test),
        predictors_type=predictors_type,
    )


def fit_and_predict_on_test(task, args):
    # load the train data as specified here:
    # autogluon_utils/benchmarking/kaggle/evaluate_results.py
    df_train = pd.read_csv(task['train_path'], index_col=task[INDEX_COL], low_memory=False)
    df_test = pd.read_csv(task['test_path'], index_col=task[INDEX_COL], low_memory=False)

    # Make sure we have proper dtypes
    df_train = df_train.convert_dtypes()
    # is_classification==True
    df_train[task[LABEL_COLUMN]] = df_train[task[LABEL_COLUMN]].astype('category')
    for col in df_train.columns:
        if df_train[col].dtype.name == 'string':
            df_train[col] = df_train[col].astype('category')
        try:
            df_test[col] = df_test[col].astype(df_train[col].dtype)
        except Exception as e:
            print(f"Error on col={col}: {str(e)}")
    print(df_train.dtypes)

    if args.subsample:
        df_train = df_train.sample(n=3000)
        df_test = df_test.sample(n=3000)

    # Apply preprocessing
    # autogluon_utils/benchmarking/baselines/autosklearn_base/autosklearn_base.py
    X_train, y_train, ag_predictor = processData(
        data=df_train, label_column=task[LABEL_COLUMN],
        problem_type=task[PROBLEM_TYPE], eval_metric=task[EVAL_METRIC]
    )

    X_test, y_test, _ = processData(data=df_test, label_column=ag_predictor._learner.label,
                                    ag_predictor=ag_predictor)

    # Get a config file for the automlbenchmark
    config = translate_task_2_automlbenchmark(task, args)

    # Train the predictor
    # autogluon_utils/benchmarking/kaggle/predictors/predictors.py
    if args.framework == 'AutoGluon':
        dataset = generate_autogluon_dataset(X_train, y_train, X_test, y_test, task)
        print(f"dataset={dataset} config={config}")
        result = autogluon_run(dataset=dataset, config=config)
    elif 'AutoSklearn' in args.framework:
        dataset = generate_autosklearn_dataset(X_train, y_train, X_test, y_test, task)
        print(f"dataset={dataset} config={config}")
        if args.framework == 'AutoSklearn':
            result = autosklearn_run(dataset=dataset, config=config)
        elif args.framework == 'AutoSklearnAutoGluonStrategy':
            result = autosklearn_autogluonstrategy_run(dataset=dataset, config=config)
        elif args.framework == 'AutoSklearnEnsembleIntensification':
            result = autosklearn_ensembleintensification_run(dataset=dataset, config=config)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print(f"result={result}")

    return ag_predictor._learner.label_cleaner.inverse_transform(pd.Series(result['predictions']))


def str2bool(v):
    """
    Process any bool
    Taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '-f',
        '--framework',
        help='What framework to manage',
        choices=['AutoGluon',
                 'AutoSklearn',
                 'AutoSklearnAutoGluonStrategy',
                 'AutoSklearnEnsembleIntensification'],
        required=True
    )
    parser.add_argument(
        '-t',
        '--task',
        required=True,
        choices=['porto-seguro-safe-driver-prediction',
                 'santander-customer-satisfaction',
                 'santander-customer-transaction-prediction',
                 'ieee-fraud-detection',
                 'otto-group-product-classification-challenge',
                 'bnp-paribas-cardif-claims-management',
                 'microsoft-malware-prediction'],
        help='What specific task to run'
    )
    parser.add_argument(
        '-c',
        '--cores',
        default=8,
        type=int,
        help='The number of cores to use'
    )
    # The run was made using PROFILE_FULL_1HR = 'PROFILE_FULL_1HR'
    parser.add_argument(
        '--runtime',
        default=3600,
        type=int,
        help='The number of seconds a run should take'
    )
    parser.add_argument(
        '--seed',
        default=42,
        type=int,
        help='The seed of this job'
    )
    parser.add_argument(
        '-m',
        '--memory',
        default='32G',
        choices=['12G', '32G', '8G'],
        help='the ammount of memory to allocate to a job'
    )
    parser.add_argument(
        '--subsample',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='To test on my computer!'
    )

    args = parser.parse_args()

    # Get the data
    task = fetch_kaggle_dataset(args)

    # Fit and predict with the desired model
    y_pred = fit_and_predict_on_test(task=task, args=args)
    with open(f"{args.framework}.{args.task}.{args.seed}.{args.runtime}.{args.memory}.{args.cores}.predictions.pkl", 'wb') as f:
        pickle.dump(y_pred, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Push to kaggle
    # TODO
