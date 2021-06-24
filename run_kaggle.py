import argparse
import pathlib
import pprint
from collections import namedtuple

# Description on how the predictiors were setup
#https://github.com/Innixma/autogluon-benchmarking/blob/master/autogluon_utils/benchmarking/kaggle/predictors/predictors.py
# The run was made using PROFILE_FULL_1HR = 'PROFILE_FULL_1HR'
from frameworks.AutoGluon_020.exec import run as autogluon_run
from autogluon_utils.configs.kaggle.constants import (
    NAME,
    LABEL_COLUMN,
    EVAL_METRIC,
    FETCH_PROCESSOR,
)
from autogluon_utils.kaggle_tools.kaggle_utils import fetch_kaggle_files, kaggle_api
from autogluon_utils.benchmarking.baselines.process_data import processData

# This file is taken from here https://github.com/Innixma/autogluon-benchmarking/blob/master/autogluon_utils/configs/kaggle/kaggle_competitions.py
from autogluon-benchmarking.autogluon_utils.configs.kaggle.kaggle_competitions import KAGGLE_COMPETITIONS


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
    task['output_dir'] = pathlib.Path.home() / pathlib.Path('kaggle') / pathlib.Path(task[NAME])
    task['output_dir'].(parents=True, exist_ok=True)
    task['train_path'] = pathlib.Path.home() / pathlib.Path('kaggle') / pathlib.Path('train.preprocessed.csv')
    task['test_path'] = pathlib.Path.home() / pathlib.Path('kaggle') / pathlib.Path('test.preprocessed.csv')
    return task


def fetch_kaggle_dataset(args):
    # Autogluon fetches the data once as described in fetch() of
    # autogluon_utils/benchmarking/kaggle/fetch_datasets.py

    task = get_task_setup(args.task)

    # Generate the file once
    if task['train_path'].exists():
        return task

    # Fetch the file as done here autogluon_utils/kaggle_tools/kaggle_utils.py
    files = fetch_kaggle_files(competition=task[NAME], outdir=task['output_dir'])
    print(f'Retrieved files:')
    pprint.pprint(files)
    fetch_processor = task.get(FETCH_PROCESSOR)
    if fetch_processor:
        files = fetch_processor(files)
    file
    'train.csv', test_name: str = 'test.csv'
    df_train, df_test = task[PRE_PROCESSOR].preprocess(
        file_to_location={
            'train.csv': task['output_dir'].join('train.csv'),
            'test.csv': task['output_dir'].join('test.csv'),
        },
        competition_meta=task,
    )
    df_train.to_csv(task['train_path'], index=task[INDEX_COL] is not None)
    df_test.to_csv(task['test_path'], index=task[INDEX_COL] is not None)
    return task


config = namedtuple('config', [
    'name', 'framework_params', 'type', 'max_runtime_seconds', 'output_predictions_file',
    'output_dir', 'name', 'fold', 'cores', 'max_mem_size_mb', 'seed'])


def framework2params(framework):
    if framework == 'AutoGluon':
        return {'presets': 'best_quality'}
    else:
        raise NotImplementedError


def str2mem(memory):
    return {'12G': 12288, '32G': 32768, '8G': 4096}[memory]


def translate_task_2_automlbenchmark(task, args):
    return config(
        metric=task['EVAL_METRIC'],
        framework_params=framework2params(args.framework),
        type='classification',
        max_runtime_seconds=args.runtime,
        output_predictions_file=f"{args.framework}.{args.task}.{args.seed}.{args.runtime}.csv",
        output_dir='.',
        name=args.framework,
        fold=0,
        cores=args.cores,
        max_mem_size_mb=str2mem(args.memory),
        seed=args.seed,
    )


data = namedtuple('data', ['data'])
column = namedtuple('column', ['name'])
dataset = namedtuple('dataset', [
    'columns', 'train', 'test', 'target', 'problem_type'])


def generate_autogluon_dataset(X_train, y_train, X_test, y_test, task):
    X_train[task[PROBLEM_TYPE]] = y_train
    return dataset(
        columns=X_train.columns,
        target=column(name=task[LABEL_COLUMN]),
        problem_type=task[PROBLEM_TYPE],
        test=data(data=X_train),
        train=data(data=X_test),
    )


def fit_and_predict_on_test(task, framework):
    # load the train data as specified here:
    # autogluon_utils/benchmarking/kaggle/evaluate_results.py
    df_train = pd.read_csv(train_path, index_col=task[INDEX_COL], low_memory=False)

    # Apply preprocessing
    # autogluon_utils/benchmarking/baselines/autosklearn_base/autosklearn_base.py
    X_train, y_train, ag_predictor = processData(
        data=df_train, label_column=task[LABEL_COLUMN],
        problem_type=task[PROBLEM_TYPE], eval_metric=task[EVAL_METRIC]
    )
    # Predict
    df_test = pd.read_csv(test_path, index_col=competition_meta[INDEX_COL], low_memory=False)
    X_test, y_test, _ = processData(data=df_test, label_column=ag_predictor._learner.label,
                                    ag_predictor=ag_predictor)

    # Get a config file for the automlbenchmark
    config = translate_task_2_automlbenchmark(task)

    # Train the predictor
    # autogluon_utils/benchmarking/kaggle/predictors/predictors.py
    if framework == 'AutoGluon':
        dataset = generate_autogluon_dataset(X_train, y_train, X_test, y_test)
        result = autogluon_run(dataset=dataset, config=config)
    else:
        raise NotImplementedError

    return ag_predictor._learner.label_cleaner.inverse_transform(pd.Series(result.prediction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '-f',
        '--framework',
        help='What framework to manage',
        choices=['AutoGluon', 'AutoSklearn', 'AutoSklearnAutoGluonStrategy', 'AutoSklearnEnsembleIntensification'],
        required=True
    )
    parser.add_argument(
        '-t',
        '--task',
        required=True,
        choices=['porto-seguro-safe-driver-prediction','santander-customer-satisfaction','santander-customer-transaction-prediction', 'ieee-fraud-detection', 'microsoft-malware-prediction'],
        help='What specific task to run'
    )
    parser.add_argument(
        '-c',
        '--cores',
        default=8,
        type=int,
        help='The number of cores to use'
    )
    parser.add_argument(
        '--runtime',
        default=3600,
        type=int,
        help='The number of seconds a run should take'
    )
    parser.add_argument(
        '-m',
        '--memory',
        default='32G',
        choices=['12G', '32G', '8G'],
        help='the ammount of memory to allocate to a job'
    )
    parser.add_argument(
        '-p',
        '--partition',
        default='bosch_cpu-cascadelake',
        choices=['ml_cpu-ivy', 'test_cpu-ivy', 'bosch_cpu-cascadelake'],
        help='In what partition to launch'
    )
    parser.add_argument(
        '--run_mode',
        type=str,
        choices=['single', 'array', 'None', 'interactive'],
        default=None,
        help='Launches the run to sbatch'
    )
    parser.add_argument(
        '--collect_overfit',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='generates a problems dataframe'
    )
    args = parser.parse_args()

    # Get the data
    task = fetch_kaggle_dataset(args)

    # Fit and predict with the desired model
    y_pred = fit_and_predict_on_test(task=task, framework=args.framework)

    # Push to kaggle
    # TODO
