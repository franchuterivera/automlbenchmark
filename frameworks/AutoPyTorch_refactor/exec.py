import logging
import math
import os
import tempfile as tmp
import warnings
import pickle

import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd
from shutil import copyfile

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import autoPyTorch
from autoPyTorch.api.tabular_classification import TabularClassificationTask
import autoPyTorch.metrics as metrics

from frameworks.shared.callee import call_run, result, output_subdir, utils



def run(dataset, config):
    print("\n**** AutoPyTorch {} ****\n".format(autoPyTorch.__version__))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    # Mapping of benchmark metrics to AutoPyTorch metrics
    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        balacc=metrics.balanced_accuracy,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        # AutoPyTorch can optimize on mse, and we compute rmse independently on predictions
        rmse=metrics.mean_squared_error,
        r2=metrics.r2
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through,
        # or if we use a strict mapping
        #log.warning("Performance metric %s not supported.", config.metric)
       print("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    print(
        "Running AutoPyTorch with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
        config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric
    )
    print("Environment: %s", os.environ)

    # Data Management -- We pass the interpreted dataframe
    # make sure that test and train have both same data type
    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
    label = dataset.target.name
    print(f"Columns Before dtypes:\n{train.dtypes}")
    test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)
    X_train = train.drop(columns=label)
    y_train = train[label]
    X_test = test.drop(columns=label)
    y_test = test[label]
    # Objects happen in automlbenchmark -- fix that
    X_train = X_train.convert_dtypes()
    for x in X_train.columns:
        X_test[x] = X_test[x].astype(X_train[x].dtypes.name)
        if not is_numeric_dtype(X_test[x]):
            X_test[x] = X_test[x].astype('category')
            X_train[x] = X_train[x].astype('category')
    print(f"Columns dtypes after fix :\n{X_train.dtypes}")

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    ml_memory_limit = config.framework_params.get('_ml_memory_limit', 'auto')

    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = utils.system_memory_mb().total
    if ml_memory_limit == 'auto':
        ml_memory_limit = max(
            min(
                config.max_mem_size_mb / n_jobs,
                math.ceil(total_memory_mb / n_jobs)
            ),
            3072  # 3072 is AutoPyTorch default and we use it as a lower bound
        )

    print("Using %sMB memory per ML job and %sMB for ensemble job on a total of %s jobs.",
             ml_memory_limit, ml_memory_limit, n_jobs)

    # TODO: do we need to set per_run_time_limit too?
    estimator = TabularClassificationTask if is_classification else NotImplementedError()
    api = estimator(n_jobs=n_jobs,
                    delete_tmp_folder_after_terminate=False,
                    **training_params)
    print(f"X_train={X_train.iloc[0]} X_test={X_test.iloc[0]}")
    with utils.Timer() as training:
        api.search(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            optimize_metric=perf_metric.name,
            # Be a little bit pessimistic towards time
            total_walltime_limit=config.max_runtime_seconds-10,
        )

    # Convert output to strings for classification
    #log.info("Predicting on the test set.")
    print("Predicting on the test set.")
    predictions = api.predict(X_test)
    probabilities = api.predict_proba(X_test) if is_classification else None

    print("Saving Artifacts")
    save_artifacts(api, config)
    print("Done saving Artifacts")

    try:
        print(f"The trajectory: ")
        print(api.run_history, api.trajectory)
        print(f"The selected models: ")
        print(api.show_models())
        print(f"Finish the run")
    except Exception as e:
        print(f"Run into {e} while printing information")

    # Looks like pynisher is not able to kill all jobs
    # Terminate them with brute-force and debug how to improve this
    import psutil
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        print('Child pid is {}:{}:{}'.format(child.pid, vars(child), vars(child)['_exitcode']))
        child.terminate()
        os.kill(child.pid, 9)
    def on_terminate(proc):
        print("process {} terminated with exit code {}".format(proc, proc.returncode))
    gone, alive = psutil.wait_procs(children, timeout=3, callback=on_terminate)
    print(f"gone={gone} and alive={alive}")

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  # TODO
                  models_count=len([]),
                  training_duration=training.duration)


def save_artifacts(estimator, config):
    try:
        #models_repr = estimator.show_models()
        #log.debug("Trained Ensemble:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
        if 'debug' in artifacts:
            print("Saving Artifacts -- debug")
            debug_dir = output_subdir('debug', config)
            ignore_extensions = ['.npy', '.pcs', '.model', '.ensemble', '.pkl', '.cv_model', '.pth']
            tmp_directory = estimator._backend.temporary_directory
            files_to_copy = []
            for r, d, f in os.walk(tmp_directory):
                for file_name in f:
                    base, ext = os.path.splitext(file_name)
                    if ext not in ignore_extensions:
                        files_to_copy.append(os.path.join(r, file_name))
            for filename in files_to_copy:
                dst = filename.replace(tmp_directory, debug_dir+'/')
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copyfile(filename, dst)
    except Exception as e:
        log.debug(f"Error when saving artifacts= {e}.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
