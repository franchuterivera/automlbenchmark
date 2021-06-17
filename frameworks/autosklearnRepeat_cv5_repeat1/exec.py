import pandas as pd
from shutil import copyfile
import logging
import math
import os
import tempfile as tmp
import warnings
import pickle

import numpy as np

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import autosklearn
from autosklearn.estimators import AutoSklearnClassifier, AutoSklearnRegressor
import autosklearn.metrics as metrics
from packaging import version

from frameworks.shared.callee import call_run, result, output_subdir, utils

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** AutoSklearn {} ****\n".format(autosklearn.__version__))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    # Mapping of benchmark metrics to autosklearn metrics
    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        balacc=metrics.balanced_accuracy,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        rmse=metrics.mean_squared_error,  # autosklearn can optimize on mse, and we compute rmse independently on predictions
        r2=metrics.r2
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("Running auto-sklearn with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric)
    log.info("Environment: %s", os.environ)

    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    X_test = dataset.test.X_enc
    y_test = dataset.test.y_enc
    predictors_type = dataset.predictors_type
    log.debug("predictors_type=%s", predictors_type)
    # log.info("finite=%s", np.isfinite(X_train))

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}
    seed = 42
    repeats = 1
    if 'repeats' in training_params:
        repeats = int(training_params.pop('repeats'))
    folds = 5
    if 'folds' in training_params:
        folds = int(training_params.pop('folds'))

    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    memory_limit = config.framework_params.get('_memory_limit', 'auto')

    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = utils.system_memory_mb().total
    if memory_limit == 'auto':
        memory_limit = max(min(math.ceil(config.max_mem_size_mb / n_jobs),
                               math.ceil(total_memory_mb / n_jobs)),
                               3072)  # 3072 is autosklearn defaults
    log.info("Using %sMB memory per ML job and %sMB for ensemble job on a total of %s jobs.", memory_limit, memory_limit, n_jobs)

    log.warning("Using meta-learned initialization, which might be bad (leakage).")
    # TODO: do we need to set per_run_time_limit too?
    estimator = AutoSklearnClassifier if is_classification else AutoSklearnRegressor

    if version.parse(autosklearn.__version__) >= version.parse("0.8"):
        constr_extra_params = dict(metric=perf_metric)
        fit_extra_params = {}
    else:
        constr_extra_params = {}
        fit_extra_params = dict(metric=perf_metric)

    auto_sklearn = estimator(time_left_for_this_task=config.max_runtime_seconds,
                             n_jobs=n_jobs,
                             memory_limit=memory_limit,
                             resampling_strategy='cv',
                             resampling_strategy_arguments={'folds': folds, 'repeats': repeats},
                             delete_tmp_folder_after_terminate=False,
                             delete_output_folder_after_terminate=False,
                             seed=seed,
                             **constr_extra_params,
                             **training_params)
    with utils.Timer() as training:
        auto_sklearn.fit(X_train, y_train, X_test, y_test, feat_type=predictors_type, **fit_extra_params)

    # Convert output to strings for classification
    log.info("Predicting on the test set.")
    predictions = auto_sklearn.predict(X_test)
    probabilities = auto_sklearn.predict_proba(X_test) if is_classification else None

    overfit_frame = generate_overfit_artifacts(auto_sklearn, X_train, y_train, X_test, y_test)

    save_artifacts(auto_sklearn, config, overfit_frame)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(auto_sklearn.get_models_with_weights()),
                  training_duration=training.duration)


def generate_overfit_artifacts(estimator, X_train, y_train, X_test, y_test):
    dataframe = []
    run_keys = [v for v in estimator.automl_.runhistory_.data.values() if v.additional_info and 'train_loss' in v.additional_info]
    best_validation_index = np.argmin([v.cost for v in run_keys])
    val_score = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_validation_index].cost)
    train_score = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_validation_index].additional_info['train_loss'])
    test_score = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_validation_index].additional_info['test_loss'])
    dataframe.append({
        'model': 'best_individual_model',
        'test': test_score,
        'val': val_score,
        'train': train_score,
    })

    best_test_index = np.argmin([v.additional_info['test_loss'] for v in run_keys])
    val_score2 = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_test_index].cost)
    train_score2 = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_test_index].additional_info['train_loss'])
    test_score2 = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_test_index].additional_info['test_loss'])
    dataframe.append({
        'model': 'best_ever_test_score_individual_model',
        'test': test_score2,
        'val': val_score2,
        'train': train_score2,
    })

    best_ensemble_index = np.argmax([v['ensemble_optimization_score'] for v in estimator.automl_.ensemble_performance_history])
    dataframe.append({
        'model': 'best_ensemble_model',
        'test': estimator.automl_.ensemble_performance_history[best_ensemble_index]['ensemble_test_score'],
        'val': np.inf,
        'train': estimator.automl_.ensemble_performance_history[best_ensemble_index]['ensemble_optimization_score'],
    })

    best_ensemble_index_test = np.argmax([v['ensemble_test_score'] for v in estimator.automl_.ensemble_performance_history])
    dataframe.append({
        'model': 'best_ever_test_score_ensemble_model',
        'test': estimator.automl_.ensemble_performance_history[best_ensemble_index_test]['ensemble_test_score'],
        'val': np.inf,
        'train': estimator.automl_.ensemble_performance_history[best_ensemble_index_test]['ensemble_optimization_score'],
    })

    try:
        dataframe.append({
            'model': 'rescore_final',
            'test': estimator.score(X_test, y_test),
            'val': np.inf,
            'train': estimator.score(X_train, y_train),
        })
    except Exception as e:
        print(e)
    return pd.DataFrame(dataframe)


def save_artifacts(estimator, config, overfit_frame):
    try:
        models_repr = estimator.show_models()
        log.debug("Trained Ensemble:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
        if 'overfit' in artifacts:
            overfit_file = os.path.join(output_subdir('overfit', config), 'overfit.csv')
            overfit_frame.to_csv(overfit_file)
        if 'debug' in artifacts:
            print('Saving debug artifacts!')
            debug_dir = output_subdir('debug', config)
            ignore_extensions = ['.npy', '.pcs', '.model', '.ensemble', '.pkl']
            tmp_directory = estimator.automl_._backend.temporary_directory
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
            # save the ensemble performance history file
            ensemble_performance_frame = pd.DataFrame(estimator.automl_.ensemble_performance_history)
            ensemble_performance_frame.to_csv(os.path.join(debug_dir, 'ensemble_history.csv'))
    except Exception as e:
        log.debug("Error when saving artifacts= {e}.".format(e), exc_info=True)


if __name__ == '__main__':
    call_run(run)
