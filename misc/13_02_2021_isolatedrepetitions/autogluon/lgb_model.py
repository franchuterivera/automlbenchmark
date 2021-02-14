import gc
import logging
import os
import random
import re
import time
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from abstract_model import AbstractModel, fixedvals_from_searchspaces, accuracy

import collections, warnings, time, os, psutil, logging
from operator import gt, lt

from lightgbm.callback import _format_eval_result, EarlyStopException
import lightgbm as lgb

logger = logging.getLogger(__name__)

DEFAULT_NUM_BOOST_ROUND = 10000  # default for single training run


def get_param_baseline_custom(problem_type, num_classes=None):
    if problem_type == 'binary':
        return get_param_binary_baseline_custom()
    elif problem_type == 'multiclass':
        return get_param_multiclass_baseline_custom(num_classes=num_classes)
    else:
        return get_param_binary_baseline_custom()


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == 'binary':
        return get_param_binary_baseline()
    elif problem_type == 'multiclass':
        return get_param_multiclass_baseline(num_classes=num_classes)
    else:
        return get_param_binary_baseline()


def get_param_multiclass_baseline_custom(num_classes):
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'multiclass',
        'num_classes': num_classes,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
        'two_round': True,
        'seed_value': 0,
        # 'device': 'gpu'  # needs GPU-enabled lightGBM build
        # TODO: Bin size max increase
    }
    return params.copy()


def get_param_binary_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_multiclass_baseline(num_classes):
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'multiclass',
        'num_classes': num_classes,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_regression_baseline():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'regression',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
    }
    return params


def get_param_binary_baseline_dummy_gpu():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'two_round': True,
        'device_type': 'gpu',
    }
    return params


def get_param_binary_baseline_custom():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'binary',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        # 'is_unbalance': True,  # TODO: Set is_unbalanced: True for F1-score, AUC!
        'two_round': True,
        'seed_value': 0,
    }
    return params.copy()


def get_param_regression_baseline_custom():
    params = {
        'num_boost_round': DEFAULT_NUM_BOOST_ROUND,
        'num_threads': -1,
        'objective': 'regression',
        'verbose': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 5,
        'two_round': True,
        'seed_value': 0,
    }
    return params.copy()


def get_param_softclass_baseline(num_classes):
    params = get_param_multiclass_baseline(num_classes)
    params.pop('metric', None)
    return params.copy()


def get_param_softclass_baseline_custom(num_classes):
    params = get_param_multiclass_baseline_custom(num_classes)
    params.pop('metric', None)
    return params.copy()


# TODO: Add option to stop if current run's metric value is X% lower, such as min 30%, current 40% -> Stop
def early_stopping_custom(stopping_rounds, first_metric_only=False, metrics_to_use=None, start_time=None, time_limit=None, verbose=True, max_diff=None, ignore_dart_warning=False, manual_stop_file=None, train_loss_name=None, reporter=None):
    """Create a callback that activates early stopping.
    Note
    ----
    Activates early stopping.
    The model will train until the validation score stops improving.
    Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
    to continue training.
    Requires at least one validation data and one metric.
    If there's more than one, will check all of them. But the training data is ignored anyway.
    To check only the first metric set ``first_metric_only`` to True.
    Parameters
    ----------
    stopping_rounds : int
       The possible number of rounds without the trend occurrence.
    first_metric_only : bool, optional (default=False)
       Whether to use only the first metric for early stopping.
    verbose : bool, optional (default=True)
        Whether to print message with early stopping information.
    train_loss_name : str, optional (default=None):
        Name of metric that contains training loss value.
    reporter : optional (default=None):
        reporter object from AutoGluon scheduler.
    Returns
    -------
    callback : function
        The callback that activates early stopping.
    """
    best_score = []
    best_iter = []
    best_score_list = []
    best_trainloss = []  # stores training losses at corresponding best_iter
    cmp_op = []
    enabled = [True]
    indices_to_check = []
    timex = [time.time()]
    mem_status = psutil.Process()
    init_mem_rss = []
    init_mem_avail = []

    def _init(env):
        if not ignore_dart_warning:
            enabled[0] = not any((boost_alias in env.params
                                  and env.params[boost_alias] == 'dart') for boost_alias in ('boosting',
                                                                                             'boosting_type',
                                                                                             'boost'))
        if not enabled[0]:
            warnings.warn('Early stopping is not available in dart mode')
            return
        if not env.evaluation_result_list:
            raise ValueError('For early stopping, '
                             'at least one dataset and eval metric is required for evaluation')

        if verbose:
            msg = "Training until validation scores don't improve for {} rounds."
            logger.debug(msg.format(stopping_rounds))
            if manual_stop_file:
                logger.debug('Manually stop training by creating file at location: ', manual_stop_file)

        for eval_ret in env.evaluation_result_list:
            best_iter.append(0)
            best_score_list.append(None)
            best_trainloss.append(None)
            if eval_ret[3]:
                best_score.append(float('-inf'))
                cmp_op.append(gt)
            else:
                best_score.append(float('inf'))
                cmp_op.append(lt)

        if metrics_to_use is None:
            for i in range(len(env.evaluation_result_list)):
                indices_to_check.append(i)
                if first_metric_only:
                    break
        else:
            for i, eval in enumerate(env.evaluation_result_list):
                if (eval[0], eval[1]) in metrics_to_use:
                    indices_to_check.append(i)
                    if first_metric_only:
                        break

        init_mem_rss.append(mem_status.memory_info().rss)
        init_mem_avail.append(4469755084)

    def _callback(env):
        if not cmp_op:
            _init(env)
        if not enabled[0]:
            return
        if train_loss_name is not None:
            train_loss_evals = [eval for eval in env.evaluation_result_list if eval[0] == 'train_set' and eval[1] == train_loss_name]
            train_loss_val = train_loss_evals[0][2]
        else:
            train_loss_val = 0.0
        for i in indices_to_check:
            score = env.evaluation_result_list[i][2]
            if best_score_list[i] is None or cmp_op[i](score, best_score[i]):
                best_score[i] = score
                best_iter[i] = env.iteration
                best_score_list[i] = env.evaluation_result_list
                best_trainloss[i] = train_loss_val
            if reporter is not None:  # Report current best scores for iteration, used in HPO
                if i == indices_to_check[0]:  # TODO: documentation needs to note that we assume 0th index is the 'official' validation performance metric.
                    if cmp_op[i] == gt:
                        validation_perf = score
                    else:
                        validation_perf = -score
                    reporter(epoch=env.iteration + 1,
                             validation_performance=validation_perf,
                             train_loss=best_trainloss[i],
                             best_iter_sofar=best_iter[i] + 1,
                             best_valperf_sofar=best_score[i])
            if env.iteration - best_iter[i] >= stopping_rounds:
                if verbose:
                    logger.log(15, 'Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            elif (max_diff is not None) and (abs(score - best_score[i]) > max_diff):
                if verbose:
                    logger.debug('max_diff breached!')
                    logger.debug(abs(score - best_score[i]))
                    logger.log(15, 'Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if env.iteration == env.end_iteration - 1:
                if verbose:
                    logger.log(15, 'Did not meet early stopping criterion. Best iteration is:\n[%d]\t%s' % (
                        best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
            if verbose:
                logger.debug((env.iteration - best_iter[i], env.evaluation_result_list[i]))
        if manual_stop_file:
            if os.path.exists(manual_stop_file):
                i = indices_to_check[0]
                logger.log(20, 'Found manual stop file, early stopping. Best iteration is:\n[%d]\t%s' % (
                    best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])
        if time_limit:
            time_elapsed = time.time() - start_time
            time_left = time_limit - time_elapsed
            if time_left <= 0:
                i = indices_to_check[0]
                logger.log(20, '\tRan out of time, early stopping on iteration ' + str(env.iteration+1) + '. Best iteration is:\n\t[%d]\t%s' % (
                    best_iter[i] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[i]])))
                raise EarlyStopException(best_iter[i], best_score_list[i])

        # TODO: Add toggle parameter to early_stopping to disable this
        # TODO: Identify optimal threshold values for early_stopping based on lack of memory
        if env.iteration % 10 == 0:
            available = 4469755084
            cur_rss = mem_status.memory_info().rss

            if cur_rss < init_mem_rss[0]:
                init_mem_rss[0] = cur_rss
            estimated_model_size_mb = (cur_rss - init_mem_rss[0]) >> 20
            available_mb = available >> 20

            model_size_memory_ratio = estimated_model_size_mb / available_mb
            if verbose or (model_size_memory_ratio > 0.25):
                logging.debug('Available Memory: '+str(available_mb)+' MB')
                logging.debug('Estimated Model Size: '+str(estimated_model_size_mb)+' MB')

            early_stop = False
            if model_size_memory_ratio > 1.0:
                logger.warning('Warning: Large GBM model size may cause OOM error if training continues')
                logger.warning('Available Memory: '+str(available_mb)+' MB')
                logger.warning('Estimated GBM model size: '+str(estimated_model_size_mb)+' MB')
                early_stop = True

            # TODO: We will want to track size of model as well, even if we early stop before OOM, we will still crash when saving if the model is large enough
            if available_mb < 512:  # Less than 500 MB
                logger.warning('Warning: Low available memory may cause OOM error if training continues')
                logger.warning('Available Memory: '+str(available_mb)+' MB')
                logger.warning('Estimated GBM model size: '+str(estimated_model_size_mb)+' MB')
                early_stop = True

            if early_stop:
                logger.warning('Warning: Early stopped GBM model prior to optimal result to avoid OOM error. Please increase available memory to avoid subpar model quality.')
                logger.log(15, 'Early stopping, best iteration is:\n[%d]\t%s' % (
                        best_iter[0] + 1, '\t'.join([_format_eval_result(x) for x in best_score_list[0]])))
                raise EarlyStopException(best_iter[0], best_score_list[0])

    _callback.order = 30
    return _callback





# Mapping to specialized LightGBM metrics that are much faster than the standard metric computation
_ag_to_lgbm_metric_dict = {
    'binary': dict(
        accuracy='binary_error',
        balanced_accuracy='binary_error',
        log_loss='binary_logloss',
        roc_auc='auc',
    ),
    'multiclass': dict(
        accuracy='multi_error',
        balanced_accuracy='multi_error',
        log_loss='multi_logloss',
    ),
}


def convert_ag_metric_to_lgbm(ag_metric_name, problem_type):
    return _ag_to_lgbm_metric_dict.get(problem_type, dict()).get(ag_metric_name, None)


def func_generator(metric, is_higher_better, needs_pred_proba, problem_type):
    if needs_pred_proba:
        if problem_type == 'multiclass':
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.reshape(len(np.unique(y_true)), -1).T
                return metric.name, metric(y_true, y_hat), is_higher_better
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                return metric.name, metric(y_true, y_hat), is_higher_better
    else:
        if problem_type == 'multiclass':
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = y_hat.reshape(len(np.unique(y_true)), -1)
                y_hat = y_hat.argmax(axis=0)
                return metric.name, metric(y_true, y_hat), is_higher_better
        else:
            def function_template(y_hat, data):
                y_true = data.get_label()
                y_hat = np.round(y_hat)
                return metric.name, metric(y_true, y_hat), is_higher_better
    return function_template


def softclass_lgbobj(preds, train_data):
    """ Custom LightGBM loss function for soft (probabilistic, vector-valued) class-labels only,
        which have been appended to lgb.Dataset (train_data) as additional ".softlabels" attribute (2D numpy array).
    """
    softlabels = train_data.softlabels
    num_classes = softlabels.shape[1]
    preds=np.reshape(preds, (len(softlabels), num_classes), order='F')
    preds = np.exp(preds)
    preds = np.multiply(preds, 1/np.sum(preds, axis=1)[:, np.newaxis])
    grad = (preds - softlabels)
    hess = 2.0 * preds * (1.0-preds)
    return grad.flatten('F'), hess.flatten('F')


def construct_dataset(x: DataFrame, y: Series, location=None, reference=None, params=None, save=False, weight=None):

    dataset = lgb.Dataset(data=x, label=y, reference=reference, free_raw_data=True, params=params, weight=weight)

    if save:
        assert location is not None
        saving_path = f'{location}.bin'
        if os.path.exists(saving_path):
            os.remove(saving_path)

        os.makedirs(os.path.dirname(saving_path), exist_ok=True)
        dataset.save_binary(saving_path)
        # dataset_binary = lgb.Dataset(location + '.bin', reference=reference, free_raw_data=False)# .construct()

    return dataset













warnings.filterwarnings("ignore", category=UserWarning, message="Starting from version")  # lightGBM brew libomp warning
logger = logging.getLogger(__name__)


# TODO: Save dataset to binary and reload for HPO. This will avoid the memory spike overhead when training each model and instead it will only occur once upon saving the dataset.
class LGBModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._internal_feature_map = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type, num_classes=self.num_classes)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # Use specialized LightGBM metric if available (fast), otherwise use custom func generator
    def get_eval_metric(self):
        eval_metric = convert_ag_metric_to_lgbm(ag_metric_name=self.stopping_metric.name, problem_type=self.problem_type)
        if eval_metric is None:
            eval_metric = func_generator(metric=self.stopping_metric, is_higher_better=True, needs_pred_proba=not self.stopping_metric.needs_y_pred, problem_type=self.problem_type)
            eval_metric_name = self.stopping_metric.name
        else:
            eval_metric_name = eval_metric
        return eval_metric, eval_metric_name

    def _fit(self, X_train=None, y_train=None, X_val=None, y_val=None, dataset_train=None, dataset_val=None, time_limit=None, num_gpus=0, **kwargs):
        start_time = time.time()
        params = self.params.copy()

        self.total_number_of_classes = len(np.unique(y_train))
        if 'multiclass' in self.problem_type:
            params['num_classes'] = len(np.unique(y_train))

        # TODO: kwargs can have num_cpu, num_gpu. Currently these are ignored.
        verbosity = kwargs.get('verbosity', 2)
        params = fixedvals_from_searchspaces(params)

        if verbosity <= 1:
            verbose_eval = False
        elif verbosity == 2:
            verbose_eval = 1000
        elif verbosity == 3:
            verbose_eval = 50
        else:
            verbose_eval = 1

        eval_metric, eval_metric_name = self.get_eval_metric()
        dataset_train, dataset_val = self.generate_datasets(X_train=X_train, y_train=y_train, params=params, X_val=X_val, y_val=y_val, dataset_train=dataset_train, dataset_val=dataset_val)
        gc.collect()

        num_boost_round = params.pop('num_boost_round', 1000)
        dart_retrain = params.pop('dart_retrain', False)  # Whether to retrain the model to get optimal iteration if model is trained in 'dart' mode.
        if num_gpus != 0:
            if 'device' not in params:
                # TODO: lightgbm must have a special install to support GPU: https://github.com/Microsoft/LightGBM/tree/master/python-package#build-gpu-version
                #  Before enabling GPU, we should add code to detect that GPU-enabled version is installed and that a valid GPU exists.
                #  GPU training heavily alters accuracy, often in a negative manner. We will have to be careful about when to use GPU.
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # params['device'] = 'gpu'
                pass
        logger.log(15, f'Training Gradient Boosting Model for {num_boost_round} rounds...')
        logger.log(15, "with the following hyperparameter settings:")
        logger.log(15, params)

        num_rows_train = len(dataset_train.data)
        if 'min_data_in_leaf' in params:
            if params['min_data_in_leaf'] > num_rows_train:  # TODO: may not be necessary
                params['min_data_in_leaf'] = max(1, int(num_rows_train / 5.0))

        # TODO: Better solution: Track trend to early stop when score is far worse than best score, or score is trending worse over time
        if (dataset_val is not None) and (dataset_train is not None):
            modifier = 1 if num_rows_train <= 10000 else 10000 / num_rows_train
            early_stopping_rounds = max(round(modifier * 150), 10)
        else:
            early_stopping_rounds = 150

        callbacks = []
        valid_names = ['train_set']
        valid_sets = [dataset_train]
        if dataset_val is not None:
            reporter = kwargs.get('reporter', None)
            train_loss_name = self._get_train_loss_name() if reporter is not None else None
            if train_loss_name is not None:
                if 'metric' not in params or params['metric'] == '':
                    params['metric'] = train_loss_name
                elif train_loss_name not in params['metric']:
                    params['metric'] = f'{params["metric"]},{train_loss_name}'
            callbacks += [
                # Note: Don't use self.params_aux['max_memory_usage_ratio'] here as LightGBM handles memory per iteration optimally.  # TODO: Consider using when ratio < 1.
                early_stopping_custom(early_stopping_rounds, metrics_to_use=[('valid_set', eval_metric_name)], max_diff=None, start_time=start_time, time_limit=time_limit,
                                      ignore_dart_warning=True, verbose=False, manual_stop_file=False, reporter=reporter, train_loss_name=train_loss_name),
            ]
            valid_names = ['valid_set'] + valid_names
            valid_sets = [dataset_val] + valid_sets

        seed_val = params.pop('seed_value', 0)
        train_params = {
            'params': params,
            'train_set': dataset_train,
            'num_boost_round': num_boost_round,
            'valid_sets': valid_sets,
            'valid_names': valid_names,
            'callbacks': callbacks,
            'verbose_eval': verbose_eval,
        }
        if not isinstance(eval_metric, str):
            train_params['feval'] = eval_metric
        else:
            if 'metric' not in train_params['params'] or train_params['params']['metric'] == '':
                train_params['params']['metric'] = eval_metric
            elif eval_metric not in train_params['params']['metric']:
                train_params['params']['metric'] = f'{train_params["params"]["metric"]},{eval_metric}'
        if seed_val is not None:
            train_params['params']['seed'] = seed_val
            random.seed(seed_val)
            np.random.seed(seed_val)

        # Train LightGBM model:
        with warnings.catch_warnings():
            # Filter harmless warnings introduced in lightgbm 3.0, future versions plan to remove: https://github.com/microsoft/LightGBM/issues/3379
            warnings.filterwarnings('ignore', message='Overriding the parameters from Reference Dataset.')
            warnings.filterwarnings('ignore', message='categorical_column in param dict is overridden.')
            self.model = lgb.train(**train_params)
            retrain = False
            if train_params['params'].get('boosting_type', '') == 'dart':
                if dataset_val is not None and dart_retrain and (self.model.best_iteration != num_boost_round):
                    retrain = True
                    if time_limit is not None:
                        time_left = time_limit + start_time - time.time()
                        if time_left < 0.5 * time_limit:
                            retrain = False
                    if retrain:
                        logger.log(15, f"Retraining LGB model to optimal iterations ('dart' mode).")
                        train_params.pop('callbacks')
                        train_params['num_boost_round'] = self.model.best_iteration
                        self.model = lgb.train(**train_params)
                    else:
                        logger.log(15, f"Not enough time to retrain LGB model ('dart' mode)...")

        if dataset_val is not None and not retrain:
            self.params_trained['num_boost_round'] = self.model.best_iteration
        else:
            self.params_trained['num_boost_round'] = self.model.current_iteration()

    def _predict_proba(self, X, preprocess=True):
        if preprocess:
            X = self.preprocess(X)

        y_pred_proba = self.model.predict(X)
        if self.problem_type == 'binary':
            if len(y_pred_proba.shape) == 1:
                #return np.eye(self.total_number_of_classes)[np.round(np.expand_dims(y_pred_proba, axis=1)).astype(np.int32)]
                y_pred_proba = np.round(y_pred_proba).astype(np.int32)
                y_pred_proba = np.eye(self.total_number_of_classes)[y_pred_proba]
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif self.problem_type == 'multiclass':
            return y_pred_proba
        else:
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 2:  # Should this ever happen?
                return y_pred_proba
            else:  # Should this ever happen?
                return y_pred_proba[:, 1]

    def preprocess(self, X, is_train=False):
        X = super().preprocess(X=X)

        if is_train:
            for column in X.columns:
                if isinstance(column, str):
                    new_column = re.sub(r'[",:{}[\]]', '', column)
                    if new_column != column:
                        self._internal_feature_map = {feature: i for i, feature in enumerate(list(X.columns))}
                        break

        if self._internal_feature_map:
            new_columns = [self._internal_feature_map[column] for column in list(X.columns)]
            X_new = X.copy(deep=False)
            X_new.columns = new_columns
            return X_new
        else:
            return X

    def generate_datasets(self, X_train: DataFrame, y_train: Series, params, X_val=None, y_val=None, dataset_train=None, dataset_val=None, save=False):
        lgb_dataset_params_keys = ['objective', 'two_round', 'num_threads', 'num_classes', 'verbose']  # Keys that are specific to lightGBM Dataset object construction.
        data_params = {key: params[key] for key in lgb_dataset_params_keys if key in params}.copy()

        W_train = None  # TODO: Add weight support
        W_test = None  # TODO: Add weight support
        if X_train is not None:
            X_train = self.preprocess(X_train, is_train=True)
        if X_val is not None:
            X_val = self.preprocess(X_val)
        # TODO: Try creating multiple Datasets for subsets of features, then combining with Dataset.add_features_from(), this might avoid memory spike

        y_train_og = None
        y_val_og = None

        if not dataset_train:
            # X_train, W_train = self.convert_to_weight(X=X_train)
            dataset_train = construct_dataset(x=X_train, y=y_train, location=f'{self.path}datasets{os.path.sep}train', params=data_params, save=save, weight=W_train)
            # dataset_train = construct_dataset_lowest_memory(X=X_train, y=y_train, location=self.path + 'datasets/train', params=data_params)
        if (not dataset_val) and (X_val is not None) and (y_val is not None):
            # X_val, W_val = self.convert_to_weight(X=X_val)
            dataset_val = construct_dataset(x=X_val, y=y_val, location=f'{self.path}datasets{os.path.sep}val', reference=dataset_train, params=data_params, save=save, weight=W_test)
            # dataset_val = construct_dataset_lowest_memory(X=X_val, y=y_val, location=self.path + 'datasets/val', reference=dataset_train, params=data_params)
        return dataset_train, dataset_val

    def debug_features_to_use(self, X_val_in):
        feature_splits = self.model.feature_importance()
        total_splits = feature_splits.sum()
        feature_names = list(X_val_in.columns.values)
        feature_count = len(feature_names)
        feature_importances = pd.DataFrame(data=feature_names, columns=['feature'])
        feature_importances['splits'] = feature_splits
        feature_importances_unused = feature_importances[feature_importances['splits'] == 0]
        feature_importances_used = feature_importances[feature_importances['splits'] >= (total_splits / feature_count)]
        logger.debug(feature_importances_unused)
        logger.debug(feature_importances_used)
        logger.debug(f'feature_importances_unused: {len(feature_importances_unused)}')
        logger.debug(f'feature_importances_used: {len(feature_importances_used)}')
        features_to_use = list(feature_importances_used['feature'].values)
        logger.debug(str(features_to_use))
        return features_to_use

    # TODO: Consider adding _internal_feature_map functionality to abstract_model
    def compute_feature_importance(self, **kwargs) -> pd.Series:
        feature_importances = super().compute_feature_importance(**kwargs)
        if self._internal_feature_map is not None:
            inverse_internal_feature_map = {i: feature for feature, i in self._internal_feature_map.items()}
            feature_importances = {inverse_internal_feature_map[i]: importance for i, importance in feature_importances.items()}
            feature_importances = pd.Series(data=feature_importances)
            feature_importances = feature_importances.sort_values(ascending=False)
        return feature_importances

    def _get_train_loss_name(self):
        if self.problem_type == 'binary':
            train_loss_name = 'binary_logloss'
        elif self.problem_type == 'multiclass':
            train_loss_name = 'multi_logloss'
        else:
            raise ValueError(f"unknown problem_type for LGBModel: {self.problem_type}")
        return train_loss_name

    def get_model_feature_importance(self, use_original_feature_names=False):
        feature_names = self.model.feature_name()
        importances = self.model.feature_importance()
        importance_dict = {feature_name: importance for (feature_name, importance) in zip(feature_names, importances)}
        if use_original_feature_names and (self._internal_feature_map is not None):
            inverse_internal_feature_map = {i: feature for feature, i in self._internal_feature_map.items()}
            importance_dict = {inverse_internal_feature_map[i]: importance for i, importance in importance_dict.items()}
        return importance_dict

if __name__ == '__main__':
    import sklearn.datasets
    from pandas.api.types import is_numeric_dtype
    from sklearn.utils.multiclass import type_of_target
    #return np.eye(self.total_number_of_classes)[np.round(np.expand_dims(y_pred_proba, axis=1)).astype(np.int32)]
    for task in [40981, 40975]:
        X, y = sklearn.datasets.fetch_openml(data_id=task, return_X_y=True, as_frame=True)
        X = X.convert_dtypes()
        for x in X.columns:
            if not is_numeric_dtype(X[x]):
                X[x] = X[x].astype('category')
            elif 'Int' in str(X[x].dtype):
                X[x] = X[x].astype(np.int)
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y = pd.DataFrame(y)

        problem_type = type_of_target(y)
        print(f"X={X.dtypes} problem={problem_type} {y}")
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
        model = LGBModel(path='/tmp/', problem_type=problem_type, metric=accuracy, name='LGB', eval_metric=accuracy)
        model.fit(X_train=X_train, y_train=y_train)
        print(model.predict_proba(X_test).shape)
        print(accuracy(y_test.to_numpy().astype(int), model.predict(X_test)))
