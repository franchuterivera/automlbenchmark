from sklearn.utils.multiclass import type_of_target
import copy
import gc
import logging
import os
import pickle
import sys
import time
import warnings
from typing import Union

import numpy as np
import pandas as pd


class TimeLimitExceeded(Exception):
    pass


class NotEnoughMemoryError(Exception):
    pass


class NoGPUError(Exception):
    pass


class NoValidFeatures(Exception):
    pass


import copy
from abc import ABCMeta, abstractmethod
from functools import partial

import scipy
import scipy.stats
import sklearn.metrics


def get_pred_from_proba(y_pred_proba, problem_type):
    if problem_type == 'binary':
        y_pred = [1 if pred >= 0.5 else 0 for pred in y_pred_proba]
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred


class Scorer(object, metaclass=ABCMeta):
    def __init__(self, name, score_func, optimum, sign, kwargs):
        self.name = name
        self._kwargs = kwargs
        self._score_func = score_func
        self._optimum = optimum
        self._sign = sign
        self.alias = set()

    def add_alias(self, alias):
        if alias == self.name:
            raise ValueError(f'The alias "{alias}" is the same as the original name "{self.name}". '
                             f'This is not allowed.')
        self.alias.add(alias)

    @property
    def greater_is_better(self) -> bool:
        """Return whether the score is greater the better.
        We use the stored `sign` object to decide the property.
        Returns
        -------
        flag
            The "greater_is_better" flag.
        """
        return self._sign > 0

    def convert_score_to_sklearn_val(self, score):
        """Scores are always greater_is_better, this flips the sign of metrics who were originally lower_is_better."""
        return self._sign * score

    @abstractmethod
    def __call__(self, y_true, y_pred, sample_weight=None):
        pass

    def __repr__(self):
        return self.name

    def sklearn_scorer(self):
        ret = sklearn.metrics.scorer.make_scorer(score_func=self, greater_is_better=True, needs_proba=self.needs_proba, needs_threshold=self.needs_threshold)
        return ret

    @property
    @abstractmethod
    def needs_pred(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_proba(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def needs_threshold(self) -> bool:
        raise NotImplementedError


class _PredictScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted target values for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        type_true = type_of_target(y_true)

        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1 or type_true == 'continuous':
            pass  # must be regression, all other task types would return at least two probabilities
        elif type_true in ['binary', 'multiclass']:
            y_pred = np.argmax(y_pred, axis=1)
        elif type_true == 'multilabel-indicator':
            y_pred[y_pred > 0.5] = 1.0
            y_pred[y_pred <= 0.5] = 0.0
        else:
            raise ValueError(type_true)

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred,
                                                 **self._kwargs)

    @property
    def needs_pred(self):
        return True

    @property
    def needs_proba(self):
        return False

    @property
    def needs_threshold(self):
        return False


class _ProbaScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    @property
    def needs_pred(self):
        return False

    @property
    def needs_proba(self):
        return True

    @property
    def needs_threshold(self):
        return False


class _ThresholdScorer(Scorer):
    def __call__(self, y_true, y_pred, sample_weight=None):
        """Evaluate decision function output for X relative to y_true.
        Parameters
        ----------
        y_true : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        y_pred : array-like, [n_samples x n_classes]
            Model predictions
        sample_weight : array-like, optional (default=None)
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        if isinstance(y_true, list):
            y_true = np.array(y_true)
        if isinstance(y_pred, list):
            y_pred = np.array(y_pred)
        y_type = type_of_target(y_true)
        if y_type not in ("binary", "multilabel-indicator"):
            raise ValueError("{0} format is not supported".format(y_type))

        if y_type == "binary":
            pass
            # y_pred = y_pred[:, 1]
        elif isinstance(y_pred, list):
            y_pred = np.vstack([p[:, -1] for p in y_pred]).T

        if sample_weight is not None:
            return self._sign * self._score_func(y_true, y_pred,
                                                 sample_weight=sample_weight,
                                                 **self._kwargs)
        else:
            return self._sign * self._score_func(y_true, y_pred, **self._kwargs)

    @property
    def needs_pred(self):
        return False

    @property
    def needs_proba(self):
        return False

    @property
    def needs_threshold(self):
        return True


def scorer_expects_y_pred(scorer: Scorer):
    if isinstance(scorer, _ProbaScorer):
        return False
    elif isinstance(scorer, _ThresholdScorer):
        return False
    else:
        return True


def make_scorer(name, score_func, optimum=1, greater_is_better=True,
                needs_proba=False, needs_threshold=False, **kwargs) -> Scorer:
    """Make a scorer from a performance metric or loss function.
    Factory inspired by scikit-learn which wraps scikit-learn scoring functions
    to be used in auto-sklearn.
    Parameters
    ----------
    score_func : callable
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.
    optimum : int or float, default=1
        The best score achievable by the score function, i.e. maximum in case of
        scorer function and minimum in case of loss function.
    greater_is_better : boolean, default=True
        Whether score_func is a score function (default), meaning high is good,
        or a loss function, meaning low is good. In the latter case, the
        scorer object will sign-flip the outcome of the score_func.
    needs_proba : boolean, default=False
        Whether score_func requires predict_proba to get probability estimates
        out of a classifier.
    needs_threshold : boolean, default=False
        Whether score_func takes a continuous decision certainty.
        This only works for binary classification.
    **kwargs : additional arguments
        Additional parameters to be passed to score_func.
    Returns
    -------
    scorer
        Callable object that returns a scalar score; greater is better.
    """
    sign = 1 if greater_is_better else -1
    if needs_proba:
        cls = _ProbaScorer
    elif needs_threshold:
        cls = _ThresholdScorer
    else:
        cls = _PredictScorer
    return cls(name, score_func, optimum, sign, kwargs)


# Standard Classification Scores
accuracy = make_scorer('accuracy',
                       sklearn.metrics.accuracy_score)
balanced_accuracy = make_scorer('balanced_accuracy',
                       sklearn.metrics.balanced_accuracy_score)
accuracy.add_alias('balacc')


def customized_log_loss(y_true, y_pred, eps=1e-15):
    """
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels for n_samples samples.
    y_pred : array-like of float
        The predictions. shape = (n_samples, n_classes) or (n_samples,)
    eps : float
        The epsilon
    Returns
    -------
    loss
        The negative log-likelihood
    """
    assert y_true.ndim == 1
    if y_pred.ndim == 1:
        # First clip the y_pred which is also used in sklearn
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    else:
        assert y_pred.ndim == 2, 'Only ndim=2 is supported'
        labels = np.arange(y_pred.shape[1], dtype=np.int32)
        return sklearn.metrics.log_loss(y_true.astype(np.int32), y_pred,
                                        labels=labels,
                                        eps=eps)


# Score function for probabilistic classification
log_loss = make_scorer('log_loss',
                       customized_log_loss,
                       optimum=0,
                       greater_is_better=False,
                       needs_proba=True)
log_loss.add_alias('nll')


CLASSIFICATION_METRICS = dict()
for scorer in [balanced_accuracy, accuracy]:
    CLASSIFICATION_METRICS[scorer.name] = scorer
    for alias in scorer.alias:
        CLASSIFICATION_METRICS[alias] = scorer



def calculate_score(solution, prediction, task_type, metric,
                    all_scoring_functions=False):
    score = metric(solution, prediction)

    return score


def get_metric(metric, problem_type=None, metric_type=None) -> Scorer:
    """Returns metric function by using its name if the metric is str.
    Performs basic check for metric compatibility with given problem type."""
    all_available_metric_names = list(CLASSIFICATION_METRICS.keys())
    if metric is not None and isinstance(metric, str):
        if metric in CLASSIFICATION_METRICS:
            return CLASSIFICATION_METRICS[metric]
        else:
            raise ValueError(
                f"{metric} is an unknown metric, all available metrics are "
                f"'{all_available_metric_names}'. You can also refer to "
                f"autogluon.core.metrics to see how to define your own {metric_type} function"
            )
    else:
        return metric


class AbstractModel:
    """
    Abstract model implementation from which all AutoGluon models inherit.
    Parameters
    ----------
    path (str): directory where to store all outputs.
    name (str): name of subdirectory inside path where model will be saved.
    problem_type (str): type of problem this model will handle. Valid options: ['binary', 'multiclass', 'regression'].
    eval_metric (str or autogluon.core.metrics.Scorer): objective function the model intends to optimize. If None, will be inferred based on problem_type.
    hyperparameters (dict): various hyperparameters that will be used by model (can be search spaces instead of fixed values).
    feature_metadata (autogluon.core.features.feature_metadata.FeatureMetadata): contains feature type information that can be used to identify special features such as text ngrams and datetime as well as which features are numerical vs categorical
    """

    model_file_name = 'model.pkl'
    model_info_name = 'info.pkl'
    model_info_json_name = 'info.json'

    def __init__(self,
                 path: str,
                 name: str,
                 problem_type: str,
                 eval_metric: Union[str, Scorer] = None,
                 hyperparameters=None,
                 feature_metadata = None,
                 num_classes=None,
                 stopping_metric=None,
                 features=None,
                 **kwargs):

        self.name = name  # TODO: v0.1 Consider setting to self._name and having self.name be a property so self.name can't be set outside of self.rename()
        self.path_root = path
        self.path = self.create_contexts(self.path_root + self.path_suffix)  # TODO: Make this path a function for consistency.
        self.num_classes = num_classes
        self.model = None
        self.problem_type = problem_type
        self.eval_metric = eval_metric

        self.normalize_pred_probas = False

        if feature_metadata is not None:
            feature_metadata = copy.deepcopy(feature_metadata)
        self.feature_metadata = feature_metadata  # TODO: Should this be passed to a model on creation? Should it live in a Dataset object and passed during fit? Currently it is being updated prior to fit by trainer
        self.features = features

        self.fit_time = None  # Time taken to fit in seconds (Training data)
        self.predict_time = None  # Time taken to predict in seconds (Validation data)
        self.val_score = None  # Score with eval_metric (Validation data)

        self.params = {}
        self.params_aux = {}

        self._set_default_auxiliary_params()

        if stopping_metric is None:
            self.stopping_metric = self.params_aux.get('stopping_metric', self._get_default_stopping_metric())
        else:
            if 'stopping_metric' in self.params_aux:
                raise AssertionError('stopping_metric was specified in both hyperparameters ag_args_fit and model init. Please specify only once.')
            self.stopping_metric = stopping_metric
        self.stopping_metric = get_metric(self.stopping_metric, self.problem_type, 'stopping_metric')

        self._set_default_params()
        self.nondefault_params = []
        if hyperparameters is not None:
            self.params.update(hyperparameters)
            self.nondefault_params = list(hyperparameters.keys())[:]  # These are hyperparameters that user has specified.
        self.params_trained = dict()

    @property
    def path_suffix(self):
        return self.name + os.path.sep

    # Checks if model is capable of inference on new data (if normal model) or has produced out-of-fold predictions (if bagged model)
    def is_valid(self) -> bool:
        return self.is_fit()

    # Checks if model is capable of inference on new data
    def can_infer(self) -> bool:
        return self.is_valid()

    # Checks if a model has been fit
    def is_fit(self) -> bool:
        return self.model is not None

    # TODO: v0.1 update to be aligned with _set_default_auxiliary_params(), add _get_default_params()
    def _set_default_params(self):
        pass

    def _set_default_auxiliary_params(self):
        """
        Sets the default aux parameters of the model.
        This method should not be extended by inheriting models, instead extend _get_default_auxiliary_params.
        """
        # TODO: Consider adding to get_info() output
        default_auxiliary_params = self._get_default_auxiliary_params()
        for key, value in default_auxiliary_params.items():
            self._set_default_param_value(key, value, params=self.params_aux)

    # TODO: v0.1 consider adding documentation to each model highlighting which feature dtypes are valid
    def _get_default_auxiliary_params(self) -> dict:
        """
        Dictionary of auxiliary parameters that dictate various model-agnostic logic, such as:
            Which column dtypes are filtered out of the input data, or how much memory the model is allowed to use.
        """
        default_auxiliary_params = dict(
            max_memory_usage_ratio=1.0,  # Ratio of memory usage allowed by the model. Values > 1.0 have an increased risk of causing OOM errors. Used in memory checks during model training to avoid OOM errors.
            # TODO: Add more params
            # max_memory_usage=None,
            # max_disk_usage=None,
            max_time_limit_ratio=1.0,  # ratio of given time_limit to use during fit(). If time_limit == 10 and max_time_limit_ratio=0.3, time_limit would be changed to 3.
            max_time_limit=None,  # max time_limit value during fit(). If the provided time_limit is greater than this value, it will be replaced by max_time_limit. Occurs after max_time_limit_ratio is applied.
            min_time_limit=0,  # min time_limit value during fit(). If the provided time_limit is less than this value, it will be replaced by min_time_limit. Occurs after max_time_limit is applied.
            # num_cpus=None,
            # num_gpus=None,
            # ignore_hpo=False,
            # max_early_stopping_rounds=None,
            # TODO: add option for only top-k ngrams
            ignored_type_group_special=None,  # List, drops any features in `self.feature_metadata.type_group_map_special[type]` for type in `ignored_type_group_special`. | Currently undocumented in task.
            ignored_type_group_raw=None,  # List, drops any features in `self.feature_metadata.type_group_map_raw[type]` for type in `ignored_type_group_raw`. | Currently undocumented in task.
            get_features_kwargs=None,  # Kwargs for `autogluon.tabular.features.feature_metadata.FeatureMetadata.get_features()`. Overrides ignored_type_group_special and ignored_type_group_raw. | Currently undocumented in task.
            # TODO: v0.1 Document get_features_kwargs_extra in task.fit
            get_features_kwargs_extra=None,  # If not None, applies an additional feature filter to the result of get_feature_kwargs. This should be reserved for users and be None by default. | Currently undocumented in task.
        )
        return default_auxiliary_params

    def _set_default_param_value(self, param_name, param_value, params=None):
        if params is None:
            params = self.params
        if param_name not in params:
            params[param_name] = param_value

    def _get_default_searchspace(self) -> dict:
        """
        Get the default hyperparameter searchspace of the model.
        See `autogluon.core.space` for available space classes.
        Returns
        -------
        dict of hyperparameter search spaces.
        """
        return {}

    def _set_default_searchspace(self):
        """ Sets up default search space for HPO. Each hyperparameter which user did not specify is converted from
            default fixed value to default search space.
        """
        def_search_space = self._get_default_searchspace().copy()
        # Note: when subclassing AbstractModel, you must define or import get_default_searchspace() from the appropriate location.
        for key in self.nondefault_params:  # delete all user-specified hyperparams from the default search space
            def_search_space.pop(key, None)
        if self.params is not None:
            self.params.update(def_search_space)

    # TODO: v0.1 Change this to update path_root only, path change to property
    def set_contexts(self, path_context):
        self.path = self.create_contexts(path_context)
        self.path_root = self.path.rsplit(self.path_suffix, 1)[0]

    @staticmethod
    def create_contexts(path_context):
        path = path_context
        return path

    def rename(self, name: str):
        """Renames the model and updates self.path to reflect the updated name."""
        self.path = self.path[:-len(self.name) - 1] + name + os.path.sep
        self.name = name

    def preprocess(self, X, preprocess_nonadaptive=True, preprocess_stateful=True, **kwargs):
        X = self._preprocess(X, **kwargs)
        return X

    # TODO: Remove kwargs?
    def _preprocess(self, X: pd.DataFrame, **kwargs):
        """
        Data transformation logic should be added here.
        In bagged ensembles, preprocessing code that lives in `_preprocess` will be executed on each child model once per inference call.
        If preprocessing code could produce different output depending on the child model that processes the input data, then it must live here.
        When in doubt, put preprocessing code here instead of in `_preprocess_nonadaptive`.
        """
        return X

    def _preprocess_fit_args(self, **kwargs):
        time_limit = kwargs.get('time_limit', None)
        max_time_limit_ratio = self.params_aux.get('max_time_limit_ratio', 1)
        if time_limit is not None:
            time_limit *= max_time_limit_ratio
        max_time_limit = self.params_aux.get('max_time_limit', None)
        if max_time_limit is not None:
            if time_limit is None:
                time_limit = max_time_limit
            else:
                time_limit = min(time_limit, max_time_limit)
        min_time_limit = self.params_aux.get('min_time_limit', 0)
        if min_time_limit is None:
            time_limit = min_time_limit
        elif time_limit is not None:
            time_limit = max(time_limit, min_time_limit)
        kwargs['time_limit'] = time_limit
        kwargs = self._preprocess_fit_resources(**kwargs)
        return kwargs

    def _preprocess_fit_resources(self, silent=False, **kwargs):
        default_num_cpus, default_num_gpus = self._get_default_resources()
        num_cpus = self.params_aux.get('num_cpus', 'auto')
        num_gpus = self.params_aux.get('num_gpus', 'auto')
        kwargs['num_cpus'] = kwargs.get('num_cpus', num_cpus)
        kwargs['num_gpus'] = kwargs.get('num_gpus', num_gpus)
        if kwargs['num_cpus'] == 'auto':
            kwargs['num_cpus'] = default_num_cpus
        if kwargs['num_gpus'] == 'auto':
            kwargs['num_gpus'] = default_num_gpus
        return kwargs

    def fit(self, **kwargs):
        kwargs = self._preprocess_fit_args(**kwargs)
        if 'time_limit' not in kwargs or kwargs['time_limit'] is None or kwargs['time_limit'] > 0:
            self._fit(**kwargs)
        else:
            raise TimeLimitExceeded

    def _fit(self, X_train, y_train, **kwargs):
        # kwargs may contain: num_cpus, num_gpus
        X_train = self.preprocess(X_train)
        self.model = self.model.fit(X_train, y_train)

    def predict(self, X, **kwargs):
        y_pred_proba = self.predict_proba(X, **kwargs)
        y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
        return y_pred

    def predict_proba(self, X, normalize=None, **kwargs):
        if normalize is None:
            normalize = self.normalize_pred_probas
        y_pred_proba = self._predict_proba(X=X, **kwargs)
        if normalize:
            y_pred_proba = normalize_pred_probas(y_pred_proba, self.problem_type)
        y_pred_proba = y_pred_proba.astype(np.float32)
        return y_pred_proba

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X, **kwargs)

        y_pred_proba = self.model.predict_proba(X)
        if self.problem_type == 'binary':
            if len(y_pred_proba.shape) == 1:
                return y_pred_proba
            elif y_pred_proba.shape[1] > 1:
                return y_pred_proba[:, 1]
            else:
                return y_pred_proba
        elif y_pred_proba.shape[1] > 2:
            return y_pred_proba
        else:
            return y_pred_proba[:, 1]

    def score(self, X, y, eval_metric=None, metric_needs_y_pred=None, **kwargs):
        if eval_metric is None:
            eval_metric = self.eval_metric
        if metric_needs_y_pred is None:
            metric_needs_y_pred = eval_metric.needs_pred
        if metric_needs_y_pred:
            y_pred = self.predict(X=X, **kwargs)
            return eval_metric(y, y_pred)
        else:
            y_pred_proba = self.predict_proba(X=X, **kwargs)
            return eval_metric(y, y_pred_proba)

    def score_with_y_pred_proba(self, y, y_pred_proba, eval_metric=None, metric_needs_y_pred=None):
        if eval_metric is None:
            eval_metric = self.eval_metric
        if metric_needs_y_pred is None:
            metric_needs_y_pred = eval_metric.needs_pred
        if metric_needs_y_pred:
            y_pred = get_pred_from_proba(y_pred_proba=y_pred_proba, problem_type=self.problem_type)
            return eval_metric(y, y_pred)
        else:
            return eval_metric(y, y_pred_proba)

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        """
        Loads the model from disk to memory.
        Parameters
        ----------
        path : str
            Path to the saved model, minus the file name.
            This should generally be a directory path ending with a '/' character (or appropriate path separator value depending on OS).
            The model file is typically located in path + cls.model_file_name.
        reset_paths : bool, default True
            Whether to reset the self.path value of the loaded model to be equal to path.
            It is highly recommended to keep this value as True unless accessing the original self.path value is important.
            If False, the actual valid path and self.path may differ, leading to strange behaviour and potential exceptions if the model needs to load any other files at a later time.
        verbose : bool, default True
            Whether to log the location of the loaded file.
        Returns
        -------
        model : cls
            Loaded model object.
        """
        file_path = path + cls.model_file_name
        model = load_pkl.load(path=file_path, verbose=verbose)
        if reset_paths:
            model.set_contexts(path)
        return model

    # Compute feature importance via permutation importance
    # Note: Expensive to compute
    #  Time to compute is O(predict_time*num_features)
    def compute_permutation_importance(self, X, y, features: list, eval_metric=None, silent=False, **kwargs) -> pd.DataFrame:
        if eval_metric is None:
            eval_metric = self.eval_metric
        transform_func = self.preprocess
        if eval_metric.needs_pred:
            predict_func = self.predict
        else:
            predict_func = self.predict_proba
        transform_func_kwargs = dict(preprocess_stateful=False)
        predict_func_kwargs = dict(preprocess_nonadaptive=False)

        return compute_permutation_feature_importance(
            X=X, y=y, features=features, eval_metric=self.eval_metric, predict_func=predict_func, predict_func_kwargs=predict_func_kwargs,
            transform_func=transform_func, transform_func_kwargs=transform_func_kwargs, silent=silent, **kwargs
        )

    # Custom feature importance values for a model (such as those calculated from training)
    def get_model_feature_importance(self) -> dict:
        return dict()

    # Hyperparameters of trained model
    def get_trained_params(self) -> dict:
        trained_params = self.params.copy()
        trained_params.update(self.params_trained)
        return trained_params

    # After calling this function, returned model should be able to be fit as if it was new, as well as deep-copied.
    def convert_to_template(self):
        model = self.model
        self.model = None
        template = copy.deepcopy(self)
        template.reset_metrics()
        self.model = model
        return template

    # After calling this function, model should be able to be fit without test data using the iterations trained by the original model
    def convert_to_refit_full_template(self):
        params_trained = self.params_trained.copy()
        template = self.convert_to_template()
        template.params.update(params_trained)
        template.name = template.name + 'something'
        template.set_contexts(self.path_root + template.name + os.path.sep)
        return template

    def _get_init_args(self):
        hyperparameters = self.params.copy()
        hyperparameters = {key: val for key, val in hyperparameters.items() if key in self.nondefault_params}
        init_args = dict(
            path=self.path_root,
            name=self.name,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            num_classes=self.num_classes,
            stopping_metric=self.stopping_metric,
            model=None,
            hyperparameters=hyperparameters,
            features=self.features,
            feature_metadata=self.feature_metadata
        )
        return init_args

    # Resets metrics for the model
    def reset_metrics(self):
        self.fit_time = None
        self.predict_time = None
        self.val_score = None
        self.params_trained = dict()

    # TODO: Experimental, currently unused
    #  Has not been tested on Windows
    #  Does not work if model is located in S3
    #  Does not work if called before model was saved to disk (Will output 0)
    def get_disk_size(self) -> int:
        # Taken from https://stackoverflow.com/a/1392549
        from pathlib import Path
        model_path = Path(self.path)
        model_disk_size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
        return model_disk_size

    # TODO: This results in a doubling of memory usage of the model to calculate its size.
    #  If the model takes ~40%+ of memory, this may result in an OOM error.
    #  This is generally not an issue because the model already needed to do this when being saved to disk, so the error would have been triggered earlier.
    #  Consider using Pympler package for memory efficiency: https://pympler.readthedocs.io/en/latest/asizeof.html#asizeof
    def get_memory_size(self) -> int:
        gc.collect()  # Try to avoid OOM error
        return sys.getsizeof(pickle.dumps(self, protocol=4))

    # Removes non-essential objects from the model to reduce memory and disk footprint.
    # If `remove_fit=True`, enables the removal of variables which are required for fitting the model. If the model is already fully trained, then it is safe to remove these.
    # If `remove_info=True`, enables the removal of variables which are used during model.get_info(). The values will be None when calling model.get_info().
    # If `requires_save=True`, enables the removal of variables which are part of the model.pkl object, requiring an overwrite of the model to disk if it was previously persisted.
    def reduce_memory_size(self, remove_fit=True, remove_info=False, requires_save=True, **kwargs):
        pass

    # Deletes the model from disk.
    # WARNING: This will DELETE ALL FILES in the self.path directory, regardless if they were created by AutoGluon or not.
    #  DO NOT STORE FILES INSIDE OF THE MODEL DIRECTORY THAT ARE UNRELATED TO AUTOGLUON.
    def delete_from_disk(self):
        from pathlib import Path
        import shutil
        model_path = Path(self.path)
        # TODO: Report errors?
        shutil.rmtree(path=model_path, ignore_errors=True)

    def get_info(self) -> dict:
        info = {
            'name': self.name,
            'model_type': type(self).__name__,
            'problem_type': self.problem_type,
            'eval_metric': self.eval_metric.name,
            'stopping_metric': self.stopping_metric.name,
            'fit_time': self.fit_time,
            'num_classes': self.num_classes,
            'predict_time': self.predict_time,
            'val_score': self.val_score,
            'hyperparameters': self.params,
            'hyperparameters_fit': self.params_trained,  # TODO: Explain in docs that this is for hyperparameters that differ in final model from original hyperparameters, such as epochs (from early stopping)
            'hyperparameters_nondefault': self.nondefault_params,
            AG_ARGS_FIT: self.params_aux,
            'num_features': len(self.features) if self.features else None,
            'features': self.features,
            'feature_metadata': self.feature_metadata,
            # 'disk_size': self.get_disk_size(),
            'memory_size': self.get_memory_size(),  # Memory usage of model in bytes
        }
        return info

    @classmethod
    def load_info(cls, path, load_model_if_required=True) -> dict:
        load_path = path + cls.model_info_name
        try:
            return load_pkl.load(path=load_path)
        except:
            if load_model_if_required:
                model = cls.load(path=path, reset_paths=True)
                return model.get_info()
            else:
                raise

    def _get_default_resources(self):
        num_cpus = 1
        num_gpus = 0
        return num_cpus, num_gpus

    # TODO: v0.1 Add reference link to all valid keys and their usage or keep full docs here and reference elsewhere?
    @classmethod
    def _get_default_ag_args(cls) -> dict:
        """
        Dictionary of customization options related to meta properties of the model such as its name, the order it is trained, and the problem types it is valid for.
        """
        return {}

    def _get_default_stopping_metric(self):
        if self.eval_metric.name == 'roc_auc':
            stopping_metric = 'log_loss'
        else:
            stopping_metric = self.eval_metric
        stopping_metric = get_metric(stopping_metric, self.problem_type, 'stopping_metric')
        return stopping_metric


class AbstractNeuralNetworkModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._types_of_features = None

    # TODO: v0.1 clean method
    def _get_types_of_features(self, df, skew_threshold=None, embed_min_categories=None, use_ngram_features=None, needs_extra_types=True):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self._types_of_features is not None:
            Warning("Attempting to _get_types_of_features for Model, but previously already did this.")

        feature_types = self.feature_metadata.get_type_group_map_raw()
        categorical_featnames = feature_types['category'] + feature_types['object'] + feature_types['bool']
        continuous_featnames = feature_types['float'] + feature_types['int']  # + self.__get_feature_type_if_present('datetime')
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        valid_features = categorical_featnames + continuous_featnames + language_featnames

        if len(valid_features) < df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            df = df.drop(columns=unknown_features)
            self.features = list(df.columns)

        self.features_to_drop = df.columns[df.isna().all()].tolist()  # drop entirely NA columns which may arise after train/val split
        if self.features_to_drop:
            df = df.drop(columns=self.features_to_drop)

        if needs_extra_types is True:
            types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'embed': [], 'language': []}
            # continuous = numeric features to rescale
            # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
            # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
            features_to_consider = [feat for feat in self.features if feat not in self.features_to_drop]
            for feature in features_to_consider:
                feature_data = df[feature]  # pd.Series
                num_unique_vals = len(feature_data.unique())
                if num_unique_vals == 2:  # will be onehot encoded regardless of proc.embed_min_categories value
                    types_of_features['onehot'].append(feature)
                elif feature in continuous_featnames:
                    if np.abs(feature_data.skew()) > skew_threshold:
                        types_of_features['skewed'].append(feature)
                    else:
                        types_of_features['continuous'].append(feature)
                elif feature in categorical_featnames:
                    if num_unique_vals >= embed_min_categories:  # sufficiently many categories to warrant learned embedding dedicated to this feature
                        types_of_features['embed'].append(feature)
                    else:
                        types_of_features['onehot'].append(feature)
                elif feature in language_featnames:
                    types_of_features['language'].append(feature)
        else:
            types_of_features = []
            for feature in valid_features:
                if feature in categorical_featnames:
                    feature_type = 'CATEGORICAL'
                elif feature in continuous_featnames:
                    feature_type = 'SCALAR'
                elif feature in language_featnames:
                    feature_type = 'TEXT'
                else:
                    raise ValueError(f'Invalid feature: {feature}')

                types_of_features.append({"name": feature, "type": feature_type})

        return types_of_features, df


if __name__ == '__main__':
    model = AbstractModel(path='/tmp', name='test', problem_type='binary',
                          eval_metric=accuracy)
