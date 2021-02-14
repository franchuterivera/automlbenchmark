import logging
import math
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd

from abstract_model import AbstractModel, NotEnoughMemoryError, TimeLimitExceeded, accuracy

import math

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from rf_model import FeatureMetadata

logger = logging.getLogger(__name__)


# TODO: Add weight support?
# TODO: Can these be optimized? What computational cost do they have compared to the default catboost versions?
class CustomMetric:
    def __init__(self, metric, is_higher_better, needs_pred_proba):
        self.metric = metric
        self.is_higher_better = is_higher_better
        self.needs_pred_proba = needs_pred_proba

    @staticmethod
    def get_final_error(error, weight):
        return error

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        raise NotImplementedError


class BinaryCustomMetric(CustomMetric):
    @staticmethod
    def _get_y_pred_proba(approxes):
        return np.array(approxes[0])

    @staticmethod
    def _get_y_pred(y_pred_proba):
        return np.round(y_pred_proba)

    def evaluate(self, approxes, target, weight):
        y_pred_proba = self._get_y_pred_proba(approxes=approxes)

        # TODO: Binary log_loss doesn't work for some reason
        if self.needs_pred_proba:
            score = self.metric(np.array(target), y_pred_proba)
        else:
            raise NotImplementedError('Custom Catboost Binary prob metrics are not supported by AutoGluon.')
            # y_pred = self._get_y_pred(y_pred_proba=y_pred_proba)  # This doesn't work at the moment because catboost returns some strange valeus in approxes which are not the probabilities
            # score = self.metric(np.array(target), y_pred)

        return score, 1


class MulticlassCustomMetric(CustomMetric):
    @staticmethod
    def _get_y_pred_proba(approxes):
        return np.array(approxes)

    @staticmethod
    def _get_y_pred(y_pred_proba):
        return y_pred_proba.argmax(axis=0)

    def evaluate(self, approxes, target, weight):
        y_pred_proba = self._get_y_pred_proba(approxes=approxes)
        if self.needs_pred_proba:
            raise NotImplementedError('Custom Catboost Multiclass proba metrics are not supported by AutoGluon.')
            # y_pred_proba = y_pred_proba.reshape(len(np.unique(np.array(target))), -1).T
            # score = self.metric(np.array(target), y_pred_proba)  # This doesn't work at the moment because catboost returns some strange valeus in approxes which are not the probabilities
        else:
            y_pred = self._get_y_pred(y_pred_proba=y_pred_proba)
            score = self.metric(np.array(target), y_pred)

        return score, 1


class RegressionCustomMetric(CustomMetric):
    @staticmethod
    def _get_y_pred(approxes):
        return np.array(approxes[0])

    def evaluate(self, approxes, target, weight):
        y_pred = self._get_y_pred(approxes=approxes)
        score = self.metric(np.array(target), y_pred)

        return score, 1


metric_classes_dict = {
    'binary': BinaryCustomMetric,
    'multiclass': MulticlassCustomMetric,
}


# TODO: Refactor as a dictionary mapping as done in LGBM
def construct_custom_catboost_metric(metric, is_higher_better, needs_pred_proba, problem_type):
    if (metric.name == 'log_loss') and (problem_type == 'multiclass') and needs_pred_proba:
        return 'MultiClass'
    if metric.name == 'accuracy':
        return 'Accuracy'
    if (metric.name == 'log_loss') and (problem_type == 'binary') and needs_pred_proba:
        return 'Logloss'
    if (metric.name == 'roc_auc') and (problem_type == 'binary') and needs_pred_proba:
        return 'AUC'
    if (metric.name == 'f1') and (problem_type == 'binary') and not needs_pred_proba:
        return 'F1'
    if (metric.name == 'balanced_accuracy') and (problem_type == 'binary') and not needs_pred_proba:
        return 'BalancedAccuracy'
    if (metric.name == 'recall') and (problem_type == 'binary') and not needs_pred_proba:
        return 'Recall'
    if (metric.name == 'precision') and (problem_type == 'binary') and not needs_pred_proba:
        return 'Precision'
    metric_class = metric_classes_dict[problem_type]
    return metric_class(metric=metric, is_higher_better=is_higher_better, needs_pred_proba=needs_pred_proba)


DEFAULT_ITERATIONS = 10000


def get_param_baseline(problem_type, num_classes=None):
    if problem_type == 'binary':
        return get_param_binary_baseline()
    elif problem_type in ['multiclass']:
        return get_param_multiclass_baseline(num_classes=num_classes)
    else:
        return get_param_binary_baseline()


def get_param_binary_baseline():
    params = {
        'iterations': DEFAULT_ITERATIONS,
        'learning_rate': 0.1,
    }
    return params


def get_param_multiclass_baseline(num_classes):
    params = {
        'iterations': DEFAULT_ITERATIONS,
        'learning_rate': 0.1,
    }
    return params


def get_param_regression_baseline():
    params = {
        'iterations': DEFAULT_ITERATIONS,
        'learning_rate': 0.1,
    }
    return params





# TODO: Consider having CatBoost variant that converts all categoricals to numerical as done in RFModel, was showing improved results in some problems.
class CatBoostModel(AbstractModel):
    """
    CatBoost model: https://catboost.ai/
    Hyperparameter options: https://catboost.ai/docs/concepts/python-reference_parameters-list.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._category_features = None

    def _set_default_params(self):
        default_params = get_param_baseline(problem_type=self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
        self._set_default_param_value('random_seed', 0)  # Remove randomness for reproducibility
        # Set 'allow_writing_files' to True in order to keep log files created by catboost during training (these will be saved in the directory where AutoGluon stores this model)
        self._set_default_param_value('allow_writing_files', False)  # Disables creation of catboost logging files during training by default
        self._set_default_param_value('eval_metric', construct_custom_catboost_metric(self.stopping_metric, True, not self.stopping_metric.needs_pred, self.problem_type))

    def _get_default_searchspace(self):
        return get_default_searchspace(self.problem_type, num_classes=self.num_classes)

    def preprocess(self, X, preprocess_nonadaptive=True, preprocess_stateful=True, **kwargs):
        if preprocess_nonadaptive:
            X = self._preprocess_nonadaptive(X, **kwargs)
        if preprocess_stateful:
            X = self._preprocess(X, **kwargs)
        return X

    def _preprocess_nonadaptive(self, X, **kwargs):
        if self.features is None:
            self._preprocess_set_features(X=X)
        if list(X.columns) != self.features:
            X = X[self.features]
        if self._category_features is None:
            self._category_features = list(X.select_dtypes(include='category').columns)
        if self._category_features:
            X = X.copy()
            for category in self._category_features:
                current_categories = X[category].cat.categories
                if '__NaN__' in current_categories:
                    X[category] = X[category].fillna('__NaN__')
                else:
                    X[category] = X[category].cat.add_categories('__NaN__').fillna('__NaN__')
        return X

    # TODO: Use Pool in preprocess, optimize bagging to do Pool.split() to avoid re-computing pool for each fold! Requires stateful + y
    #  Pool is much more memory efficient, avoids copying data twice in memory
    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, num_gpus=0, **kwargs):
        params = self.params.copy()
        model_type = CatBoostClassifier
        if isinstance(params['eval_metric'], str):
            metric_name = params['eval_metric']
        else:
            metric_name = type(params['eval_metric']).__name__
        num_rows_train = len(X_train)
        num_cols_train = len(X_train.columns)
        if self.problem_type == 'multiclass':
            if self.num_classes is not None:
                num_classes = self.num_classes
            else:
                num_classes = 10  # Guess if not given, can do better by looking at y_train
        else:
            num_classes = 1

        # TODO: Add ignore_memory_limits param to disable NotEnoughMemoryError Exceptions
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        approx_mem_size_req = num_rows_train * num_cols_train * num_classes / 2  # TODO: Extremely crude approximation, can be vastly improved
        if approx_mem_size_req > 1e9:  # > 1 GB
            available_mem = 4469755084
            ratio = approx_mem_size_req / available_mem
            if ratio > (1 * max_memory_usage_ratio):
                logger.warning('\tWarning: Not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))
                raise NotEnoughMemoryError
            elif ratio > (0.2 * max_memory_usage_ratio):
                logger.warning('\tWarning: Potentially not enough memory to safely train CatBoost model, roughly requires: %s GB, but only %s GB is available...' % (round(approx_mem_size_req / 1e9, 3), round(available_mem / 1e9, 3)))

        start_time = time.time()
        X_train = self.preprocess(X_train)
        cat_features = list(X_train.select_dtypes(include='category').columns)
        X_train = Pool(data=X_train, label=y_train, cat_features=cat_features)

        if X_val is not None:
            X_val = self.preprocess(X_val)
            X_val = Pool(data=X_val, label=y_val, cat_features=cat_features)
            eval_set = X_val
            if num_rows_train <= 10000:
                modifier = 1
            else:
                modifier = 10000/num_rows_train
            early_stopping_rounds = max(round(modifier*150), 10)
            num_sample_iter_max = max(round(modifier*50), 2)
        else:
            eval_set = None
            early_stopping_rounds = None
            num_sample_iter_max = 50

        train_dir = None
        if 'allow_writing_files' in self.params and self.params['allow_writing_files']:
            if 'train_dir' not in self.params:
                try:
                    # TODO: What if path is in S3?
                    os.makedirs(os.path.dirname(self.path), exist_ok=True)
                except:
                    pass
                else:
                    train_dir = self.path + 'catboost_info'

        # TODO: Add more control over these params (specifically early_stopping_rounds)
        verbosity = kwargs.get('verbosity', 2)
        if verbosity <= 1:
            verbose = False
        elif verbosity == 2:
            verbose = False
        elif verbosity == 3:
            verbose = 20
        else:
            verbose = True

        init_model = None
        init_model_tree_count = None
        init_model_best_score = None

        num_features = len(self.features)

        if num_gpus != 0:
            if 'task_type' not in params:
                params['task_type'] = 'GPU'
                logger.log(20, f'\tTraining {self.name} with GPU, note that this may negatively impact model quality compared to CPU training.')
                # TODO: Confirm if GPU is used in HPO (Probably not)
                # TODO: Adjust max_bins to 254?

        if params.get('task_type', None) == 'GPU':
            if 'colsample_bylevel' in params:
                params.pop('colsample_bylevel')
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')
            if 'rsm' in params:
                params.pop('rsm')
                logger.log(30, f'\t\'rsm\' is not supported on GPU, using default value (Default = 1).')

        if self.problem_type == 'multiclass' and 'rsm' not in params and 'colsample_bylevel' not in params and num_features > 1000:
            if time_limit:
                # Reduce sample iterations to avoid taking unreasonable amounts of time
                num_sample_iter_max = max(round(num_sample_iter_max/2), 2)
            # Subsample columns to speed up training
            if params.get('task_type', None) != 'GPU':  # RSM does not work on GPU
                params['colsample_bylevel'] = max(min(1.0, 1000 / num_features), 0.05)
                logger.log(30, f'\tMany features detected ({num_features}), dynamically setting \'colsample_bylevel\' to {params["colsample_bylevel"]} to speed up training (Default = 1).')
                logger.log(30, f'\tTo disable this functionality, explicitly specify \'colsample_bylevel\' in the model hyperparameters.')
            else:
                params['colsample_bylevel'] = 1.0
                logger.log(30, f'\t\'colsample_bylevel\' is not supported on GPU, using default value (Default = 1).')

        logger.log(15, f'\tCatboost model hyperparameters: {params}')

        if train_dir is not None:
            params['train_dir'] = train_dir

        if time_limit:
            time_left_start = time_limit - (time.time() - start_time)
            if time_left_start <= time_limit * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded
            params_init = params.copy()
            num_sample_iter = min(num_sample_iter_max, params_init['iterations'])
            params_init['iterations'] = num_sample_iter
            self.model = model_type(
                **params_init,
            )
            self.model.fit(
                X_train,
                eval_set=eval_set,
                use_best_model=True,
                verbose=verbose,
                # early_stopping_rounds=early_stopping_rounds,
            )

            init_model_tree_count = self.model.tree_count_
            init_model_best_score = self.model.get_best_score()['validation'][metric_name]

            time_left_end = time_limit - (time.time() - start_time)
            time_taken_per_iter = (time_left_start - time_left_end) / num_sample_iter
            estimated_iters_in_time = round(time_left_end / time_taken_per_iter)
            init_model = self.model

            if self.stopping_metric._optimum == init_model_best_score:
                # Done, pick init_model
                params_final = None
            else:
                params_final = params.copy()

                # TODO: This only handles memory with time_limit specified, but not with time_limit=None, handle when time_limit=None
                available_mem = 4469755084
                model_size_bytes = sys.getsizeof(pickle.dumps(self.model))

                max_memory_proportion = 0.3 * max_memory_usage_ratio
                mem_usage_per_iter = model_size_bytes / num_sample_iter
                max_memory_iters = math.floor(available_mem * max_memory_proportion / mem_usage_per_iter)
                if params.get('task_type', None) == 'GPU':
                    # Cant use init_model
                    iterations_left = params['iterations']
                else:
                    iterations_left = params['iterations'] - num_sample_iter
                params_final['iterations'] = min(iterations_left, estimated_iters_in_time)
                if params_final['iterations'] > max_memory_iters - num_sample_iter:
                    if max_memory_iters - num_sample_iter <= 500:
                        logger.warning('\tWarning: CatBoost will be early stopped due to lack of memory, increase memory to enable full quality models, max training iterations changed to %s from %s' % (max_memory_iters, params_final['iterations'] + num_sample_iter))
                    params_final['iterations'] = max_memory_iters - num_sample_iter
        else:
            params_final = params.copy()

        if params_final is not None and params_final['iterations'] > 0:
            self.model = model_type(
                **params_final,
            )

            fit_final_kwargs = dict(
                eval_set=eval_set,
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds,
            )

            # TODO: Strangely, this performs different if clone init_model is sent in than if trained for same total number of iterations. May be able to optimize catboost models further with this
            warm_start = False
            if params_final.get('task_type', None) == 'GPU':
                # Cant use init_model
                fit_final_kwargs['use_best_model'] = True
            elif init_model is not None:
                fit_final_kwargs['init_model'] = init_model
                warm_start = True
            self.model.fit(X_train, **fit_final_kwargs)

            if init_model is not None:
                final_model_best_score = self.model.get_best_score()['validation'][metric_name]

                if self.stopping_metric._optimum == init_model_best_score:
                    # Done, pick init_model
                    self.model = init_model
                else:
                    if (init_model_best_score > self.stopping_metric._optimum) or (final_model_best_score > self.stopping_metric._optimum):
                        init_model_best_score = -init_model_best_score
                        final_model_best_score = -final_model_best_score

                    if warm_start:
                        if init_model_best_score >= final_model_best_score:
                            self.model = init_model
                        else:
                            best_iteration = init_model_tree_count + self.model.get_best_iteration()
                            self.model.shrink(ntree_start=0, ntree_end=best_iteration + 1)
                    else:
                        if init_model_best_score >= final_model_best_score:
                            self.model = init_model

        self.params_trained['iterations'] = self.model.tree_count_

    def _predict_proba(self, X, **kwargs):
        return super()._predict_proba(X, **kwargs)

    def get_model_feature_importance(self):
        importance_df = self.model.get_feature_importance(prettified=True)
        importance_df['Importances'] = importance_df['Importances'] / 100
        importance_series = importance_df.set_index('Feature Id')['Importances']
        importance_dict = importance_series.to_dict()
        return importance_dict

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=['object'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def _preprocess_set_features(self, X: pd.DataFrame):
        self.features = list(X.columns)
        # TODO: Consider changing how this works or where it is done
        if self.feature_metadata is None:
            feature_metadata = FeatureMetadata.from_df(X)
        else:
            feature_metadata = self.feature_metadata
        get_features_kwargs = self.params_aux.get('get_features_kwargs', None)
        if get_features_kwargs is not None:
            valid_features = feature_metadata.get_features(**get_features_kwargs)
        else:
            ignored_type_group_raw = self.params_aux.get('ignored_type_group_raw', None)
            ignored_type_group_special = self.params_aux.get('ignored_type_group_special', None)
            valid_features = feature_metadata.get_features(invalid_raw_types=ignored_type_group_raw, invalid_special_types=ignored_type_group_special)
        get_features_kwargs_extra = self.params_aux.get('get_features_kwargs_extra', None)
        if get_features_kwargs_extra is not None:
            valid_features_extra = feature_metadata.get_features(**get_features_kwargs_extra)
            valid_features = [feature for feature in valid_features if feature in valid_features_extra]
        dropped_features = [feature for feature in self.features if feature not in valid_features]
        logger.log(10, f'\tDropped {len(dropped_features)} of {len(self.features)} features.')
        self.features = [feature for feature in self.features if feature in valid_features]
        self.feature_metadata = feature_metadata.keep_features(self.features)
        if not self.features:
            raise NoValidFeatures



if __name__ == '__main__':
    import sklearn.datasets
    import sklearn.model_selection
    from pandas.api.types import is_numeric_dtype
    from sklearn.utils.multiclass import type_of_target
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
        model = CatBoostModel(path='/tmp/', problem_type=problem_type, metric=accuracy, name='CatboostModel', eval_metric=accuracy)
        model.fit(X_train=X_train, y_train=y_train)
        print(model.predict_proba(X_test).shape)
        print(accuracy(y_test.to_numpy().astype(int), model.predict(X_test)))
