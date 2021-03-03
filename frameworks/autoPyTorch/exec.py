import json
from multiprocessing import Process
import random
import torch
import logging
import typing
import math
import os
import tempfile as tmp
import warnings
import time
import numpy as np
import sklearn.utils
from sklearn.utils.multiclass import type_of_target

#os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
#os.environ['OMP_NUM_THREADS'] = '1'
#os.environ['OPENBLAS_NUM_THREADS'] = '1'
#os.environ['MKL_NUM_THREADS'] = '1'

from sklearn.metrics import accuracy_score
import ConfigSpace as cs
from autoPyTorch import AutoNetClassification, AutoNetRegression, AutoNetEnsemble
from autoPyTorch import HyperparameterSearchSpaceUpdates
from autoPyTorch.pipeline.nodes import LogFunctionsSelector, BaselineTrainer
from autoPyTorch.components.metrics.additional_logs import (
    test_result_ens,
    test_result,
    gradient_norm,
    gradient_mean,
    gradient_std,
)
from autoPyTorch.utils.ensemble import test_predictions_for_ensemble

from frameworks.shared.callee import call_run, result, utils, output_subdir

from collections import Counter
import random

import numpy as np

from autoPyTorch.components.ensembles.abstract_ensemble import AbstractEnsemble
import random
from collections import Counter
from typing import Any, Dict, List, Tuple, Union, cast


from autoPyTorch.components.metrics import balanced_accuracy, accuracy, auc_metric, mae, rmse, multilabel_accuracy, cross_entropy
from autoPyTorch.pipeline.nodes.metric_selector import AutoNetMetric, undo_ohe, default_minimize_transform
#from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection
from hpbandster.core.result import logged_results_to_HBS_result


log = logging.getLogger(__name__)


class EnsembleSelection(AbstractEnsemble):
    """Ensemble Selection algorithm extracted from auto-sklearn"""

    def __init__(self, ensemble_size, metric,
                 sorted_initialization_n_best=0, only_consider_n_best=0,
                 random_state=None,
                 bagging=False, mode='fast'):
        self.ensemble_size = ensemble_size
        self.metric = accuracy_score
        self.sorted_initialization_n_best = sorted_initialization_n_best
        self.only_consider_n_best = only_consider_n_best
        self.bagging = bagging
        self.mode = mode

    def fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')

        if self.mode not in ('fast', 'slow'):
            raise ValueError('Unknown mode %s' % self.mode)

        self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> AbstractEnsemble:
        self._fast(predictions, labels)
        return self

    def _fast(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        for i in range(ensemble_size):
            scores = np.zeros(
                (len(predictions)),
                dtype=np.float64,
            )
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble
                # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                # We overwrite the contents of fant_ensemble_prediction
                # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
                np.add(
                    weighted_ensemble_prediction,
                    pred,
                    out=fant_ensemble_prediction
                )
                np.multiply(
                    fant_ensemble_prediction,
                    (1. / float(s + 1)),
                    out=fant_ensemble_prediction
                )

                # Calculate score is versatile and can return a dict of score
                # when scoring_functions=None, we know it will be a float
                calculated_score = self.metric(labels, fant_ensemble_prediction.argmax(1))
                scores[j] = 1 - calculated_score

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = np.random.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]
        print(f"Fitted ensemble with trajectory={self.trajectory_} self.indices_ = {self.indices_}")

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average

    def __str__(self) -> str:
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def no_transform(value):
    return value


class EnsembleTrajectorySimulator():

    def __init__(self, ensemble_pred_dir, ensemble_config, seed, perf_metric):

        self.perf_metric = perf_metric
        self.ensemble_pred_dir = os.path.join(ensemble_pred_dir,
                                              "predictions_for_ensemble.npy")
        self.ensemble_pred_dir_test = os.path.join(ensemble_pred_dir,
                                                   "test_predictions_for_ensemble.npy")
        self.ensemble_config = ensemble_config
        self.seed = seed

        self.read_runfiles()

        self.ensemble_selection = EnsembleSelection(**ensemble_config)

        # Probability of each model is too low
        # from sklearn.preprocessing import OneHotEncoder

    def read_runfiles(self, val_split=0.5):

        self.ensemble_identifiers = []
        self.ensemble_predictions = []
        self.ensemble_predictions_ensemble_val = []
        self.ensemble_timestamps = []

        print(f"self.ensemble_pred_dir={self.ensemble_pred_dir}")
        with open(self.ensemble_pred_dir, "rb") as f:
            self.labels = np.load(f, allow_pickle=True)

            while True:
                try:
                    job_id, budget, timestamps = np.load(f, allow_pickle=True)
                    predictions = np.array(np.load(f, allow_pickle=True))

                    shape = np.shape(predictions)
                    if len(shape) == 2 and type_of_target(self.labels) == 'binary' and self.perf_metric == 'auc_metric':
                        # Looks like autopytorch does not support argmaxing scores
                        # hardcode here as when this case happens we have a multiclass
                        print(f"before predictions shape == {predictions.shape}")
                        predictions = np.argmax(predictions, axis=1)
                        # This is for a bug in autopytroch that does force one hot encoding when shapes are not same
                        if len(self.labels.shape) != len(predictions.shape):
                            if len(self.labels.shape) == 2 and len(predictions.shape) == 1:
                                predictions = predictions.reshape(-1, 1)
                            elif len(self.labels.shape) == 1 and len(predictions.shape) == 2:
                                predictions = predictions.reshape(-1)
                    print(f"after predictions shape == {predictions.shape}")
                    print(f"after predictions {job_id} {budget}  == {predictions}")

                    self.ensemble_identifiers.append(job_id + (budget, ))
                    self.ensemble_predictions.append(predictions)
                    self.ensemble_timestamps.append(timestamps)
                except (EOFError, OSError):
                    break

        self.ensemble_predictions_test = []
        self.test_labels = None

        if os.path.exists(self.ensemble_pred_dir_test):
            with open(self.ensemble_pred_dir_test, "rb") as f:
                try:
                    self.test_labels = np.load(f, allow_pickle=True)
                except (EOFError, OSError):
                    pass

                while True:
                    try:
                        job_id, budget, timestamps = np.load(f, allow_pickle=True)
                        predictions = np.array(np.load(f, allow_pickle=True))

                        # Somehow the prediction shape is (1, batch, num_class)
                        # reshape to match (batch, num_classes)
                        if len(predictions.shape) == 3 and predictions.shape[0] == 1:
                            predictions = predictions.reshape(predictions.shape[1], predictions.shape[2])

                        self.ensemble_predictions_test.append(predictions)
                        print("==> Adding test labels with shape", predictions.shape)
                    except (EOFError, OSError):
                        break

        print("==> Found %i val preds" % len(self.ensemble_predictions))
        print("==> Found %i test preds" % len(self.ensemble_predictions_test))
        print("==> Found %i timestamps" % len(self.ensemble_timestamps))

    def get_ensemble_test_prediction(self, test_labels):

        for idx, oof_prediction, test_prediction in zip(self.ensemble_identifiers, self.ensemble_predictions, self.ensemble_predictions_test):
            if len(oof_prediction.shape) == 1:
                oof_prediction = oof_prediction.reshape(-1, 1)
            if len(test_prediction.shape) == 1:
                test_prediction = test_prediction.reshape(-1, 1)

            try:
                score = accuracy_score(self.labels, oof_prediction.argmax(1))
            except Exception as e:
                score = None
                print(f"Failed during score calculation for {idx} with {e} for solution={self.labels.shape} and oof_prediction={oof_prediction.shape}")
            try:
                score_test = accuracy_score(test_labels, test_prediction.argmax(1))
            except Exception as e:
                score_test = None
                print(f"Failed during score_test calculation for {idx} with {e} for solution={test_labels.shape} and oof_prediction={test_prediction.shape}")
            try:
                score_test_withownlabels = accuracy_score(self.test_labels, test_prediction.argmax(1))
            except Exception as e:
                score_test_withownlabels = None
                print(f"Failed during score calculation for {idx} with {e} for solution={self.test_labels.shape} and oof_prediction={test_prediction.shape}")
            print(f"{idx}-{oof_prediction.shape}-{test_prediction.shape}-Accuracy: score_val={score} score_test={score_test} score_test_withownlabels={score_test_withownlabels}")

        # create ensemble
        tag = time.time()
        with open(f"/tmp/test_{tag}.npy", 'wb') as f:
            np.save(f, np.array(self.ensemble_predictions_test))
        with open(f"/tmp/train_{tag}.npy", 'wb') as f:
            np.save(f, np.array(self.ensemble_predictions))
        with open(f"/tmp/labels_{tag}.npy", 'wb') as f:
            np.save(f, np.array(self.labels))
        self.ensemble_selection.fit(
            self.ensemble_predictions,
            self.labels, self.ensemble_identifiers)

        # get test performance
        self.test_preds = self.ensemble_selection.predict(
            self.ensemble_predictions_test)

        if len(self.test_preds.shape) == 3:
            self.test_preds = self.test_preds[0]
        if len(self.test_preds.shape) == 2:
            self.test_preds = np.argmax(self.test_preds, axis=1)
        test_performance = accuracy(self.test_labels, self.test_preds)
        print(f"===> Test performance of ensemble is {test_performance} sklearn_performance={accuracy_score(self.test_labels, self.test_preds)}")

        return self.test_preds


def get_bohb_rundirs(rundir):

    rundirs = []

    dataset_dirs = [os.path.join(rundir, p) for p in os.listdir(rundir) if not p.endswith("cluster")]

    for ds_path in dataset_dirs:
        rundirs = rundirs + [os.path.join(ds_path, rundir) for rundir in os.listdir(ds_path)]

    return rundirs


def minimize_trf(value):
        return -1*value


def get_ensemble_config():
    ensemble_config = {
            "ensemble_size":50,
            "ensemble_only_consider_n_best":20,
            "ensemble_sorted_initialization_n_best":0
            }
    return ensemble_config


def get_simulator_ensemble_config(metric_name=None):

    # acc='accuracy',
    autonet_accuracy = AutoNetMetric(name="accuracy", metric=accuracy,
                                     loss_transform=minimize_trf, ohe_transform=undo_ohe)
    # bac='balanced_accuracy',
    autonet_balanced_accuracy = AutoNetMetric(name="balanced_accuracy", metric=balanced_accuracy,
                                              loss_transform=minimize_trf, ohe_transform=undo_ohe)
    # f1='auc_metric',
    # auc='auc_metric',
    autonet_auc = AutoNetMetric(name="auc", metric=auc_metric,
                                loss_transform=minimize_trf, ohe_transform=no_transform)
    # logloss='cross_entropy',
    autonet_cross_entropy = AutoNetMetric(name="cross_entropy", metric=cross_entropy,
                                          loss_transform=no_transform, ohe_transform=no_transform)
    # mae='mean_distance',
    # mse='mean_distance',
    autonet_mean_distance = AutoNetMetric(name="mean_distance", metric=mae,
                                          loss_transform=no_transform, ohe_transform=no_transform)
    # r2='auc_metric'

    METRIC_DICT = {
        "accuracy": autonet_accuracy,
        "balanced_accuracy": autonet_balanced_accuracy,
        "cross_entropy": autonet_cross_entropy,
        'mean_distance': autonet_mean_distance,
        "auc_metric": autonet_auc
    }

    ensemble_config = {"ensemble_size": 35,
                       "only_consider_n_best": 10,
                       "sorted_initialization_n_best": 1,
                       'random_state': np.random.RandomState(0),
                       'metric': METRIC_DICT[metric_name]
                       }
    return ensemble_config


def get_hyperparameter_search_space_updates_lcbench():
    search_space_updates = HyperparameterSearchSpaceUpdates()
    search_space_updates.append(node_name="InitializationSelector",
                                hyperparameter="initializer:initialize_bias",
                                value_range=["Yes"])
    search_space_updates.append(node_name="CreateDataLoader",
                                hyperparameter="batch_size",
                                value_range=[16, 512],
                                log=True)
    search_space_updates.append(node_name="LearningrateSchedulerSelector",
                                hyperparameter="cosine_annealing:T_max",
                                value_range=[50, 50])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:activation",
                                value_range=["relu"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:max_units",
                                value_range=[64, 1024],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:max_units",
                                value_range=[32, 512],
                                log=True)
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:num_groups",
                                value_range=[1, 5])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:blocks_per_group",
                                value_range=[1, 3])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:resnet_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedresnet:activation",
                                value_range=["relu"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:mlp_shape",
                                value_range=["funnel"])
    search_space_updates.append(node_name="NetworkSelector",
                                hyperparameter="shapedmlpnet:num_layers",
                                value_range=[1, 6])
    return search_space_updates


def get_autonet_config_lcbench(min_budget, max_budget, max_runtime, run_id, task_id,
                               num_workers, logdir, seed, memory_limit_mb, is_classification):
    autonet_config = {
            'additional_logs': [],
            'additional_metrics': ["balanced_accuracy"],
            'algorithm': 'bohb',
            'batch_loss_computation_techniques': ['standard', 'mixup'],
            'best_over_epochs': False,
            'budget_type': 'epochs',
            'categorical_features': None,
            #'cross_validator': 'stratified_k_fold',
            #'cross_validator_args': dict({"n_splits":5}),
            'cross_validator': 'none',
            'cuda': False,
            'dataset_name': None,
            'early_stopping_patience': 10,
            'early_stopping_reset_parameters': False,
            'embeddings': ['none', 'learned'],
            'eta': 2,
            'final_activation': 'softmax' if is_classification else 'none',
            'full_eval_each_epoch': True,
            'hyperparameter_search_space_updates': get_hyperparameter_search_space_updates_lcbench(),
            'imputation_strategies': ['mean'],
            'initialization_methods': ['default'],
            'initializer': 'simple_initializer',
            'log_level': 'info',
            'loss_modules': ['cross_entropy_weighted'],
            'lr_scheduler': ['cosine_annealing'],
            'max_budget': max_budget,
            'max_runtime': max_runtime,
            'memory_limit_mb': memory_limit_mb,
            'min_budget': min_budget,
            'min_budget_for_cv': 0,
            'min_workers': num_workers,
            #'network_interface_name': 'eth0',
            'network_interface_name': 'lo',
            'networks': ['shapedmlpnet', 'shapedresnet'],
            'normalization_strategies': ['standardize'],
            'num_iterations': 300,
            'optimize_metric': 'accuracy',
            'optimizer': ['sgd', 'adam'],
            'over_sampling_methods': ['none'],
            'preprocessors': ['none', 'truncated_svd'],
            'random_seed': seed,
            'refit_validation_split': 0.2,
            'result_logger_dir': logdir,
            'run_id': run_id,
            'run_worker_on_master_node': True,
            'shuffle': True,
            'target_size_strategies': ['none'],
            'task_id': task_id,
            'torch_num_threads': 2,
            'under_sampling_methods': ['none'],
            'use_pynisher': True,
            'use_tensorboard_logger': False,
            'validation_split': 0.2,
            'working_dir': tmp.gettempdir(),
            }
    return autonet_config


def get_autonet_instance_for_id(
    run_id: str,
    task_id: int,
    logdir: str,
    config: typing.Any,  # Automlbenchmark Namespace Object
    cat_feats: typing.List[str],
    ensemble_setting: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perf_metric: str,
) -> typing.Union[AutoNetClassification, AutoNetRegression, AutoNetEnsemble]:
    """
    Given an automl task setting, this function returns an autonet object that is able to fit a X_train/Y_train pair.
    Notice that for multiprocessing jobs this function is likely gonna be called with the same run_id among autopytorch objects,
    and task_id from 1 to n_jobs, where 1 is going to be the master
    Args:
        run_id (str): A common identifier among all autonet objects
        task_id (int): The id of the multiprocessing object, where 0 is master
        logdir (str): where the results are gonna be populated
        config (typing.Any): A configuration that dictates information about the task
        cat_feats (typing.List[str]): The list of categorical features
        ensemble_setting (str): ensemble/normal
        X_test (np.ndarray): testing features
        y_test (np.ndarray): testing labels
        perf_metric (str): The performance metric to optimizer
    Returns:
        Autonet object that is able to fit a model
    """
    # Extract runtime settings from the configuration
    # Be at least 10 seconds pessimistic, because automlbenchmark will kill the run
    # and killing bohb causes hang!
    max_search_runtime = config.max_runtime_seconds - 5*60 if config.max_runtime_seconds > 6*60 else config.max_runtime_seconds - 10
    max_search_runtime = max_search_runtime - int(0.1*max_search_runtime)
    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    ml_memory_limit = config.framework_params.get('_ml_memory_limit', 'auto')

    # Use for now same auto memory setting from autosklearn
    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = utils.system_memory_mb().total
    if ml_memory_limit == 'auto':
        ml_memory_limit = max(min(config.max_mem_size_mb,
                                  math.ceil(total_memory_mb / n_jobs)),
                              3072)  # 3072 is autosklearn defaults

    autonet_config = get_autonet_config_lcbench(
            min_budget=10,
            max_budget=50,
            max_runtime=max_search_runtime,
            run_id=run_id,
            task_id=task_id,
            num_workers=n_jobs,
            logdir=logdir,
            seed=config.seed,
            memory_limit_mb=ml_memory_limit,
            is_classification=config.type == 'classification',
    )

    autonet = AutoNetRegression
    if config.type == 'classification':
        autonet_config["algorithm"] = "portfolio_bohb"
        autonet_config["portfolio_type"] = "greedy"
        autonet = AutoNetClassification

    # Categoricals
    if any(cat_feats):
        autonet_config["categorical_features"] = cat_feats
    autonet_config["embeddings"] = ['none', 'learned']

    # Test logging
    autonet_config["additional_logs"] = [
        test_predictions_for_ensemble.__name__,
        test_result_ens.__name__
    ]

    # Set up ensemble
    ensemble_config = get_ensemble_config()
    autonet_config = {**autonet_config, **ensemble_config}
    autonet_config["optimize_metric"] = perf_metric

    if ensemble_setting == "ensemble":
        auto_pytorch = AutoNetEnsemble(autonet, config_preset="full_cs", **autonet_config)
    else:
        auto_pytorch = autonet(config_preset="full_cs", **autonet_config)

    # Test logging cont.
    auto_pytorch.pipeline[LogFunctionsSelector.get_name()].add_log_function(
        name=test_predictions_for_ensemble.__name__,
        log_function=test_predictions_for_ensemble(auto_pytorch, X_test, y_test),
        loss_transform=False
    )
    auto_pytorch.pipeline[LogFunctionsSelector.get_name()].add_log_function(
        name=test_result_ens.__name__,
        log_function=test_result_ens(auto_pytorch, X_test, y_test)
    )

    auto_pytorch.pipeline[BaselineTrainer.get_name()].add_test_data(X_test)

    return auto_pytorch


def delayed_auto_pytorch_fit(
    auto_pytorch: typing.Union[AutoNetClassification, AutoNetRegression, AutoNetEnsemble],
    arguments: typing.Dict[str, typing.Any]
) -> None:
    """
    We need the master to be ready when we setup this function. This handy
    function just fits a model AFTER waiting for the master to be ready
    Args:
        auto_pytorch ()
        arguments ()
    """

    time.sleep(10)
    print(f"started autopytorch child with {arguments} PID={os.getpid()}")
    try:
        auto_pytorch.fit(**arguments)
    except Exception as e:
        # Print as in multiprocessing, the stdout is dis-associated
        print(f"Failed with {e}")
    return


def launch_complementary_autopytorch_jobs(
    run_id: str,
    logdir: str,
    config: typing.Any,  # Automlbenchmark Namespace Object
    cat_feats: typing.List[str],
    ensemble_setting: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perf_metric: str,
) -> typing.List[Process]:
    """
    HPBanster requires that we launch the workers manually, so this function creates n_job-1 threads,
    accounting for the fact that the master process will also be a job.
    Args:
        run_id (str): A common identifier among all autonet objects
        task_id (int): The id of the multiprocessing object, where 0 is master
        logdir (str): where the results are gonna be populated
        config (typing.Any): A configuration that dictates information about the task
        cat_feats (typing.List[str]): The list of categorical features
        ensemble_setting (str): ensemble/normal
        X_test (np.ndarray): testing features
        y_test (np.ndarray): testing labels
        perf_metric (str): The performance metric to optimizer
    Returns:
        List of process launched for final join()
    """
    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    launched = []
    for task_id in range(2, n_jobs+1):
        auto_pytorch_child = get_autonet_instance_for_id(
            run_id=run_id,
            task_id=task_id,
            logdir=logdir,
            config=config,
            cat_feats=cat_feats,
            ensemble_setting=ensemble_setting,
            X_test=X_test,
            y_test=y_test,
            perf_metric=perf_metric
        )
        arguments = {
            'X_train': X_train,
            'Y_train': y_train,
        }
        arguments.update(auto_pytorch_child.get_current_autonet_config())
        arguments['refit'] = False
        p = Process(target=delayed_auto_pytorch_fit, args=(auto_pytorch_child, arguments))
        p.start()
        launched.append(p)

    return launched


def run(dataset, config):
    """
    This Method builds a Pytorch network that best fits a given dataset
    setting
    Args:
            dataset: object containing the data to be fitted as well as the expected values
            config: Configuration of additional details, like whether is a regression or
                    classification task
    Returns:
            A dict with the number of elements that conform the PyTorch network that best
            fits the data (models_count) and the duration of the task (training_duration)
    """

    log.info("\n**** AutoPyTorch ****\n")
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    seed_everything(config.seed)

    # Mapping of benchmark metrics to AutoPyTorch metrics
    # TODO Some metrics are not yet implemented in framework
    metrics_mapping = dict(
        acc='accuracy',
        bac='balanced_accuracy',
        auc='auc_metric',
        f1='auc_metric',
        logloss='cross_entropy',
        mae='rmse',
        mse='rmse',
        r2='auc_metric'
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        log.exception("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("AutoPyTorch with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric)
    log.info("Environment: %s", os.environ)

    # Data Processing
    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    X_test = dataset.test.X_enc
    y_test = dataset.test.y_enc
    cat_feats = [type(f) == str for f in X_train[0]]

    # Find the model that best fit the data
    # TODO add seed support when implemented on the framework
    is_classification = config.type == 'classification'

    run_id = int(time.time())
    ensemble_setting = "ensemble"
    logdir = os.path.join(
        tmp.gettempdir(),
        "logs/run_" + str(run_id),
    )

    # Master Job
    auto_pytorch = get_autonet_instance_for_id(
        run_id=run_id,
        task_id=1,
        logdir=logdir,
        config=config,
        cat_feats=cat_feats,
        ensemble_setting=ensemble_setting,
        X_test=X_test,
        y_test=y_test,
        perf_metric=perf_metric
    )

    # Child jobs if any
    complementary_autopytorch_jobs = launch_complementary_autopytorch_jobs(
        run_id=run_id,
        logdir=logdir,
        config=config,
        cat_feats=cat_feats,
        ensemble_setting=ensemble_setting,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        perf_metric=perf_metric
    )

    with utils.Timer() as training:
        auto_pytorch.fit(
            X_train, y_train, **auto_pytorch.get_current_autonet_config(), refit=False)

        # Build ensemble
        log.info("Building ensemble and predicting on the test set.")
        ensemble_config = get_simulator_ensemble_config(perf_metric)

        simulator = EnsembleTrajectorySimulator(
            ensemble_pred_dir=logdir, ensemble_config=ensemble_config, seed=config.seed,
            perf_metric=perf_metric,
        )

    with utils.Timer() as predict:
        print(f"Predicting on X_test={X_test.shape} and y_test={y_test.shape}")
        predictions = np.array(simulator.get_ensemble_test_prediction(y_test), dtype='float')

    predictions = sklearn.utils.check_array(
            predictions,
            force_all_finite=True,
            accept_sparse='csr',
            ensure_2d=False,
    )
    print(f"Predicted= {predictions} {type(predictions)} {np.shape(predictions)} {predictions.dtype}")

    # Convert output to strings for classification
    if is_classification:
        probabilities = "predictions"  # encoding is handled by caller in `__init__.py`
    else:
        probabilities = None

    save_artifacts(auto_pytorch, config)
    [p.join() for p in complementary_autopytorch_jobs]
    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=None,
                  # Todo -- predict is performed internally, so
                  # below duration is not yet meaningful
                  predict_duration=predict.duration,
                  training_duration=training.duration)


def save_artifacts(autonet, config):
    try:
        models_repr = autonet.get_pytorch_model()
        log.debug("Trained Model:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
    except Exception as e:
        log.debug("Error when saving artifacts = {}.".format(e), exc_info=True)


if __name__ == '__main__':
    call_run(run)
