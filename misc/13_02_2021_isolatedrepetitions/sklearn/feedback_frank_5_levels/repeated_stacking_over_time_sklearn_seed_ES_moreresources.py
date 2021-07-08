from collections import Counter
import time
from random import randrange
import traceback
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import openml
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import argparse
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
import sklearn.datasets
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.neural_network import MLPClassifier
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
import random
from typing import Any, Dict, List, Optional, Tuple, Union, cast


# Mimic ensemble selection from AutoSklearn
class EnsembleSelection:
    def __init__(
        self,
        ensemble_size=20,
        metric=balanced_accuracy_score,
        random_state=np.random.RandomState(42),
    ) -> None:
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.random_state = random_state

    def fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        identifiers: List[Tuple[int, int, float]],
    ):
        self.ensemble_size = int(self.ensemble_size)
        self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
    ):
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
            losses = np.ones(
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

                # calculate_loss is versatile and can return a dict of losses
                # when scoring_functions=None, we know it will be a float
                #losses[j] = cast(
                #    float,
                #    calculate_loss(
                #        solution=labels,
                #        prediction=fant_ensemble_prediction,
                #        task_type=self.task_type,
                #        metric=self.metric,
                #        scoring_functions=None
                #    )
                #)
                losses[j] = 1.0 - self.metric(labels, fant_ensemble_prediction.argmax(1))

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]

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

    def get_models_with_weights(
        self,
        models,
    ):
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]


def fit_and_return_ensemble_predictions(
    X_train, y_train, X_test, y_test, model_list
):
    # First perform non repeated ensemble selection
    model = EnsembleSelection()
    model.fit(
        predictions=[np.mean(a, axis=0) for a in X_train],
        labels=y_train,
        identifiers=model_list,
    )
    val_score_repeatedFALSE = 1.0 - model.get_validation_performance()
    test_score_repeatedFALSE = balanced_accuracy_score(y_test,
                                                       model.predict([np.mean(a, axis=0) for a in X_test]).argmax(1))

    val_score_repeatedTRUE = None
    test_score_repeatedTRUE = None

    print(f"From X_train={[(a, np.shape(b)) for a, b in zip(model_list, X_train)]} val_score_repeatedTRUE={val_score_repeatedTRUE} val_score_repeatedFALSE={val_score_repeatedFALSE} test_score_repeatedTRUE={test_score_repeatedTRUE} test_score_repeatedFALSE={test_score_repeatedFALSE}")

    return val_score_repeatedTRUE, val_score_repeatedFALSE, test_score_repeatedTRUE, test_score_repeatedFALSE


def fit_and_return_avg_predictions(
    args, X_train, y_train, X_test, y_test, model_name, repeat
):
    # Predicitions
    indices = [None] * args.n_splits * repeat
    oof_predictions = [None] * args.n_splits * repeat
    test_predictions = [None] * args.n_splits * repeat
    model = [None] * args.n_splits * repeat

    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits,
                                   n_repeats=repeat, random_state=args.seed)

    # First train everything, then worry about format at end
    for i, (train_index, val_index) in enumerate(rskf.split(X_train.copy(), y_train.copy())):
        print(f"==> {i} Fitting model {model_name}")

        model[i] = model_func[model_name](**HPO[model_name]).fit(
            X_train[train_index],
            y_train[train_index],
        )

        # Predict to eval the model
        indices[i] = val_index
        oof_predictions[i] = model[i].predict_proba(X_train[val_index])
        test_predictions[i] = model[i].predict_proba(X_test)

    indices = np.concatenate([indices[i] for i in range(args.n_splits*repeat)
                              if indices[i] is not None])

    oof_predictions = np.concatenate([oof_predictions[i] for i in range(args.n_splits*repeat)
                                      if oof_predictions[i] is not None])
    indices = np.split(indices, repeat)
    oof_predictions = np.split(oof_predictions, repeat)
    for i in range(repeat):
        oof_predictions[i] = oof_predictions[i][np.argsort(indices[i])]

    # Convert to an average array
    this_test_history = []
    for i in range(len(test_predictions)):
        this_test_predictions = [test_predictions[a] for a in list(range(i + 1))]
        this_test_history.append(balanced_accuracy_score(y_test, np.mean(this_test_predictions, axis=0).argmax(1)))

    # calculate oof score
    val_score = balanced_accuracy_score(y_train, np.mean([a for a in oof_predictions], axis=0).argmax(1))

    # Convert to an average array
    test_predictions = [np.mean(test_predictions[args.n_splits*i:args.n_splits*(i+1)], axis=0) for i in range(0, repeat)]
    print(f"From X_train={X_train.shape} val_pred={np.mean(oof_predictions, axis=0).shape} and test_pred={np.mean(test_predictions, axis=0).shape} with val_score={val_score} test_score={this_test_history[-1]}")

    final_model = VotingClassifier(estimators=None, voting='soft', )
    final_model.estimators_ = model

    return oof_predictions, test_predictions, final_model, val_score, this_test_history


def save_frame(args, repeated_frame, history_frame):
    # Integrate that through progression
    path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_sklearn_feedback_frank_{args.openml_id}_{args.seed}.csv")
    pd.DataFrame(repeated_frame).to_csv(path)
    print(f"{time.ctime()}: saved {path}")
    path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_sklearn_feedback_frank_history_{args.openml_id}_{args.seed}.csv")
    pd.DataFrame(history_frame).to_csv(path)
    print(f"{time.ctime()}: saved {path}")


parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
parser.add_argument(
    '--openml_id',
    required=True,
    type=int,
)
parser.add_argument(
    '--seed',
    required=False,
    default=None,
    type=int,
)
parser.add_argument(
    '--n_splits',
    required=False,
    default=5,
    type=int,
)
parser.add_argument(
    '--n_repeats',
    required=False,
    default=20,
    type=int,
)
parser.add_argument(
    '--enable_HPO',
    required=False,
    default=False,
    type=bool,
)
args = parser.parse_args()

if args.seed is None:
     args.seed = randrange(100)

task = openml.tasks.get_task(args.openml_id)
X, y = sklearn.datasets.fetch_openml(data_id=task.dataset_id, return_X_y=True, as_frame=False)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp_mean.fit_transform(X)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=args.seed,
)
print(f"Loaded {args.openml_id} with {X_train.shape} train datapoints and {X_test.shape} test datapoints")

HPO = {
    'HistGradientBoostingClassifier': {
        'loss': 'auto',
        'learning_rate': 0.1,
        'max_iter': 100,
        'min_samples_leaf': 20,
        'max_depth': None,
        'max_leaf_nodes': 31,
        'max_bins': 255,
        'l2_regularization': 1E-10,
        'tol': 1e-7,
        'scoring': 'loss',
        'n_iter_no_change': 10,
        'validation_fraction': 0.1,
        'warm_start': True,
        'random_state': args.seed,
    },
    'RandomForestClassifier': {
        'n_estimators': 100,
        'criterion': "gini",
        'max_features': 0.5,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_weight_fraction_leaf': 0.,
        'bootstrap': True,
        'max_leaf_nodes': None,
        'min_impurity_decrease': 0.0,
        'random_state': args.seed,
    },
    'DecisionTreeClassifier': {'random_state': args.seed},
    'LinearDiscriminantAnalysis': {},
    'GradientBoostingClassifier': {},
    'MLPClassifier': {},
}
model_func = {
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
    'GradientBoostingClassifier': GradientBoostingClassifier,
    'MLPClassifier': MLPClassifier,
}


df = []
test_history = []
test_history_counter = {}

oof_predictions_avg_repeat, test_predictions_avg_repeat = {}, {}
for repeat in [1, 2, 5, 10, 20]:
    for single_model in [False, True]:
        test_history_counter = {}
        for level in range(0, 7):
            for model_name in list(HPO.keys()) + ['EnsembleSelection']:

                # Save memory
                if level > 2 and 'EnsembleSelection' not in model_name:
                    del oof_predictions_avg_repeat[model_name][level-2]
                    del test_predictions_avg_repeat[model_name][level-2]

                # Prepare for this level model
                if model_name not in oof_predictions_avg_repeat and 'EnsembleSelection' not in model_name:
                    oof_predictions_avg_repeat[model_name] = {}
                    test_predictions_avg_repeat[model_name] = {}

                if model_name not in test_history_counter:
                    test_history_counter[model_name] = 0

                print(f"\nTry fitting {model_name} level={level} with single_model={single_model}")
                try:
                    if model_name == 'EnsembleSelection':
                        past_oof_prediction = [
                            oof_predictions_avg_repeat[model_name_aux][level]
                            for model_name_aux in oof_predictions_avg_repeat.keys() if len(oof_predictions_avg_repeat[model_name_aux]) > 0]
                        past_test_prediction = [
                            test_predictions_avg_repeat[model_name_aux][level]
                            for model_name_aux in test_predictions_avg_repeat.keys() if len(test_predictions_avg_repeat[model_name_aux]) > 0]
                        val_score_repeatedTRUE, val_score_repeatedFALSE, test_score_repeatedTRUE, test_score_repeatedFALSE = fit_and_return_ensemble_predictions(
                            X_train=past_oof_prediction,
                            y_train=y_train.copy(),
                            X_test=past_test_prediction,
                            y_test=y_test.copy(),
                            model_list=[m for m in oof_predictions_avg_repeat.keys() if len(oof_predictions_avg_repeat[m]) > 0],
                        )
                    else:
                        features_train = X_train.copy()
                        features_test = X_test.copy()
                        if level > 0:
                            if not single_model:
                                past_oof_prediction_aux = np.concatenate([
                                    np.mean([a for a in oof_predictions_avg_repeat[model_name_aux][level-1]], axis=0)
                                    for model_name_aux in oof_predictions_avg_repeat.keys()], axis=1)
                                features_train = np.concatenate([
                                    features_train,  past_oof_prediction_aux], axis=1)
                                past_test_prediction_aux = np.concatenate([
                                    np.mean(test_predictions_avg_repeat[model_name_aux][level-1], axis=0)
                                    for model_name_aux in test_predictions_avg_repeat.keys()], axis=1)
                                features_test = np.concatenate([
                                    features_test,  past_test_prediction_aux], axis=1)
                            else:
                                features_train = np.concatenate([
                                    features_train,
                                    np.mean([a for a in oof_predictions_avg_repeat[model_name][level-1]], axis=0)],
                                                                axis=1)
                                features_test = np.concatenate([
                                    features_test,
                                    np.mean(test_predictions_avg_repeat[model_name][level-1], axis=0)],
                                                               axis=1)

                        oof_predictions_avg_repeat[model_name][level], test_predictions_avg_repeat[model_name][level], model, val_score, this_test_history = fit_and_return_avg_predictions(
                            args=args,
                            X_train=features_train,
                            y_train=y_train.copy(),
                            X_test=features_test,
                            y_test=y_test.copy(),
                            model_name=model_name,
                            repeat=repeat,
                        )

                    if model_name == 'EnsembleSelection':
                        for repeated_ensemble in [False, True]:
                            df.append({
                                'hue': f"test_performance_singlemodel{single_model}",
                                'repeated_ensemble': repeated_ensemble,
                                'performance': test_score_repeatedTRUE if repeated_ensemble else test_score_repeatedFALSE,
                                'model': model_name,
                                'dataset_name': args.openml_id,
                                'level': level,
                                'repeat': repeat,
                                'seed': args.seed,
                                'val_score': val_score_repeatedTRUE if repeated_ensemble else val_score_repeatedFALSE,
                            })
                    else:
                        test_score = this_test_history[-1]
                        df.append({
                            'hue': f"test_performance_singlemodel{single_model}",
                            'repeated_ensemble': False,
                            'performance': test_score,
                            'model': model_name,
                            'dataset_name': args.openml_id,
                            'level': level,
                            'repeat': repeat,
                            'seed': args.seed,
                            'val_score': val_score,
                        })

                        for item in this_test_history:
                            test_history.append({
                                'single_model': single_model,
                                'repeated_ensemble': False,
                                'dataset_name': args.openml_id,
                                'model': model_name,
                                'level': level,
                                'repeat': repeat,
                                'iteration': test_history_counter[model_name],
                                'seed': args.seed,
                                'performance': item,
                            })
                            test_history_counter[model_name] += 1

                except Exception as e:
                    traceback.print_exc()
                    print(f"Run into {e} for {model_name}")
            # Save for every level in case of failure
            save_frame(args, df, test_history)

save_frame(args, df, test_history)
