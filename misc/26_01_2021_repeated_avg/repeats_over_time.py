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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
import sklearn.datasets
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from sklearn import preprocessing
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
parser.add_argument(
    '--openml_id',
    required=True,
    type=int,
)
parser.add_argument(
    '--seed',
    required=False,
    default=42,
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
}


df = []

rskf = RepeatedStratifiedKFold(n_splits=args.n_splits,
                               n_repeats=args.n_repeats, random_state=args.seed)
for model_name in HPO.keys():
    print(f"Try fitting {model_name}")
    try:
        if model_name == 'HistGradientBoostingClassifier':
            model_func = HistGradientBoostingClassifier
        elif model_name == 'RandomForestClassifier':
            model_func = RandomForestClassifier
        elif model_name == 'DecisionTreeClassifier':
            model_func = DecisionTreeClassifier
        elif model_name == 'LinearDiscriminantAnalysis':
            model_func = LinearDiscriminantAnalysis

        model = [model_func(**HPO[model_name]) for i in range(args.n_splits*args.n_repeats)]

        # Predicitions
        y_targets = [None] * args.n_splits * args.n_repeats
        oof_predictions = [None] * args.n_splits * args.n_repeats
        test_predictions = [None] * args.n_splits * args.n_repeats
        oof_indices = [None] * args.n_splits * args.n_repeats
        train_scores = [None] * args.n_splits * args.n_repeats

        for i, (train_index, val_index) in enumerate(rskf.split(X_train, y_train)):
            print(f"==> Fitting model {model_name} {i}")
            X_train_, X_val_ = X_train[train_index], X_train[val_index]
            y_train_, y_val_ = y_train[train_index], y_train[val_index]
            oof_indices[i] = val_index
            y_targets[i] = y_val_

            # Fit the model to the train data
            model[i].fit(X_train_, y_train_)

            # Predict to eval the model
            oof_predictions[i] = model[i].predict_proba(X_val_)
            test_predictions[i] = model[i].predict_proba(X_test)
            train_scores[i] = balanced_accuracy_score(y_train_, model[i].predict_proba(X_train_).argmax(1))

        oof_indices = np.concatenate([oof_indices[i] for i in range(args.n_splits*args.n_repeats)
                                      if oof_indices[i] is not None])

        oof_predictions = np.concatenate([oof_predictions[i] for i in range(args.n_splits*args.n_repeats)
                                          if oof_predictions[i] is not None])

        y_targets = np.concatenate([y_targets[i] for i in range(args.n_splits*args.n_repeats)
                                    if y_targets[i] is not None])

        oof_indices = np.split(oof_indices, args.n_repeats)
        oof_predictions = np.split(oof_predictions, args.n_repeats)
        y_targets = np.split(y_targets, args.n_repeats)
        for i in range(args.n_repeats):
            y_targets[i] = y_targets[i][np.argsort(oof_indices[i])]
            oof_predictions[i] = oof_predictions[i][np.argsort(oof_indices[i])]

        # All targets are same, pick first one
        y_targets = y_targets[0]

        # Convert to repetitions
        # Test
        results = {
            'avg_test_performance': [],
            'avg_val_performance': [],
            'avg_train_performance': [],
            # If we do not average
            'nonavg_test_performance': [],
            'nonavg_val_performance': [],
            'nonavg_train_performance': [],
            'model': [model_name] * args.n_repeats,
            'dataset_name': [args.openml_id] * args.n_repeats,
            'repeat': list(range(args.n_repeats)),
        }

        for average in [True, False]:
            for subset in ['train', 'test', 'val']:
                for i in range(args.n_repeats):
                    # Adjust depending on avg
                    key = f"nonavg_{subset}_performance" if not average else f"avg_{subset}_performance"
                    if subset == 'train':
                        # Every fold has a train performance
                        start_fold = i*args.n_splits if not average else 0
                        end_fold = (i+1)*args.n_splits
                        results[key].append(np.mean(train_scores[start_fold:end_fold]))
                        continue
                    elif subset == 'test':
                        # Every split has a test prediction
                        start_fold = i*args.n_splits if not average else 0
                        end_fold = (i+1)*args.n_splits
                        expected = y_test
                        prediction_source = test_predictions
                    elif subset == 'val':
                        # You only have here a full y_train per repetition
                        start_fold = i if not average else 0
                        end_fold = i + 1
                        expected = y_targets
                        prediction_source = oof_predictions
                    else:
                        raise NotImplementedError(subset)
                    prediction = np.mean(prediction_source[start_fold:end_fold], axis=0)
                    results[key].append(balanced_accuracy_score(expected, prediction.argmax(1)))
        df.append(results)
    except Exception as e:
        print(f"Run into {e} for {model_name}")

# Integrate that through progression
df = pd.concat([pd.DataFrame(d) for d in df]).reset_index(drop=True)
pd.set_option('display.max_rows', len(df))
print(df)
path = os.path.join(os.getenv('HOME'), f"df_repeats_{args.openml_id}.csv")
print(f"check {path} for results")
df.to_csv(path)
