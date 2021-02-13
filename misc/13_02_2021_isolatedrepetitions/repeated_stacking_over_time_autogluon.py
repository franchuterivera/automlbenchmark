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
model_func = {
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
}



def fit_and_return_avg_predictions(
        args, X_train, y_train, X_test, model_name
):
    # Predicitions
    indices = [None] * args.n_splits * args.n_repeats
    oof_predictions = [None] * args.n_splits * args.n_repeats
    test_predictions = [None] * args.n_splits * args.n_repeats
    model = [None] * args.n_splits * args.n_repeats

    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits,
                                   n_repeats=args.n_repeats, random_state=args.seed)

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
        test_predictions[i] = model[i].predict_proba(features_test)

    indices = np.concatenate([indices[i] for i in range(args.n_splits*args.n_repeats)
                              if indices[i] is not None])

    oof_predictions = np.concatenate([oof_predictions[i] for i in range(args.n_splits*args.n_repeats)
                                      if oof_predictions[i] is not None])
    indices = np.split(indices, args.n_repeats)
    oof_predictions = np.split(oof_predictions, args.n_repeats)
    for i in range(args.n_repeats):
        oof_predictions[i] = oof_predictions[i][np.argsort(indices[i])]

    # Convert to an average array
    oof_predictions = np.concatenate([a.argmax(1).reshape(-1, 1) for a in oof_predictions], axis=1)
    test_predictions = np.concatenate([a.argmax(1).reshape(-1, 1) for a in test_predictions], axis=1)
    print(f"Produced {oof_predictions.shape} and {test_predictions.shape} with val score={balanced_accuracy_score(y_train, np.round(np.mean(oof_predictions, axis=1), decimals=0))}")

    final_model = VotingClassifier(estimators=None, voting='soft', )
    final_model.estimators_ = model

    return np.round(oof_predictions.mean(axis=1), decimals=0).reshape(-1, 1), np.round(test_predictions.mean(axis=1), decimals=0).reshape(-1, 1), final_model


df = []
oof_predictions_avg_repeat, test_predictions_avg_repeat = {}, {}
for single_model in [True, False]:
    for level in range(0, 10):
        for model_name in HPO.keys():
            # Save memory
            if level > 2:
                del oof_predictions_avg_repeat[model_name][level-2]
                del test_predictions_avg_repeat[model_name][level-2]

            # Prepare for this level model
            if model_name not in oof_predictions_avg_repeat:
                oof_predictions_avg_repeat[model_name] = {}
                test_predictions_avg_repeat[model_name] = {}

            print(f"\nTry fitting {model_name} level={level} with single_model={single_model}")
            try:
                features_train = X_train.copy()
                features_test = X_test.copy()
                if level > 0:
                    if not single_model:
                        past_oof_prediction_aux = np.concatenate([
                            oof_predictions_avg_repeat[model_name_aux][level-1]
                            for model_name_aux in oof_predictions_avg_repeat.keys()], axis=1)
                        features_train = np.concatenate([
                            features_train,  past_oof_prediction_aux], axis=1)
                        past_test_prediction_aux = np.concatenate([
                            test_predictions_avg_repeat[model_name_aux][level-1]
                            for model_name_aux in test_predictions_avg_repeat.keys()], axis=1)
                        features_test = np.concatenate([
                            features_test,  past_test_prediction_aux], axis=1)
                    else:
                        features_train = np.concatenate([
                            features_train, oof_predictions_avg_repeat[model_name][level-1]],
                                                        axis=1)
                        features_test = np.concatenate([
                            features_test, test_predictions_avg_repeat[model_name][level-1]],
                                                       axis=1)

                oof_predictions_avg_repeat[model_name][level], test_predictions_avg_repeat[model_name][level], model = fit_and_return_avg_predictions(
                    args=args,
                    X_train=features_train,
                    y_train=y_train.copy(),
                    X_test=X_test.copy(),
                    model_name=model_name,
                )
                train_score = balanced_accuracy_score(y_train,
                                                      model.predict_proba(features_train).argmax(1))
                test_score = balanced_accuracy_score(y_test,
                                                     model.predict_proba(features_test).argmax(1))

                df.append({
                    'hue': f"train_performance_singlemodel{single_model}",
                    'performance': train_score,
                    'model': model_name,
                    'dataset_name': args.openml_id,
                    'level': level,
                })
                df.append({
                    'hue': f"test_performance_singlemodel{single_model}",
                    'performance': test_score,
                    'model': model_name,
                    'dataset_name': args.openml_id,
                    'level': level,
                })
            except Exception as e:
                traceback.print_exc()
                print(f"Run into {e} for {model_name}")

# Integrate that through progression
df = pd.DataFrame(df)
pd.set_option('display.max_rows', len(df))
print(df)
path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_{args.openml_id}.csv")
print(f"check {path} for results")
df.to_csv(path)
