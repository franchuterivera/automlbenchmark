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
        this_test_predictions = np.concatenate([test_predictions[a].argmax(1).reshape(-1, 1) for a in list(range(i + 1))], axis=1)
        this_test_history.append(balanced_accuracy_score(y_test, np.round(this_test_predictions.mean(axis=1))))

    # calculate oof score
    val_score = balanced_accuracy_score(y_train, np.round(np.mean([a.argmax(1).reshape(-1, 1) for a in oof_predictions], axis=0)))

    # Convert to an average array
    oof_predictions = np.concatenate([a for a in oof_predictions], axis=1)
    test_predictions = np.concatenate(          [np.mean(test_predictions[args.n_splits*i:args.n_splits*(i+1)], axis=0) for i in range(0, repeat)], axis=1)
    print(f"From X_train={X_train.shape} {oof_predictions.shape} and {test_predictions.shape} with val val_score={val_score} test_score={this_test_history[-1]}")

    final_model = VotingClassifier(estimators=None, voting='soft', )
    final_model.estimators_ = model

    return oof_predictions, test_predictions, final_model, val_score, this_test_history



def save_frame(args, repeated_frame, history_frame):
    # Integrate that through progression
    path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_sklearn_lgbmhpoonly_{args.openml_id}_{args.seed}.csv")
    pd.DataFrame(repeated_frame).to_csv(path)
    print(f"{time.ctime()}: saved {path}")
    path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_sklearn_lgbmhpoonly_history_{args.openml_id}_{args.seed}.csv")
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
    'LGBMClassifierDefault': {},
    'LGBMClassifierAutoGluon': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'two_round': True,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
    },
    'LGBMClassifier1': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.001,
        'two_round': True,
        'num_leaves': 128,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
    },
    'LGBMClassifier2': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'two_round': True,
        'num_leaves': 60,
        'feature_fraction': 0.8,
    },
    'LGBMClassifier3': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.06,
        'two_round': True,
        'num_leaves': 5,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 4,
    },
    'LGBMClassifier4': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'two_round': True,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 3,
    },
    'LGBMClassifier5': {
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'two_round': True,
        'num_leaves': 50,
        'feature_fraction': 0.9,
        'min_data_in_leaf': 3,
    },
    'MLPClassifier': {},
}
model_func = {
    'LGBMClassifier': LGBMClassifier,
    'LGBMClassifierDefault': LGBMClassifier,
    'LGBMClassifierAutoGluon': LGBMClassifier,
    'LGBMClassifier1': LGBMClassifier,
    'LGBMClassifier2': LGBMClassifier,
    'LGBMClassifier3': LGBMClassifier,
    'LGBMClassifier4': LGBMClassifier,
    'LGBMClassifier5': LGBMClassifier,
    'MLPClassifier': MLPClassifier,
}


df = []
test_history = []
test_history_counter = {}

oof_predictions_avg_repeat, test_predictions_avg_repeat = {}, {}
for repeat in [1, 2, 5, 10, 20]:
    for single_model in [True, False]:
        test_history_counter = {}
        for level in range(0, 4):
            for model_name in HPO.keys():
                # Save memory
                if level > 2:
                    del oof_predictions_avg_repeat[model_name][level-2]
                    del test_predictions_avg_repeat[model_name][level-2]

                # Prepare for this level model
                if model_name not in oof_predictions_avg_repeat:
                    oof_predictions_avg_repeat[model_name] = {}
                    test_predictions_avg_repeat[model_name] = {}

                if model_name not in test_history_counter:
                           test_history_counter[model_name] = 0

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

                    oof_predictions_avg_repeat[model_name][level], test_predictions_avg_repeat[model_name][level], model, val_score, this_test_history = fit_and_return_avg_predictions(
                        args=args,
                        X_train=features_train,
                        y_train=y_train.copy(),
                        X_test=features_test,
                        y_test=y_test.copy(),
                        model_name=model_name,
                        repeat=repeat,
                    )
                    #train_score = balanced_accuracy_score(y_train,
                    #                                      model.predict_proba(features_train).argmax(1))
                    #df.append({
                    #    'hue': f"train_performance_singlemodel{single_model}",
                    #    'performance': train_score,
                    #    'model': model_name,
                    #    'dataset_name': args.openml_id,
                    #    'level': level,
                    #})
                    test_score = this_test_history[-1]

                    df.append({
                        'hue': f"test_performance_singlemodel{single_model}",
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