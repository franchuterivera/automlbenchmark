import time
from random import randrange
import tempfile
from sklearn.utils.multiclass import type_of_target
import traceback
import os
import openml
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import argparse
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
import sklearn.datasets
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from sklearn import preprocessing
import sys
import numpy
from xgboost_model import XGBoostModel
from rf_model import RFModel, XTModel
from abstract_model import balanced_accuracy
from pandas.api.types import is_numeric_dtype
from tabular_nn_model import TabularNeuralNetModel
from lr_model import LinearModel
from lgb_model import LGBModel
from catboost_model import CatBoostModel

numpy.set_printoptions(threshold=sys.maxsize)


def fit_and_return_avg_predictions(
        args, X_train, y_train, X_test, y_test, model_name, repeat
):
    # Predicitions
    indices = [None] * args.n_splits * repeat
    oof_predictions = [None] * args.n_splits * repeat
    test_predictions = [None] * args.n_splits * repeat
    model = [None] * args.n_splits * repeat

    problem_type = type_of_target(y_train)

    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits,
                                   n_repeats=repeat, random_state=args.seed)

    # First train everything, then worry about format at end
    for i, (train_index, val_index) in enumerate(rskf.split(X_train.copy(), y_train.copy())):
        print(f"==> {i} Fitting model {model_name}")

        with tempfile.TemporaryDirectory() as tmpdirname:
            model[i] = model_func[model_name](
                name=model_name,
                path=f"{tmpdirname}/",
                problem_type=problem_type,
                metric=balanced_accuracy,
                eval_metric=balanced_accuracy,
            )
            model[i].fit(
                X_train=X_train.iloc[train_index],
                y_train=y_train.iloc[train_index],
            )

            # Predict to eval the model
            indices[i] = val_index
            oof_predictions[i] = model[i].predict_proba(X_train.iloc[val_index])
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
        this_test_history.append(balanced_accuracy_score(
            y_test.to_numpy().astype(int),
            np.round(this_test_predictions.mean(axis=1))))

    oof_predictions = np.concatenate([a.argmax(1).reshape(-1, 1) for a in oof_predictions], axis=1)
    test_predictions = np.concatenate([a.argmax(1).reshape(-1, 1) for a in test_predictions], axis=1)
    print(f"From {X_train.shape} produced {oof_predictions.shape} and {test_predictions.shape} with val score={balanced_accuracy_score(y_train, np.round(np.mean(oof_predictions, axis=1), decimals=0))}")

    final_model = VotingClassifier(estimators=None, voting='soft', )
    final_model.estimators_ = model

    return oof_predictions.mean(axis=1).reshape(-1, 1), np.round(test_predictions.mean(axis=1), decimals=0).reshape(-1, 1), final_model, this_test_history

def save_frame(args, repeated_frame, history_frame):
    # Integrate that through progression
    path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_autogluon_{args.openml_id}.csv")
    pd.DataFrame(repeated_frame).to_csv(path)
    print(f"{time.ctime()}: saved {path}")
    path = os.path.join(os.getenv('HOME'), f"df_repeated_stacking_autogluon_history_{args.openml_id}.csv")
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
    '--n_repeat',
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
X, y = sklearn.datasets.fetch_openml(data_id=task.dataset_id, return_X_y=True, as_frame=True)
X = X.convert_dtypes()
for x in X.columns:
    if not is_numeric_dtype(X[x]):
        X[x] = X[x].astype('category')
    elif 'Int' in str(X[x].dtype):
        X[x] = X[x].astype(np.int)
print(f"Working on task {task} with data X({np.shape(X)})={X.dtypes}")
#imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
#X = imp_mean.fit_transform(X)
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
y = pd.DataFrame(y)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, random_state=args.seed,
)
print(f"Loaded {args.openml_id} with {X_train.shape} train datapoints and {X_test.shape} test datapoints")

model_func = {
    'XGBoostModel': XGBoostModel,
    'XTModel': XTModel,
    'RFModel': RFModel,
    'TabularNeuralNetModel': TabularNeuralNetModel,
    'LinearModel': LinearModel,
    'LGBModel': LGBModel,
    'CatBoostModel': CatBoostModel,
}

df = []
test_history = []
test_history_counter = {}
oof_predictions_avg_repeat, test_predictions_avg_repeat = {}, {}

for use_train_data in [True, False]:
    for repeat in [5, 10, 20]:
        for single_model in [True, False]:
            test_history_counter = {}
            for level in range(0, 6):
                for model_name in model_func.keys():
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

                    print(f"\nTry fitting {model_name} level={level} with single_model={single_model} repeat={repeat} use_train_data={use_train_data}")
                    try:
                        features_train = X_train.copy()
                        features_test = X_test.copy()
                        columns = features_train.columns
                        if level > 0:
                            if not single_model:
                                #past_oof_prediction_aux = np.concatenate([
                                #    oof_predictions_avg_repeat[model_name_aux][level-1]
                                #    for model_name_aux in oof_predictions_avg_repeat.keys()], axis=1)
                                #features_train = np.concatenate([
                                #    features_train,  past_oof_prediction_aux], axis=1)
                                for model_name_aux in oof_predictions_avg_repeat.keys():
                                    if level-1 not in oof_predictions_avg_repeat[model_name_aux]:
                                        print(f"Error {level - 1} oof_predictions_avg_repeat missing for {model_name_aux}")
                                        continue
                                    features_train[f"level_{level}_{model_name_aux}"] = oof_predictions_avg_repeat[model_name_aux][level-1]

                                #past_test_prediction_aux = np.concatenate([
                                #    test_predictions_avg_repeat[model_name_aux][level-1]
                                #    for model_name_aux in test_predictions_avg_repeat.keys()], axis=1)
                                #features_test = np.concatenate([
                                #    features_test,  past_test_prediction_aux], axis=1)
                                for model_name_aux in test_predictions_avg_repeat.keys():
                                    if level-1 not in test_predictions_avg_repeat[model_name_aux]:
                                        print(f"Error {level - 1} test_predictions_avg_repeat for {model_name_aux}")
                                        continue
                                    features_test[f"level_{level}_{model_name_aux}"] = test_predictions_avg_repeat[model_name_aux][level-1]

                            else:
                                #features_train = np.concatenate([
                                #    features_train, oof_predictions_avg_repeat[model_name][level-1]],
                                #                                axis=1)
                                features_train[f"level_{level}"] = oof_predictions_avg_repeat[model_name][level-1]
                                #features_test = np.concatenate([
                                #    features_test, test_predictions_avg_repeat[model_name][level-1]],
                                #                               axis=1)
                                features_test[f"level_{level}"] = test_predictions_avg_repeat[model_name][level-1]
                            if not use_train_data:
                                features_train = features_train.drop(axis='columns', labels=columns)
                                features_test = features_test.drop(axis='columns', labels=columns)

                        oof_predictions_avg_repeat[model_name][level], test_predictions_avg_repeat[model_name][level], model, this_test_history = fit_and_return_avg_predictions(
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
                        #    'repeat': repeat,
                        #    'seed': args.seed,
                        #})
                        test_score = balanced_accuracy_score(y_test,
                                                             model.predict_proba(features_test).argmax(1))
                        df.append({
                            'hue': f"test_performance_singlemodel{single_model}",
                            'performance': test_score,
                            'model': model_name,
                            'dataset_name': args.openml_id,
                            'level': level,
                            'repeat': repeat,
                            'seed': args.seed,
                            'use_train_data': use_train_data,
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
                                'use_train_data': use_train_data,
                            })
                            test_history_counter[model_name] += 1
                    except Exception as e:
                        traceback.print_exc()
                        print(f"Run into {e} for {model_name}")

            # Save for every level in case of failure
            save_frame(args, df, test_history)

save_frame(args, df, test_history)
