from operator import itemgetter
import numpy as np
import logging
import os
import shutil
import warnings
import tempfile as tmp
warnings.simplefilter("ignore")

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPrediction as task
from autogluon.core.utils.savers import save_pkl, save_pd
import  autogluon.core.metrics as metrics
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular.models import AbstractModel
import pandas as pd

from frameworks.shared.callee import call_run, result, output_subdir, utils


log = logging.getLogger(__name__)


#########################
# Create a custom model #
#########################
class MLPModel(AbstractModel):

    # TODO: X.fillna -inf? Add extra is_missing column?
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self, X_train, y_train, **kwargs):
        from sklearn.neural_network import MLPClassifier
        X_train = self.preprocess(X_train)
        self.model = MLPClassifier(
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size='auto',
            learning_rate_init=1e-3,
            warm_start=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=32,
        )
        self.model.fit(X_train, y_train)


class HistGradientBoostingClassifierModel(AbstractModel):

    # TODO: X.fillna -inf? Add extra is_missing column?
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    # The `_fit` method takes the input training data (and optionally the validation data) and trains the model.
    def _fit(self, X_train, y_train, **kwargs):
        import sklearn.ensemble
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        X_train = self.preprocess(X_train)
        self.model = sklearn.ensemble.HistGradientBoostingClassifier(
            loss='auto',
            learning_rate=0.1,
            min_samples_leaf=20,
            max_depth=None,
            max_leaf_nodes=31,
            max_bins=255,
            l2_regularization=1E-10,
            tol=1e-7,
            scoring='loss',
            early_stopping='off',
            n_iter_no_change=10,
            validation_fraction=0.1,
            warm_start=True,
        )
        self.model.fit(X_train, y_train)


def run(dataset, config):
    log.info("\n**** AutoGluon ****\n")

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        # rmse=metrics.root_mean_squared_error,  # metrics.root_mean_squared_error incorrectly registered in autogluon REGRESSION_METRICS
        rmse=metrics.mean_squared_error,  # for now, we can let autogluon optimize training on mse: anyway we compute final score from predictions.
        balacc=metrics.balanced_accuracy,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
    label = dataset.target.name
    print(f"Columns dtypes:\n{train.dtypes}")

    custom_hyperparameters = {HistGradientBoostingClassifierModel: {},
                              MLPModel: {}}
    custom_hyperparameters.update(get_hyperparameter_config('default'))
    # dict_keys(['NN', 'GBM', 'CAT', 'XGB', 'FASTAI', 'RF', 'XT', 'KNN', 'custom'])
    del custom_hyperparameters['NN']
    del custom_hyperparameters['GBM']
    del custom_hyperparameters['XGB']
    del custom_hyperparameters['FASTAI']
    del custom_hyperparameters['CAT']
    del custom_hyperparameters['custom']

    #output_dir = output_subdir("models", config)
    output_dir = tmp.gettempdir()
    with utils.Timer() as training:
        predictor = task.fit(
            hyperparameters=custom_hyperparameters,
            train_data=train,
            label=label,
            problem_type=dataset.problem_type,
            output_directory=output_dir,
            time_limits=config.max_runtime_seconds,
            eval_metric=perf_metric.name,
            **training_params
        )

    test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)
    X_test = test.drop(columns=label)
    y_test = test[label]

    print(f"Fonished training now to predict")
    with utils.Timer() as predict:
        predictions = predictor.predict(X_test)

    probabilities = predictor.predict_proba(dataset=X_test, as_pandas=True, as_multiclass=True) if is_classification else None
    prob_labels = probabilities.columns.values.tolist() if probabilities is not None else None
    print(f"Fonished even predict")

    try:
        leaderboard = predictor._learner.leaderboard(X_test, y_test, silent=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
            print(leaderboard)

        #overfit_frame = generate_overfit_artifacts(predictor, test)
        #save_artifacts(predictor, leaderboard, config, overfit_frame=None)
        num_models_trained = len(leaderboard)
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    except Exception as e:
        num_models_ensemble = 0
        num_models_trained = 0
        print(f"{e} happened")


    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def save_artifacts(predictor, leaderboard, config, overfit_frame):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        models_dir = output_subdir("models", config)
        shutil.rmtree(os.path.join(models_dir, "utils"), ignore_errors=True)

        if 'overfit' in artifacts and overfit_frame is not None:
            overfit_file = os.path.join(output_subdir('overfit', config), 'overfit.csv')
            overfit_frame.to_csv(overfit_file)

        if 'leaderboard' in artifacts:
            save_pd.save(path=os.path.join(models_dir, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            ag_info = predictor.info()
            info_dir = output_subdir("info", config)
            save_pkl.save(path=os.path.join(info_dir, "info.pkl"), object=ag_info)

        if 'models' in artifacts:
            utils.zip_path(models_dir,
                           os.path.join(models_dir, "models.zip"))

        def delete(path, isdir):
            if isdir:
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.splitext(path)[1] == '.pkl':
                os.remove(path)
        utils.walk_apply(models_dir, delete, max_depth=0)

    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)


def generate_overfit_artifacts(predictor, test_data):
    dataframe = []
    X, y = predictor._learner.extract_label(test_data)
    X = predictor._learner.transform_features(X)
    y = predictor._learner.label_cleaner.transform(y)
    if predictor._learner.problem_type == MULTICLASS:
        y = y.fillna(-1)

    models = []
    train_scores = []
    val_scores = []
    test_scores = []

    models_info = predictor._trainer.get_models_info()
    for model, info in models_info.items():
        try:
            models.append(model)
            if info['train_score']:
                train_scores.append(info['train_score'])
            else:
                train_scores.append(np.inf)
            val_scores.append(info['val_score'])
            actual_model = predictor._trainer.load_model(model)
            test_scores.append(actual_model.score(X, y))
        except Exception as e:
            print(f"failed with {e} on {model}")

    individual_models = [(m, train, val, test) for m, train, val, test in  zip(models, train_scores, val_scores, test_scores) if 'ensemble' not in m]
    m, train, val, test = sorted(individual_models, key=itemgetter(2))[-1]

    dataframe.append({
        'model': 'best_individual_model' + m,
        'test': test,
        'val': val,
        'train': train,
    })

    best_ensemble_index = np.argmax(val_scores)
    dataframe.append({
        'model': 'best_ensemble_model' + models[best_ensemble_index],
        'test': test_scores[best_ensemble_index],
        'val': val_scores[best_ensemble_index],
        'train': train_scores[best_ensemble_index],
    })

    # maybe val also but train, I don't know rick
    dataframe.append({
        'model': 'overall',
        'test': predictor.evaluate(test_data),
        'val': np.inf,
        'train': predictor.info()['best_model_score_val'],
    })
    return pd.DataFrame(dataframe)


if __name__ == '__main__':
    call_run(run)
