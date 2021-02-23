from shutil import copyfile
import logging
import math
import os
import tempfile as tmp
import warnings
import pickle

import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import autoPyTorch
from autoPyTorch.api.tabular_classification import TabularClassificationTask
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
import autoPyTorch.metrics as metrics

from frameworks.shared.callee import call_run, result, output_subdir, utils


def get_updates_for_regularitzation_cocktails():
    """
    This updates mimc the regularization cocktail paper
    """

    include_updates = {}
    ##############################
    # Not Relevant to the new API:
    ##############################
    # log_level=debug
    # use_tensorboard_logger=False
    # cuda=False
    # num_iterations=840
    # use_pynisher=False

    #############################
    # Controlled by the benchmark
    #############################
    # [honored] optimize_metric=balanced_accuracy
    # [Not-honored->4096] memory_limit_mb=12000
    # [Not-honored->8cores*1h] max_runtime=345600

    ##################
    # General settings
    ##################
    # budget_type=epochs
    # max_budget=105
    # min_budget=105
    # eta=2
    # full_eval_each_epoch=True
    # validation_split=0

    #################################################
    # Hardcoded changes to meet the CS from the paper
    #################################################
    # [honored--embedding is not enabled yet] embeddings=[none]
    # [honored--sampling is not enabled yet] over_sampling_methods=[none]
    # [honored--sampling] under_sampling_methods=[none]
    # [honored--not enabled yet] target_size_strategies=[none]

    updates = HyperparameterSearchSpaceUpdates()
    # networks=['shapedresnet']
    include_updates['network_backbone'] = ['ShapedResNetBackbone']
    # search_space_updates.append(
    #     node_name="NetworkSelector",
    #     hyperparameter="shapedresnet:max_units",
    #     value_range=[args.nr_units],
    #     log=False,
    # )
    updates.append(node_name="network_backbone",
                   hyperparameter="ShapedResNetBackbone:max_units",
                   value_range=[511, 512],
                   default_value=512)
    # search_space_updates.append(
    #     node_name="NetworkSelector",
    #     hyperparameter="shapedresnet:resnet_shape",
    #     value_range=["brick"],
    # )
    updates.append(node_name="network_backbone",
                   hyperparameter="ShapedResNetBackbone:resnet_shape",
                   value_range=['brick'],
                   default_value='brick')
    # search_space_updates.append(
    #     node_name="NetworkSelector",
    #     hyperparameter="shapedresnet:num_groups",
    #     value_range=[2],
    #     log=False,
    # )
    updates.append(node_name="network_backbone",
                   hyperparameter="ShapedResNetBackbone:num_groups",
                   value_range=[1, 2],
                   default_value=2)
    # search_space_updates.append(
    #     node_name="NetworkSelector",
    #     hyperparameter="shapedresnet:blocks_per_group",
    #     value_range=[2],
    #     log=False,
    # )
    updates.append(node_name="network_backbone",
                   hyperparameter="ShapedResNetBackbone:blocks_per_group",
                   value_range=[1, 2],
                   default_value=2)
    # search_space_updates.append(
    #     node_name="CreateDataLoader",
    #     hyperparameter="batch_size",
    #     value_range=[128],
    #     log=False,
    # )
    updates.append(node_name="data_loader",
                   hyperparameter="batch_size",
                   value_range=[128, 129],
                   default_value=128)
    # optimizer=[adamw]
    include_updates['optimizer'] = ['AdamWOptimizer']
    # The learning rate is 10 to the power of -3
    # search_space_updates.append(
    #     node_name="OptimizerSelector",
    #     hyperparameter="adamw:learning_rate",
    #     value_range=[args.learning_rate],
    #     log=False,
    # )
    updates.append(node_name="optimizer",
                   hyperparameter="AdamWOptimizer:lr",
                   value_range=[10e-3, 10.1e-3],
                   default_value=10e-3)
    # initialization_methods=[default]
    include_updates['network_init'] = ['XavierInit']
    # initialization_methods=[default]
    # search_space_updates.append(
    #     node_name="InitializationSelector",
    #     hyperparameter="initializer:initialize_bias",
    #     value_range=['Yes'],
    # )
    updates.append(node_name="network_init",
                   hyperparameter="XavierInit:bias_strategy",
                   value_range=['Zero'],
                   default_value='Zero')
    # preprocessors=[none]
    include_updates['feature_preprocessor'] = ['NoFeaturePreprocessor']
    # imputation_strategies=[median]
    updates.append(node_name="imputer",
                   hyperparameter="numerical_strategy",
                   value_range=['median'],
                   default_value='median')
    # loss_modules=[cross_entropy_weighted]
    # normalization_strategies=[standardize]
    include_updates['scaler'] = ['StandardScaler']
    # @francisco let us leave the scheduler option open and
    # its parameter also open and not hardcoded
    # I was thinking, can we give values to the cosine schedulers
    # on the initial budget and multiplication factor ?
    # because if so, it would make a lot of sense,
    # to give it 12.5 as an initial budget and multiplication factor of 2
    # lr_scheduler=[cosine_annealing]
    updates.append(node_name="lr_scheduler",
                   hyperparameter="CosineAnnealingWarmRestarts:T_0",
                   value_range=[11, 12],
                   default_value=12)
    updates.append(node_name="lr_scheduler",
                   hyperparameter="CosineAnnealingWarmRestarts:T_mult",
                   value_range=[1.9999, 2.0],
                   default_value=2.0)

    # No early stopping
    pipeline_update = {"early_stopping": -1}

    return pipeline_update, updates, include_updates


def run(dataset, config):
    print("\n**** AutoPyTorch {} ****\n".format(autoPyTorch.__version__))
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)

    is_classification = config.type == 'classification'

    # Mapping of benchmark metrics to AutoPyTorch metrics
    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        balacc=metrics.balanced_accuracy,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        # AutoPyTorch can optimize on mse, and we compute rmse independently on predictions
        rmse=metrics.mean_squared_error,
        r2=metrics.r2
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through,
        # or if we use a strict mapping
        #log.warning("Performance metric %s not supported.", config.metric)
       print("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    print(
        "Running AutoPyTorch with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
        config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric
    )
    print("Environment: %s", os.environ)

    # Data Management -- We pass the interpreted dataframe
    # make sure that test and train have both same data type
    column_names, _ = zip(*dataset.columns)
    column_types = dict(dataset.columns)
    train = pd.DataFrame(dataset.train.data, columns=column_names).astype(column_types, copy=False)
    label = dataset.target.name
    print(f"Columns Before dtypes:\n{train.dtypes}")
    test = pd.DataFrame(dataset.test.data, columns=column_names).astype(column_types, copy=False)
    X_train = train.drop(columns=label)
    y_train = train[label]
    X_test = test.drop(columns=label)
    y_test = test[label]
    # Objects happen in automlbenchmark -- fix that
    X_train = X_train.convert_dtypes()
    for x in X_train.columns:
        X_test[x] = X_test[x].astype(X_train[x].dtypes.name)
        if not is_numeric_dtype(X_test[x]):
            X_test[x] = X_test[x].astype('category')
            X_train[x] = X_train[x].astype('category')
    print(f"Columns dtypes after fix :\n{X_train.dtypes}")

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    ml_memory_limit = config.framework_params.get('_ml_memory_limit', 'auto')

    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = utils.system_memory_mb().total
    if ml_memory_limit == 'auto':
        ml_memory_limit = max(
            min(
                config.max_mem_size_mb / n_jobs,
                math.ceil(total_memory_mb / n_jobs)
            ),
            3072  # 3072 is AutoPyTorch default and we use it as a lower bound
        )

    print("Using %sMB memory per ML job and %sMB for ensemble job on a total of %s jobs.",
             ml_memory_limit, ml_memory_limit, n_jobs)

    # Add updates for the regularization cocktails
    # pipeline_update -> Updates to the pipelines to be created
    # Updates to the configuration space -> the learning rate value and so on
    # include_updates-> Fix the backbone to be specifically something
    pipeline_update, search_space_updates, include_updates = get_updates_for_regularitzation_cocktails()

    estimator = TabularClassificationTask if is_classification else NotImplementedError()
    api = estimator(n_jobs=n_jobs,
                    delete_tmp_folder_after_terminate=False,
                    include_components=include_updates,
                    search_space_updates=search_space_updates,
                    **training_params)

    # Add pipeline updates
    api.set_pipeline_config(**pipeline_update)

    print(f"X_train={X_train.iloc[0]} X_test={X_test.iloc[0]}")
    with utils.Timer() as training:
        api.search(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            optimize_metric=perf_metric.name,
            # Be a little bit pessimistic towards time
            total_walltime_limit=config.max_runtime_seconds-10,
        )

    # Convert output to strings for classification
    #log.info("Predicting on the test set.")
    print("Predicting on the test set.")
    predictions = api.predict(X_test)
    probabilities = api.predict_proba(X_test) if is_classification else None

    print("Saving Artifacts")
    save_artifacts(api, config)

    try:
        print(api.run_history, api.trajectory)
        print(api.ensemble_)
        print(api.ensemble_.get_selected_model_identifiers())
        print(api.ensemble_.weights_)
        print(api.show_models())
    except Exception as e:
        print(f"Run into {e} while printing information")

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=False,
                  # TODO
                  models_count=len([]),
                  training_duration=training.duration)


def save_artifacts(estimator, config):
    try:
        #models_repr = estimator.show_models()
        #log.debug("Trained Ensemble:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            print("Saving Artifacts -- models")
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
        if 'debug' in artifacts:
            print("Saving Artifacts -- debug")
            debug_dir = output_subdir('debug', config)
            ignore_extensions = ['.npy', '.pcs', '.model', '.ensemble', '.pkl', '.cv_model', '.pth']
            tmp_directory = estimator._backend.temporary_directory
            files_to_copy = []
            for r, d, f in os.walk(tmp_directory):
                for file_name in f:
                    base, ext = os.path.splitext(file_name)
                    if ext not in ignore_extensions:
                        files_to_copy.append(os.path.join(r, file_name))
            for filename in files_to_copy:
                dst = filename.replace(tmp_directory, debug_dir+'/')
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                copyfile(filename, dst)
    except Exception as e:
        log.debug(f"Error when saving artifacts= {e}.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
