import logging
import math
import os
import tempfile as tmp
import warnings

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from autoPyTorch import AutoNetClassification, AutoNetRegression

from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import Encoder
from amlb.results import save_predictions_to_file
from amlb.utils import Timer, system_memory_mb, touch

log = logging.getLogger(__name__)


def run(dataset: Dataset, config: TaskConfig):
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

    # Mapping of benchmark metrics to AutoPyTorch metrics
    # TODO Some metrics are not yet implemented in framework
    metrics_mapping = dict(
        acc='accuracy',
        auc='auc_metric',
        f1='auc_metric',
        logloss='cross_entropy',
        mae='mean_distance',
        mse='mean_distance',
        r2='auc_metric'
    )
    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        log.exception("Performance metric %s not supported.", config.metric)

    # Set resources based on datasize
    log.info("Running AutoPyTorch with a maximum time of %ss on %s cores with %sMB, optimizing %s.",
             config.max_runtime_seconds, config.cores, config.max_mem_size_mb, perf_metric)
    log.info("Environment: %s", os.environ)

    # Data Processing
    X_train = dataset.train.X_enc
    y_train = dataset.train.y_enc
    X_test= dataset.test.X_enc
    y_test = dataset.test.y_enc

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    n_jobs = config.framework_params.get('_n_jobs', config.cores)
    ml_memory_limit = config.framework_params.get('_ml_memory_limit', 'auto')

    # Use for now same auto memory setting from autosklearn
    # when memory is large enough, we should have:
    # (cores - 1) * ml_memory_limit_mb + ensemble_memory_limit_mb = config.max_mem_size_mb
    total_memory_mb = system_memory_mb().total
    if ml_memory_limit == 'auto':
        ml_memory_limit = max(min(config.max_mem_size_mb,
                                  math.ceil(total_memory_mb / n_jobs)),
                              3072)  # 3072 is autosklearn defaults


    # Find the model that best fit the data
    # TODO add seed support when implemented on the framework
    is_classification = config.type == 'classification'
    autonet = AutoNetClassification if is_classification else AutoNetRegression
    auto_pytorch = autonet(
        budget_type="time",
        min_budget=config.max_runtime_seconds//(2*3**2), # if eta=3, this should lead to 3 budgets
        max_budget=config.max_runtime_seconds//2,
        max_runtime=config.max_runtime_seconds,
        log_level='info',
        use_pynisher=False,
        min_workers=n_jobs,
        memory_limit_mb=ml_memory_limit,
        **training_params
    )

    with Timer() as training:
        auto_pytorch.fit(X_train, y_train, optimize_metric=perf_metric)

    # Convert output to strings for classification
    log.info("Predicting on the test set.")
    predictions = auto_pytorch.predict(X_test)
    if is_classification:
        target_values_enc = dataset.target.label_encoder.transform(dataset.target.values)
        probabilities = Encoder('one-hot', target=False, encoded_type=float).fit(target_values_enc).transform(predictions)
    else:
        probabilities = None

    save_predictions_to_file(dataset=dataset,
                             output_file=config.output_predictions_file,
                             probabilities=probabilities,
                             predictions=predictions,
                             truth=y_test,
                             target_is_encoded=True)

    save_artifacts(auto_pytorch, config)

    return dict(
        models_count=len(auto_pytorch.get_pytorch_model()),
        training_duration=training.duration
    )


def make_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


def save_artifacts(autonet, config):
    try:
        models_repr = autonet.get_pytorch_model()
        log.debug("Trained Model:\n%s", models_repr)
        artifacts = config.framework_params.get('_save_artifacts', [])
        if 'models' in artifacts:
            models_file = os.path.join(make_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                f.write(models_repr)
    except:
        log.debug("Error when saving artifacts.", exc_info=True)
