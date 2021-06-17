import numpy as np
import sklearn.preprocessing
import sklearn.utils
from amlb.benchmark import TaskConfig
from amlb.data import Dataset
from amlb.datautils import Encoder, impute
from amlb.resources import config as rconfig
from amlb.utils import call_script_in_same_dir


def setup(*args, **kwargs):
    call_script_in_same_dir(__file__, "setup.sh", rconfig().root_dir, *args, **kwargs)


def run(dataset: Dataset, config: TaskConfig):
    from frameworks.shared.caller import run_in_venv

    X_train_enc, X_test_enc = impute(dataset.train.X_enc, dataset.test.X_enc)
    data = dict(
        train=dict(
            X_enc=X_train_enc,
            y_enc=dataset.train.y_enc
        ),
        test=dict(
            X_enc=X_test_enc,
            y_enc=dataset.test.y_enc
        )
    )

    def process_results(results):
        if results.probabilities is not None and not results.probabilities.shape:  # numpy load always return an array
            prob_format = results.probabilities.item()
            if prob_format == "predictions":
                try:
                    target_values_enc = dataset.target.label_encoder.transform(dataset.target.values)
                    print(f"Before we have {target_values_enc} {type(target_values_enc)} {target_values_enc.dtype} results.predictions={results.predictions} {type(results.predictions)} {results.predictions.dtype}")
                    target_values_enc = sklearn.utils.check_array(
                        target_values_enc,
                        force_all_finite=True,
                        accept_sparse='csr',
                        ensure_2d=False,
                    )
                    predictions = sklearn.utils.check_array(
                        results.predictions,
                        force_all_finite=True,
                        accept_sparse='csr',
                        ensure_2d=False,
                    )
                    results.probabilities = Encoder('one-hot', target=False, encoded_type=float).fit(target_values_enc).transform(predictions)
                except Exception as e:
                    print(f"Failed with {e} for {results.predictions}({type(results.predictions)}) {results.predictions.dtype} target_values_enc={target_values_enc}({type(target_values_enc)})")
                    golden = sklearn.utils.check_array(
                        target_values_enc,
                        force_all_finite=True,
                        accept_sparse='csr',
                        ensure_2d=False,
                    )
                    predictions = sklearn.utils.check_array(
                        results.predictions,
                        force_all_finite=True,
                        accept_sparse='csr',
                        ensure_2d=False,
                    )
                    #label_binarizer = sklearn.preprocessing.LabelBinarizer()
                    #results.probabilities = label_binarizer.fit(golden).transform(results.predictions)
                    results.probabilities = sklearn.preprocessing.OneHotEncoder(sparse=False).fit(golden.reshape(-1, 1 )).transform(predictions.reshape(-1, 1))
                    print(f"the encoding is therefore = {results.probabilities} {np.shape(results.probabilities)}")
            else:
                raise ValueError(f"Unknown probabilities format: {prob_format}")
        return results

    return run_in_venv(__file__, "exec.py",
                       input_data=data, dataset=dataset, config=config,
                       process_results=process_results)

