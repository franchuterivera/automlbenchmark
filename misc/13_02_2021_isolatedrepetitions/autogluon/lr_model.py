import pandas as pd
import logging
import re

import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from abstract_model import AbstractModel, accuracy

from sklearn.linear_model import LogisticRegression, Ridge, Lasso

from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from rf_model import FeatureMetadata


class OheFeaturesGenerator(BaseEstimator, TransformerMixin):
    missing_category_str = '!missing!'

    def __init__(self, cats_cols):
        self._feature_names = []
        self.cats = cats_cols
        self.ohe_encs = None
        self.labels = None

    def fit(self, X, y=None):
        self.ohe_encs = {f: OneHotEncoder(handle_unknown='ignore') for f in self.cats}
        self.labels = {}

        for c in self.cats:
            self.ohe_encs[c].fit(self._normalize(X[c]))
            self.labels[c] = self.ohe_encs[c].categories_
        return self

    def transform(self, X, y=None):
        Xs = [self.ohe_encs[c].transform(self._normalize(X[c])) for c in self.cats]

        # Update feature names
        self._feature_names = []
        for k, v in self.labels.items():
            for f in k + '_' + v[0]:
                self._feature_names.append(f)

        return hstack(Xs)

    def _normalize(self, col):
        return col.astype(str).fillna(self.missing_category_str).values.reshape(-1, 1)

    def get_feature_names(self):
        return self._feature_names


class NlpDataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, nlp_cols):
        self.nlp_cols = nlp_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.nlp_cols].copy()
        for c in self.nlp_cols:
            X[c] = X[c].astype(str).fillna(' ')
        X = X.apply(' '.join, axis=1).str.replace('[ ]+', ' ', regex=True)
        return X.values.tolist()


class NumericDataPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, cont_cols):
        self.cont_cols = cont_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[self.cont_cols].copy()
        return X.values.tolist()






















L1 = 'L1'
L2 = 'L2'
IGNORE = 'ignore'
ONLY = 'only'
INCLUDE = 'include'

logger = logging.getLogger(__name__)


def get_param_baseline():
    default_params = {
        'C': 1,
        'vectorizer_dict_size': 75000,  # size of TFIDF vectorizer dictionary; used only in text model
        'proc.ngram_range': (1, 5),  # range of n-grams for TFIDF vectorizer dictionary; used only in text model
        'proc.skew_threshold': 0.99,  # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        'proc.impute_strategy': 'median',  # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        'penalty': L2,  # regularization to use with regression models
        'handle_text': IGNORE, # how text should be handled: `ignore` - don't use NLP features; `only` - only use NLP features; `include` - use both regular and NLP features
    }
    return default_params


def get_model_params(problem_type: str, hyperparameters):
    penalty = hyperparameters.get('penalty', L2)
    handle_text = hyperparameters.get('handle_text', IGNORE)
    model_class = LogisticRegression

    return model_class, penalty, handle_text


def get_default_params(problem_type: str, penalty: str):
    # TODO: get seed from seeds provider
    default_params = {'C': None, 'random_state': 0, 'solver': _get_solver(problem_type), 'n_jobs': -1, 'fit_intercept': True}
    model_params = list(default_params.keys())
    return model_params, default_params


def _get_solver(problem_type):
    if problem_type == 'binary':
        # TODO explore using liblinear for smaller datasets
        solver = 'lbfgs'
    else:
        solver = 'lbfgs'
    return solver










# TODO: Can Bagged LinearModels be combined during inference to 1 model by averaging their weights?
#  What about just always using refit_full model? Should we even bag at all? Do we care that its slightly overfit?
class LinearModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_class, self.penalty, self.handle_text = get_model_params(self.problem_type, self.params)
        self.types_of_features = None
        self.pipeline = None

        self.model_params, default_params = get_default_params(self.problem_type, self.penalty)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def tokenize(self, s):
        return re.split('[ ]+', s)

    def _get_types_of_features(self, df):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self.types_of_features is not None:
            print("Attempting to _get_types_of_features for LRModel, but previously already did this.")

        feature_types = self.feature_metadata.get_type_group_map_raw()

        categorical_featnames = feature_types['category'] + feature_types['object'] + feature_types['bool']
        continuous_featnames = feature_types['float'] + feature_types['int']  # + self.__get_feature_type_if_present('datetime')
        language_featnames = []  # TODO: Disabled currently, have to pass raw text data features here to function properly
        valid_features = categorical_featnames + continuous_featnames + language_featnames
        if len(categorical_featnames) + len(continuous_featnames) + len(language_featnames) != df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            df = df.drop(columns=unknown_features)
        self.features = list(df.columns)

        types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'language': []}
        return self._select_features(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    def _select_features(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        features_selector = {
            INCLUDE: self._select_features_handle_text_include,
            ONLY: self._select_features_handle_text_only,
            IGNORE: self._select_features_handle_text_ignore,
        }.get(self.handle_text, self._select_features_handle_text_ignore)
        return features_selector(df, types_of_features, categorical_featnames, language_featnames, continuous_featnames)

    # TODO: handle collinear features - they will impact results quality
    def preprocess(self, X: DataFrame, is_train=False, vect_max_features=1000, model_specific_preprocessing=False):
        X = super().preprocess(X=X)
        if model_specific_preprocessing:  # This is hack to work-around pre-processing caching in bagging/stacker models
            if is_train:
                if self.feature_metadata is None:
                    #raise ValueError("Trainer class must set feature_metadata for this model")
                    self.feature_metadata = FeatureMetadata.from_df(X)
                feature_types = self._get_types_of_features(X)
                X = self.preprocess_train(X, feature_types, vect_max_features)
            else:
                X = self.pipeline.transform(X)
        return X

    def preprocess_train(self, X, feature_types, vect_max_features):
        transformer_list = []
        if len(feature_types['language']) > 0:
            pipeline = Pipeline(steps=[
                ("preparator", NlpDataPreprocessor(nlp_cols=feature_types['language'])),
                ("vectorizer", TfidfVectorizer(ngram_range=self.params['proc.ngram_range'], sublinear_tf=True, max_features=vect_max_features, tokenizer=self.tokenize)),
            ])
            transformer_list.append(('vect', pipeline))
        if len(feature_types['onehot']) > 0:
            pipeline = Pipeline(steps=[
                ('generator', OheFeaturesGenerator(cats_cols=feature_types['onehot'])),
            ])
            transformer_list.append(('cats', pipeline))
        if len(feature_types['continuous']) > 0:
            pipeline = Pipeline(steps=[
                ('generator', NumericDataPreprocessor(cont_cols=feature_types['continuous'])),
                ('imputer', SimpleImputer(strategy=self.params['proc.impute_strategy'])),
                ('scaler', StandardScaler())
            ])
            transformer_list.append(('cont', pipeline))
        if len(feature_types['skewed']) > 0:
            pipeline = Pipeline(steps=[
                ('generator', NumericDataPreprocessor(cont_cols=feature_types['skewed'])),
                ('imputer', SimpleImputer(strategy=self.params['proc.impute_strategy'])),
                ('quantile', QuantileTransformer(output_distribution='normal')),  # Or output_distribution = 'uniform'
            ])
            transformer_list.append(('skew', pipeline))
        self.pipeline = FeatureUnion(transformer_list=transformer_list)
        return self.pipeline.fit_transform(X)

    def _set_default_params(self):
        for param, val in get_param_baseline().items():
            self._set_default_param_value(param, val)

    # TODO: It could be possible to adaptively set max_iter [1] to approximately respect time_limit based on sample-size, feature-dimensionality, and the solver used.
    #  [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#examples-using-sklearn-linear-model-logisticregression
    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, **kwargs):
        hyperparams = self.params.copy()

        if self.problem_type == 'binary':
            y_train = y_train.astype(int).values

        X_train = self.preprocess(X_train, is_train=True, vect_max_features=hyperparams['vectorizer_dict_size'], model_specific_preprocessing=True)

        params = {k: v for k, v in self.params.items() if k in self.model_params}

        # Ridge/Lasso are using alpha instead of C, which is C^-1
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge

        # TODO: copy_X=True currently set during regression problem type, could potentially set to False to avoid unnecessary data copy.
        model = self.model_class(**params)

        logger.log(15, f'Training Model with the following hyperparameter settings:')
        logger.log(15, model)

        self.model = model.fit(X_train, y_train)

    def _predict_proba(self, X, preprocess=True):
        X = self.preprocess(X, is_train=False, model_specific_preprocessing=True)
        #return super()._predict_proba(X, preprocess=False)
        return super()._predict_proba(X)

    def hyperparameter_tune(self, X_train, y_train, X_val, y_val, scheduler_options=None, **kwargs):
        self.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, **kwargs)
        hpo_model_performances = {self.name: self.score(X_val, y_val)}
        hpo_results = {}
        self.save()
        hpo_models = {self.name: self.path}

        return hpo_models, hpo_model_performances, hpo_results

    def get_info(self):
        # TODO: All AG-Tabular models now offer a get_info method:
        # https://github.com/awslabs/autogluon/blob/master/autogluon/utils/tabular/ml/models/abstract/abstract_model.py#L474
        # dict of weights?
        return super().get_info()

    def _select_features_handle_text_include(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 10000  # FIXME research memory constraints
        for feature in self.features:
            feature_data = df[feature]
            num_unique_vals = len(feature_data.unique())
            if feature in language_featnames:
                types_of_features['language'].append(feature)
            elif feature in continuous_featnames:
                if np.abs(feature_data.skew()) > self.params['proc.skew_threshold']:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif (feature in categorical_featnames) and (num_unique_vals <= one_hot_threshold):
                types_of_features['onehot'].append(feature)
        return types_of_features

    def _select_features_handle_text_only(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        for feature in self.features:
            if feature in language_featnames:
                types_of_features['language'].append(feature)
        return types_of_features

    def _select_features_handle_text_ignore(self, df, types_of_features, categorical_featnames, language_featnames, continuous_featnames):
        # continuous = numeric features to rescale
        # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
        # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
        one_hot_threshold = 10000  # FIXME research memory constraints
        for feature in self.features:
            feature_data = df[feature]
            num_unique_vals = len(feature_data.unique())
            if feature in continuous_featnames:
                if np.abs(feature_data.skew()) > self.params['proc.skew_threshold']:
                    types_of_features['skewed'].append(feature)
                else:
                    types_of_features['continuous'].append(feature)
            elif (feature in categorical_featnames) and (num_unique_vals <= one_hot_threshold):
                types_of_features['onehot'].append(feature)
        return types_of_features

if __name__ == '__main__':
    import sklearn.datasets
    from pandas.api.types import is_numeric_dtype
    from sklearn.utils.multiclass import type_of_target
    for task in [40981, 40996]:
        X, y = sklearn.datasets.fetch_openml(data_id=task, return_X_y=True, as_frame=True)
        X = X.convert_dtypes()
        for x in X.columns:
            if not is_numeric_dtype(X[x]):
                X[x] = X[x].astype('category')
            elif 'Int' in str(X[x].dtype):
                X[x] = X[x].astype(np.int)
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y = pd.DataFrame(y)

        problem_type = type_of_target(y)
        print(f"X={X.dtypes} problem={problem_type} {y}")
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
        model = LinearModel(path='/tmp/', problem_type=problem_type, metric=accuracy, name='Linear', eval_metric=accuracy)
        model.fit(X_train=X_train, y_train=y_train)
        print(model.predict_proba(X_test).shape)
        print(accuracy(y_test.to_numpy().astype(int), model.predict(X_test)))
