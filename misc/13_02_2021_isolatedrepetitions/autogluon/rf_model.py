from sklearn.utils.multiclass import type_of_target
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import pandas as pd
from pandas.api.types import is_numeric_dtype
import sklearn.datasets
import logging
import math
import pickle
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from abstract_model import NotEnoughMemoryError, TimeLimitExceeded, AbstractModel, accuracy

import inspect
import copy
from collections import defaultdict
from typing import Dict, List, Tuple

from pandas import DataFrame, Series


# TODO: Consider NaN values as a separate value?
def is_useless_feature(X: Series) -> bool:
    """If a feature has the same value for every row, it carries no useful information"""
    return len(X.unique()) <= 1


def get_type_family_raw(dtype) -> str:
    """From dtype, gets the dtype family."""
    try:
        if dtype.name == 'category':
            return 'category'
        if 'datetime' in dtype.name:
            return 'datetime'
        elif np.issubdtype(dtype, np.integer):
            return 'int'
        elif np.issubdtype(dtype, np.floating):
            return 'float'
    except Exception as err:
        print(err)

    if dtype.name in ['bool', 'bool_']:
        return 'bool'
    elif dtype.name in ['str', 'string', 'object']:
        return 'object'
    else:
        return dtype.name


# Real dtypes
def get_type_map_real(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: v.name for k, v in features_types.items()}


# Raw dtypes (Real dtypes family)
def get_type_map_raw(df: DataFrame) -> dict:
    features_types = df.dtypes.to_dict()
    return {k: get_type_family_raw(v) for k, v in features_types.items()}


def get_type_map_special(X: DataFrame) -> dict:
    type_map_special = {}
    for column in X:
        type_special = get_type_special(X[column])
        if type_special is not None:
            type_map_special[column] = type_special
    return type_map_special


def get_type_special(X: Series) -> str:
    if check_if_datetime_as_object_feature(X):
        type_special = 'datetime_as_object'
    elif check_if_nlp_feature(X):
        type_special = 'text'
    else:
        type_special = None
    return type_special


def get_type_group_map(type_map: dict) -> defaultdict:
    type_group_map = defaultdict(list)
    for key, val in type_map.items():
        type_group_map[val].append(key)
    return type_group_map


def get_type_group_map_real(df: DataFrame) -> defaultdict:
    type_map_real = get_type_map_real(df)
    return get_type_group_map(type_map_real)


def get_type_group_map_raw(df: DataFrame) -> defaultdict:
    type_map_raw = get_type_map_raw(df)
    return get_type_group_map(type_map_raw)


# TODO: Expand to enable multiple special types per feature
def get_type_group_map_special(df: DataFrame) -> defaultdict:
    type_map_special = get_type_map_special(df)
    return get_type_group_map(type_map_special)


# TODO: Expand to int64 -> date features (milli from epoch etc)
def check_if_datetime_as_object_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    # TODO: Check if low numeric numbers, could be categorical encoding!
    # TODO: If low numeric, potentially it is just numeric instead of date
    if X.isnull().all():
        return False
    if type_family != 'object':  # TODO: seconds from epoch support
        return False
    try:
        # TODO: pd.Series(['20170204','20170205','20170206']) is incorrectly not detected as datetime_as_object
        #  But we don't want pd.Series(['184','822828','20170206']) to be detected as datetime_as_object
        #  Need some smart logic (check min/max values?, check last 2 values don't go >31?)
        X.apply(pd.to_numeric)
    except:
        try:
            X.apply(pd.to_datetime)
            return True
        except:
            return False
    else:
        return False


def check_if_nlp_feature(X: Series) -> bool:
    type_family = get_type_family_raw(X.dtype)
    if type_family != 'object':
        return False
    X_unique = X.unique()
    num_unique = len(X_unique)
    num_rows = len(X)
    unique_ratio = num_unique / num_rows
    if unique_ratio <= 0.01:
        return False
    try:
        avg_words = Series(X_unique).str.split().str.len().mean()
    except AttributeError:
        return False
    if avg_words < 3:
        return False

    return True


class FeatureMetadata:
    """
    Feature metadata contains information about features that are not directly apparent in the raw data itself.
    This enables feature generators to properly process features, and allows downstream models to properly handle features during training and inference.
    Parameters
    ----------
    type_map_raw : Dict[str, str]
        Dictionary of feature names to raw types.
        The values can be anything, but it is generally recommended they be one of:
            ['int', 'float', 'object', 'category', 'datetime']
    type_group_map_special : Dict[str, List[str]], optional
        Dictionary of special types to lists of feature names.
        The keys can be anything, but it is generally recommended they be one of:
            ['binned', 'datetime_as_int', 'datetime_as_object', 'text', 'text_as_category', 'text_special', 'text_ngram', 'stack']
        Feature names that appear in the value lists must also be keys in type_map_raw.
        Feature names are not required to have special types.
    """
    def __init__(self, type_map_raw: Dict[str, str], type_group_map_special: Dict[str, List[str]] = None):
        if type_group_map_special is None:
            type_group_map_special = defaultdict(list)
        if not isinstance(type_group_map_special, defaultdict):
            type_group_map_special = defaultdict(list, type_group_map_special)

        self.type_map_raw = type_map_raw
        self.type_group_map_special = type_group_map_special

        self._validate()

    # Confirms if inputs are valid
    def _validate(self):
        type_group_map_special_expanded = []
        for key in self.type_group_map_special:
            type_group_map_special_expanded += self.type_group_map_special[key]

        features_invalid = []
        type_map_raw_keys = self.type_map_raw.keys()
        for feature in type_group_map_special_expanded:
            if feature not in type_map_raw_keys:
                features_invalid.append(feature)
        if features_invalid:
            raise AssertionError(f"{len(features_invalid)} features are present in type_group_map_special but not in type_map_raw. Invalid features: {features_invalid}")

    # Note: This is not optimized for speed. Do not rely on this function during inference.
    # TODO: Add valid_names, invalid_names arguments which override all other arguments for the features listed?
    def get_features(self, valid_raw_types: list = None, valid_special_types: list = None, invalid_raw_types: list = None, invalid_special_types: list = None,
                     required_special_types: list = None, required_raw_special_pairs: List[Tuple[str, List[str]]] = None, required_exact=False, required_at_least_one_special=False):
        """
        Returns a list of features held within the feature metadata object after being pruned through the available parameters.
        Parameters
        ----------
        valid_raw_types : list, default None
            If a feature's raw type is not in this list, it is pruned.
            If None, then no features are pruned through this logic.
        valid_special_types : list, default None
            If a feature has a special type not in this list, it is pruned.
            Features without special types are never pruned through this logic.
            If None, then no features are pruned through this logic.
        invalid_raw_types : list, default None
            If a feature's raw type is in this list, it is pruned.
            If None, then no features are pruned through this logic.
        invalid_special_types : list, default None
            If a feature has a special type in this list, it is pruned.
            Features without special types are never pruned through this logic.
            If None, then no features are pruned through this logic.
        required_special_types : list, default None
            If a feature does not have all of the special types in this list, it is pruned.
            Features without special types are pruned through this logic.
            If None, then no features are pruned through this logic.
        required_raw_special_pairs : List[Tuple[str, List[str]]], default None
            If a feature does not satisfy the (raw_type, special_types) requirement of at least one of the elements in this list, it is pruned.
            Identical to getting the union of calling get_features(valid_raw_types=[raw_type], required_special_types=special_types) for every element of (raw_type, special_types) in required_raw_special_pairs
            If raw_type is None, then any feature will satisfy the raw type requirement.
            If special_types is None, then any feature will satisfy the special type requirement (including those with no special types).
        required_exact : bool, default False
            If True, then if a feature does not have the exact same special types (with no extra special types) as required_special_types, it is pruned.
            This is also applied to required_raw_special_pairs if specified.
            Has no effect if required_special_types and required_raw_special_pairs are None.
        required_at_least_one_special : bool, default False
            If True, then if a feature has zero special types, it is pruned.
        Returns
        -------
        features : list of feature names in feature metadata that satisfy all checks dictated by the parameters.
        """
        features = list(self.type_map_raw.keys())

        if valid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) in valid_raw_types]
        if valid_special_types is not None:
            valid_special_types_set = set(valid_special_types)
            features = [feature for feature in features if not valid_special_types_set.isdisjoint(self.get_feature_types_special(feature)) or not self.get_feature_types_special(feature)]
        if invalid_raw_types is not None:
            features = [feature for feature in features if self.get_feature_type_raw(feature) not in invalid_raw_types]
        if invalid_special_types is not None:
            invalid_special_types_set = set(invalid_special_types)
            features = [feature for feature in features if invalid_special_types_set.isdisjoint(self.get_feature_types_special(feature))]
        if required_special_types is not None:
            required_special_types_set = set(required_special_types)
            if required_exact:
                features = [feature for feature in features if required_special_types_set == set(self.get_feature_types_special(feature))]
            else:
                features = [feature for feature in features if required_special_types_set.issubset(self.get_feature_types_special(feature))]
        if required_at_least_one_special:
            features = [feature for feature in features if self.get_feature_types_special(feature)]
        if required_raw_special_pairs is not None:
            features_og = copy.deepcopy(features)
            features_to_keep = []
            for valid_raw, valid_special in required_raw_special_pairs:
                if valid_special is not None:
                    valid_special = set(valid_special)
                features_to_keep_inner = []
                for feature in features:
                    feature_type_raw = self.get_feature_type_raw(feature)
                    feature_types_special = set(self.get_feature_types_special(feature))
                    if valid_raw is None or feature_type_raw == valid_raw:
                        if valid_special is None:
                            features_to_keep_inner.append(feature)
                        elif required_exact:
                            if valid_special == feature_types_special:
                                features_to_keep_inner.append(feature)
                        elif valid_special.issubset(feature_types_special):
                            features_to_keep_inner.append(feature)
                features = [feature for feature in features if feature not in features_to_keep_inner]
                features_to_keep += features_to_keep_inner
            features = [feature for feature in features_og if feature in features_to_keep]

        return features

    def get_feature_type_raw(self, feature: str) -> str:
        return self.type_map_raw[feature]

    def get_feature_types_special(self, feature: str) -> list:
        if feature not in self.type_map_raw:
            raise KeyError(f'{feature} does not exist in {self.__class__.__name__}.')
        return self._get_feature_types(feature=feature, feature_types_dict=self.type_group_map_special)

    def get_type_group_map_raw(self):
        type_group_map_raw = defaultdict(list)
        for feature, dtype in self.type_map_raw.items():
            type_group_map_raw[dtype].append(feature)
        return type_group_map_raw

    def remove_features(self, features: list, inplace=False):
        """Removes all features from metadata that are in features"""
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(f'remove_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}')
        metadata._remove_features_from_type_map(d=metadata.type_map_raw, features=features)
        metadata._remove_features_from_type_group_map(d=metadata.type_group_map_special, features=features)
        return metadata

    def keep_features(self, features: list, inplace=False):
        """Removes all features from metadata except for those in features"""
        features_invalid = [feature for feature in features if feature not in self.get_features()]
        if features_invalid:
            raise KeyError(f'keep_features was called with a feature that does not exist in feature metadata. Invalid Features: {features_invalid}')
        features_to_remove = [feature for feature in self.get_features() if feature not in features]
        return self.remove_features(features=features_to_remove, inplace=inplace)

    @staticmethod
    def _remove_features_from_type_group_map(d, features):
        for key, features_orig in d.items():
            d[key] = [feature for feature in features_orig if feature not in features]

    @staticmethod
    def _remove_features_from_type_map(d, features):
        for feature in features:
            if feature in d:
                d.pop(feature)

    def rename_features(self, rename_map: dict, inplace=False):
        """Rename all features from metadata that are keys in rename_map to their values."""
        if inplace:
            metadata = self
        else:
            metadata = copy.deepcopy(self)
        before_len = len(metadata.type_map_raw.keys())
        metadata.type_map_raw = {rename_map.get(key, key): val for key, val in metadata.type_map_raw.items()}
        after_len = len(metadata.type_map_raw.keys())
        if before_len != after_len:
            raise AssertionError(f'key names conflicted during renaming. Do not rename features to exist feature names.')
        for dtype in metadata.type_group_map_special:
            metadata.type_group_map_special[dtype] = [rename_map.get(feature, feature) for feature in metadata.type_group_map_special[dtype]]
        return metadata

    # TODO: Add documentation on shared_raw_features usage
    def join_metadata(self, metadata, shared_raw_features='error'):
        """Join two FeatureMetadata objects together, returning a new FeatureMetadata object"""
        if shared_raw_features not in ['error', 'error_if_diff', 'overwrite']:
            raise ValueError(f"shared_raw_features must be one of {['error', 'error_if_diff', 'overwrite']}, but was: '{shared_raw_features}'")
        type_map_raw = copy.deepcopy(self.type_map_raw)
        shared_features = []
        shared_features_diff_types = []
        for key, features in metadata.type_map_raw.items():
            if key in type_map_raw:
                shared_features.append(key)
                if type_map_raw[key] != metadata.type_map_raw[key]:
                    shared_features_diff_types.append(key)
        if shared_features:
            if shared_raw_features == 'error':
                self.print_feature_metadata_full(log_prefix='\t', log_level=40)
                metadata.print_feature_metadata_full(log_prefix='\t', log_level=40)
                raise AssertionError(f"Metadata objects to join share raw features, but `shared_raw_features='error'`. Shared features: {shared_features}")
            if shared_features_diff_types:
                if shared_raw_features == 'overwrite':
                    shared_features = []
                elif shared_raw_features == 'error_if_diff':
                    self.print_feature_metadata_full(log_prefix='\t', log_level=40)
                    metadata.print_feature_metadata_full(log_prefix='\t', log_level=40)
                    raise AssertionError(f"Metadata objects to join share raw features but do not agree on raw dtypes, and `shared_raw_features='error_if_diff'`. Shared conflicting features: {shared_features_diff_types}")
        type_map_raw.update({key: val for key, val in metadata.type_map_raw.items() if key not in shared_features})

        type_group_map_special = self._add_type_group_map_special([self.type_group_map_special, metadata.type_group_map_special])

        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def _add_type_group_map_special(type_group_map_special_lst: List[dict]) -> dict:
        if not type_group_map_special_lst:
            return defaultdict(list)
        type_group_map_special_combined = copy.deepcopy(type_group_map_special_lst[0])
        for type_group_map_special in type_group_map_special_lst[1:]:
            for key, features in type_group_map_special.items():
                if key in type_group_map_special_combined:
                    features_to_add = [feature for feature in features if feature not in type_group_map_special_combined[key]]
                    type_group_map_special_combined[key] += features_to_add
                else:
                    type_group_map_special_combined[key] = features
        return type_group_map_special_combined

    @staticmethod
    def _get_feature_types(feature: str, feature_types_dict: dict) -> list:
        feature_types = []
        for dtype_family in feature_types_dict:
            if feature in feature_types_dict[dtype_family]:
                feature_types.append(dtype_family)
        feature_types = sorted(feature_types)
        return feature_types

    # Joins a list of metadata objects together, returning a new metadata object
    @staticmethod
    def join_metadatas(metadata_list, shared_raw_features='error'):
        metadata_new = copy.deepcopy(metadata_list[0])
        for metadata in metadata_list[1:]:
            metadata_new = metadata_new.join_metadata(metadata, shared_raw_features=shared_raw_features)
        return metadata_new

    def to_dict(self, inverse=False) -> dict:
        if not inverse:
            feature_metadata_dict = dict()
        else:
            feature_metadata_dict = defaultdict(list)

        for feature in self.get_features():
            feature_type_raw = self.type_map_raw[feature]
            feature_types_special = tuple(self.get_feature_types_special(feature))
            if not inverse:
                feature_metadata_dict[feature] = (feature_type_raw, feature_types_special)
            else:
                feature_metadata_dict[(feature_type_raw, feature_types_special)].append(feature)

        if inverse:
            feature_metadata_dict = dict(feature_metadata_dict)

        return feature_metadata_dict

    def print_feature_metadata_full(self, log_prefix='', print_only_one_special=False, log_level=20, max_list_len=5, return_str=False):
        feature_metadata_dict = self.to_dict(inverse=True)
        if not feature_metadata_dict:
            if return_str:
                return ''
            else:
                return
        keys = list(feature_metadata_dict.keys())
        keys = sorted(keys)
        output = [((key[0], list(key[1])), feature_metadata_dict[key]) for key in keys]
        output_str = ''
        if print_only_one_special:
            for i, ((raw, special), features) in enumerate(output):
                if len(special) == 1:
                    output[i] = ((raw, special[0]), features)
                elif len(special) > 1:
                    output[i] = ((raw, special[0]), features)
                else:
                    output[i] = ((raw, None), features)
        max_key_len = max([len(str(key)) for key, _ in output])
        max_val_len = max([len(str(len(val))) for _, val in output])
        for key, val in output:
            key_len = len(str(key))
            val_len = len(str(len(val)))
            max_key_minus_cur = max(max_key_len - key_len, 0)
            max_val_minus_cur = max(max_val_len - val_len, 0)
            if max_list_len is not None:
                features = str(val[:max_list_len])
                if len(val) > max_list_len:
                    features = features[:-1] + ', ...]'
            else:
                features = str(val)
            if val:
                message = f'{log_prefix}{key}{" " * max_key_minus_cur} : {" " * max_val_minus_cur}{len(val)} | {features}'
                if return_str:
                    output_str += message + '\n'
        if return_str:
            if output_str[-1] == '\n':
                output_str = output_str[:-1]
            return output_str

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        Construct FeatureMetadata based on the inferred feature types of an input :class:`pd.DataFrame`.
        Parameters
        ----------
        df : :class:`pd.DataFrame`
            DataFrame used to infer FeatureMetadata.
        Returns
        -------
        :class:`FeatureMetadata` object.
        """
        type_map_raw = get_type_map_raw(df)
        type_group_map_special = get_type_group_map_special(df)
        return cls(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    def __str__(self):
        return self.print_feature_metadata_full(return_str=True)



# TODO: Add option to minimize memory usage of feature names by making them integers / strings of integers
# TODO: Add ability to track which input features created which output features.
# TODO: Add log of # of observation counts to high cardinality categorical features
class AbstractFeatureGenerator:
    """
    Abstract feature generator implementation from which all AutoGluon feature generators inherit.
    The purpose of a feature generator is to transform data from one form to another in a stateful manner.
    First, the generator is initialized with various arguments that dictate the way features are generated.
    Then, the generator is fit through either the `.fit()` or `.fit_transform()` methods using training data typically in pandas DataFrame format.
    Finally, the generator can transform new data with the same initial format as the training data through the `.transform()` method.
    Parameters
    ----------
    features_in : list, default None
        List of feature names the generator will expect and use in the fit and transform methods.
        Any feature in an incoming DataFrame that is not present in features_in is dropped and will not influence the transformation logic.
        If None, infer during fit from the _infer_features_in method.
        Equivalent to feature_metadata_in.get_features() post-fit.
    feature_metadata_in : FeatureMetadata, default None
        FeatureMetadata object corresponding to the training data input features.
        If None, infer during fit from the _infer_feature_metadata_in method.
        Any features not present in features_in (if provided) will be removed from feature_metadata_in.
    post_generators : list of FeatureGenerators, default None
        FeatureGenerators which will fit and transform sequentially after this object's transformation logic, feeding their output into the next generator's input.
        The output of the final FeatureGenerator will be the used as the transformed output.
    pre_enforce_types : bool, default False
        If True, the exact raw types (int64, float32, etc.) of the training data will be enforced on future data, either converting the types to the training types or raising an exception if unable.
        This is important to set to True on the outer feature generator in a feature generation pipeline to ensure incorrect dtypes are not passed downstream, but is often redundant when used on inner feature generators inside a pipeline.
    pre_drop_useless : bool, default False
        If True, features_in will be pruned at fit time of features containing only a single unique value across all rows.
    post_drop_duplicates : bool, default False
        If True, a DropDuplicatesFeatureGenerator will be appended to post_generators.
        This feature generator will drop any duplicate features found in the data, keeping only one feature within any duplicate feature sets.
        Warning: For large datasets with many features, this may be very computationally expensive or even computationally infeasible.
    reset_index : bool, default False
        If True, for the duration of fit and transform, the input data's index is reset to be monotonically increasing from 0 to N-1 for a dataset of N rows.
        At the end of fit and transform, the original index is re-applied to the output data.
        This is important to set to True on the outer feature generator in a feature generation pipeline to ensure that a non-default index does not cause corruption of the inner feature generation if any inner feature generator does not properly handle non-default indices.
        This index reset is also applied to the y label data if provided during fit.
    column_names_as_str : bool, default True
        If True, the column names of the input data are converted to string if they were not already.
        This solves any issues related to downstream FeatureGenerators and models which cannot handle integer column names, and allows column name prefix and suffix operations to avoid errors.
        Note that for performance purposes, column names are only converted at transform time if they were not strings at fit time. Ensure consistent column names as input to avoid errors.
    name_prefix : str, default None
        Name prefix to add to all output feature names.
    name_suffix : str, default None
        Name suffix to add to all output feature names.
    infer_features_in_args : dict, default None
        Used as the kwargs input to FeatureMetadata.get_features(**kwargs) when inferring self.features_in.
        This is merged with the output dictionary of self.get_default_infer_features_in_args() depending on the value of infer_features_in_args_strategy.
        Only used when features_in is None.
        If None, then self.get_default_infer_features_in_args() is used directly.
        Refer to FeatureMetadata.get_features documentation for a full description of valid keys.
        Note: This is advanced functionality that is not necessary for most situations.
    infer_features_in_args_strategy : str, default 'overwrite'
        Determines how infer_features_in_args and self.get_default_infer_features_in_args() are combined to result in self._infer_features_in_args which dictates the features_in inference logic.
        If 'overwrite': infer_features_in_args is used exclusively and self.get_default_infer_features_in_args() is ignored.
        If 'update': self.get_default_infer_features_in_args() is dictionary updated by infer_features_in_args.
        If infer_features_in_args is None, this is ignored.
    log_prefix : str, default ''
        Prefix string added to all logging statements made by the generator.
    verbosity : int, default 2
        Controls the verbosity of logging.
        0 will silence logs, 1 will only log warnings, 2 will log info level information, and 3 will log info level information and provide detailed feature type input and output information.
        Logging is still controlled by the global logger configuration, and therefore a verbosity of 3 does not guarantee that logs will be output.
    Attributes
    ----------
    features_in : list of str
        List of feature names the generator will expect and use in the fit and transform methods.
        Equivalent to feature_metadata_in.get_features() post-fit.
    features_out : list of str
        List of feature names present in the output of fit_transform and transform methods.
        Equivalent to feature_metadata.get_features() post-fit.
    feature_metadata_in : FeatureMetadata
        The FeatureMetadata of data pre-transformation (data used as input to fit and transform methods).
    feature_metadata : FeatureMetadata
        The FeatureMetadata of data post-transformation (data outputted by fit_transform and transform methods).
    feature_metadata_real : FeatureMetadata
        The FeatureMetadata of data post-transformation consisting of the exact dtypes as opposed to the grouped raw dtypes found in feature_metadata_in, with grouped raw dtypes substituting for the special dtypes.
        This is only used in the print_feature_metadata_info method and is intended for introspection. It can be safely set to None to reduce memory and disk usage post-fit.
    """
    def __init__(
        self,
        features_in: list = None,
        feature_metadata_in: FeatureMetadata = None,
        post_generators: list = None,
        pre_enforce_types=False,
        pre_drop_useless=False,
        post_drop_duplicates=False,
        reset_index=False,
        column_names_as_str=True,
        name_prefix: str = None,
        name_suffix: str = None,
        infer_features_in_args: dict = None,
        infer_features_in_args_strategy='overwrite',
        log_prefix='',
        verbosity=2
    ):
        self._is_fit = False  # Whether the feature generator has been fit
        self.features_in = features_in  # Original features to use as input to feature generation
        self.features_out = None  # Final list of features after transformation
        self.feature_metadata_in: FeatureMetadata = feature_metadata_in  # FeatureMetadata object based on the original input features.
        self.feature_metadata: FeatureMetadata = None  # FeatureMetadata object based on the processed features. Pass to models to enable advanced functionality.
        # TODO: Consider merging feature_metadata and feature_metadata_real, have FeatureMetadata contain exact dtypes, grouped raw dtypes, and special dtypes all at once.
        self.feature_metadata_real: FeatureMetadata = None  # FeatureMetadata object based on the processed features, containing the true raw dtype information (such as int32, float64, etc.). Pass to models to enable advanced functionality.
        self._feature_metadata_before_post = None  # FeatureMetadata directly prior to applying self._post_generators.
        self._infer_features_in_args = self.get_default_infer_features_in_args()
        if infer_features_in_args is not None:
            if infer_features_in_args_strategy == 'overwrite':
                self._infer_features_in_args = copy.deepcopy(infer_features_in_args)
            elif infer_features_in_args_strategy == 'update':
                self._infer_features_in_args.update(infer_features_in_args)
            else:
                raise ValueError(f"infer_features_in_args_strategy must be one of: {['overwrite', 'update']}, but was: '{infer_features_in_args_strategy}'")

        if post_generators is None:
            post_generators = []
        elif not isinstance(post_generators, list):
            post_generators = [post_generators]
        self._post_generators: list = post_generators
        if post_drop_duplicates:
            from .drop_duplicates import DropDuplicatesFeatureGenerator
            self._post_generators.append(DropDuplicatesFeatureGenerator(post_drop_duplicates=False))
        if name_prefix or name_suffix:
            from .rename import RenameFeatureGenerator
            self._post_generators.append(RenameFeatureGenerator(name_prefix=name_prefix, name_suffix=name_suffix, inplace=True))

        if self._post_generators:
            if not self.get_tags().get('allow_post_generators', True):
                raise AssertionError(f'{self.__class__.__name__} is not allowed to have post_generators, but found: {[generator.__class__.__name__ for generator in self._post_generators]}')

        self.pre_enforce_types = pre_enforce_types
        self._pre_astype_generator = None
        self.pre_drop_useless = pre_drop_useless
        self.reset_index = reset_index
        self.column_names_as_str = column_names_as_str
        self._useless_features_in: list = None

        self._is_updated_name = False  # If feature names have been altered by name_prefix or name_suffix

        self.log_prefix = log_prefix
        self.verbosity = verbosity

        self.fit_time = None

    def fit(self, X: DataFrame, **kwargs):
        """
        Fit generator to the provided data.
        Because of how the generators track output features and types, it is generally required that the data be transformed during fit, so the fit function is rarely useful to implement beyond a simple call to fit_transform.
        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        **kwargs
            Any additional arguments that a particular generator implementation could use.
            See fit_transform method for common kwargs values.
        """
        self.fit_transform(X, **kwargs)

    def fit_transform(self, X: DataFrame, y: Series = None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        """
        Fit generator to the provided data and return the transformed version of the data as if fit and transform were called sequentially with the same data.
        This is generally more efficient than calling fit and transform separately and can be up to twice as fast if the fit process requires transformation of the data.
        This cannot be called after the generator has been fit, and will result in an AssertionError.
        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        y : Series, optional
            Input data's labels used to fit the generator. Most generators do not utilize labels.
            y.index must be equal to X.index to avoid misalignment.
        feature_metadata_in : FeatureMetadata, optional
            Identical to providing feature_metadata_in during generator initialization. Ignored if self.feature_metadata_in is already specified.
            If neither are set, feature_metadata_in will be inferred from the _infer_feature_metadata_in method.
        **kwargs
            Any additional arguments that a particular generator implementation could use. Passed to _fit_transform and _fit_generators methods.
        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        start_time = time.time()
        self._log(20, f'Fitting {self.__class__.__name__}...')
        if self._is_fit:
            raise AssertionError(f'{self.__class__.__name__} is already fit.')
        self._pre_fit_validate(X=X, y=y, feature_metadata_in=feature_metadata_in, **kwargs)

        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            X = X.reset_index(drop=True)  # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
            if y is not None and isinstance(y, Series):
                y = y.reset_index(drop=True)  # TODO: this assumes y and X had matching indices prior
        else:
            X_index = None
        if self.column_names_as_str:
            columns_orig = list(X.columns)
            X.columns = X.columns.astype(str)  # Ensure all column names are strings
            columns_new = list(X.columns)
            if columns_orig != columns_new:
                rename_map = {orig: new for orig, new in zip(columns_orig, columns_new)}
                if feature_metadata_in is not None:
                    feature_metadata_in.rename_features(rename_map=rename_map)
                self._rename_features_in(rename_map)
            else:
                self.column_names_as_str = False  # Columns were already string, so don't do conversion. Better to error if they change types at inference.
        self._ensure_no_duplicate_column_names(X=X)
        self._infer_features_in_full(X=X, feature_metadata_in=feature_metadata_in)
        if self.pre_drop_useless:
            self._useless_features_in = self._get_useless_features(X)
            if self._useless_features_in:
                self._remove_features_in(self._useless_features_in)
        if self.pre_enforce_types:
            from .astype import AsTypeFeatureGenerator
            self._pre_astype_generator = AsTypeFeatureGenerator(features_in=self.features_in, feature_metadata_in=self.feature_metadata_in, log_prefix=self.log_prefix + '\t')
            self._pre_astype_generator.fit(X)

        # TODO: Add option to return feature_metadata instead to avoid data copy
        #  If so, consider adding validation step to check that X_out matches the feature metadata, error/warning if not
        X_out, type_family_groups_special = self._fit_transform(X[self.features_in], y=y, **kwargs)

        type_map_raw = get_type_map_raw(X_out)
        self._feature_metadata_before_post = FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_family_groups_special)
        if self._post_generators:
            X_out, self.feature_metadata, self._post_generators = self._fit_generators(X=X_out, y=y, feature_metadata=self._feature_metadata_before_post, generators=self._post_generators, **kwargs)
        else:
            self.feature_metadata = self._feature_metadata_before_post
        type_map_real = get_type_map_real(X_out)
        self.features_out = list(X_out.columns)
        self.feature_metadata_real = FeatureMetadata(type_map_raw=type_map_real, type_group_map_special=self.feature_metadata.get_type_group_map_raw())

        self._post_fit_cleanup()
        if self.reset_index:
            X_out.index = X_index
        self._is_fit = True
        end_time = time.time()
        self.fit_time = end_time - start_time
        if self.verbosity >= 3:
            self.print_feature_metadata_info(log_level=20)
            self.print_generator_info(log_level=20)
        elif self.verbosity == 2:
            self.print_feature_metadata_info(log_level=15)
            self.print_generator_info(log_level=15)
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transforms input data into the output data format.
        Will raise an AssertionError if called before the generator has been fit using fit or fit_transform methods.
        Parameters
        ----------
        X : DataFrame
            Input data to be transformed by the generator.
            Input data must contain all features in features_in, and should have the same dtypes as in the data provided to fit.
            Extra columns present in X that are not in features_in will be ignored and not affect the output.
        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        if not self._is_fit:
            raise AssertionError(f'{self.__class__.__name__} is not fit.')
        if self.reset_index:
            X_index = copy.deepcopy(X.index)
            X = X.reset_index(drop=True)  # TODO: Theoretically inplace=True avoids data copy, but can lead to altering of original DataFrame outside of method context.
        else:
            X_index = None
        if self.column_names_as_str:
            X.columns = X.columns.astype(str)  # Ensure all column names are strings
        try:
            X = X[self.features_in]
        except KeyError:
            missing_cols = []
            for col in self.features_in:
                if col not in X.columns:
                    missing_cols.append(col)
            raise KeyError(f'{len(missing_cols)} required columns are missing from the provided dataset to transform using {self.__class__.__name__}. Missing columns: {missing_cols}')
        if self._pre_astype_generator:
            X = self._pre_astype_generator.transform(X)
        X_out = self._transform(X)
        if self._post_generators:
            X_out = self._transform_generators(X=X_out, generators=self._post_generators)
        if self.reset_index:
            X_out.index = X_index
        return X_out

    def _fit_transform(self, X: DataFrame, y: Series, **kwargs) -> (DataFrame, dict):
        """
        Performs the inner fit_transform logic that is non-generic (specific to the generator implementation).
        When creating a new generator class, this should be implemented.
        At the point this method is called, self.features_in and self.features_metadata_in will be set, and can be accessed and altered freely.
        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
            This data will have already been limited to only the columns present in self.features_in.
            This data may have been altered by the fit_transform method prior to entering _fit_transform in a variety of ways, but self.features_in and self.features_metadata_in will correctly correspond to X at this point in the generator's fit process.
        y : Series, optional
            Input data's labels used to fit the generator. Most generators do not utilize labels.
            y.index is always equal to X.index.
        **kwargs
            Any additional arguments that a particular generator implementation could use. Received from the fit_transform method.
        Returns
        -------
        (X_out : DataFrame, type_group_map_special : dict)
            X_out is the transformed version of the input data X
            type_group_map_special is the type_group_map_special value of X_out's intended FeatureMetadata object.
                If special types are not relevant to the generator, this can simply be dict()
                If the input and output features are identical in name and type, it may be valid to return self.feature_metadata_in.type_group_map_special to maintain any pre-existing special type information.
                Refer to existing generator implementations for guidance on setting the dict output of _fit_transform.
        """
        raise NotImplementedError

    def _transform(self, X: DataFrame) -> DataFrame:
        """
        Performs the inner transform logic that is non-generic (specific to the generator implementation).
        When creating a new generator class, this should be implemented.
        At the point this method is called, self.features_in and self.features_metadata_in will be set, and can be accessed freely.
        Parameters
        ----------
        X : DataFrame
            Input data to be transformed by the generator.
            This data will have already been limited to only the columns present in self.features_in.
            This data may have been altered by the transform method prior to entering _transform in a variety of ways, but self.features_in and self.features_metadata_in will correctly correspond to X at this point in the generator's transform process.
        Returns
        -------
        X_out : DataFrame object which is the transformed version of the input data X.
        """
        raise NotImplementedError

    def _infer_features_in_full(self, X: DataFrame, feature_metadata_in: FeatureMetadata = None):
        """
        Infers all input related feature information of X.
        This can be extended when additional input information is desired beyond feature_metadata_in and features_in.
            For example, AsTypeFeatureGenerator extends this method to also compute the exact raw feature types of the input for later use.
        After this method returns, self.features_in and self.feature_metadata_in will be set to proper values.
        This method is called by fit_transform prior to calling _fit_transform.
        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        feature_metadata_in : FeatureMetadata, optional
            If passed, then self.feature_metadata_in will be set to feature_metadata_in assuming self.feature_metadata_in was None prior.
            If both are None, then self.feature_metadata_in is inferred through _infer_feature_metadata_in(X)
        """
        if self.feature_metadata_in is None:
            self.feature_metadata_in = feature_metadata_in
        elif feature_metadata_in is not None:
            self._log(30, '\tWarning: feature_metadata_in passed as input to fit_transform, but self.feature_metadata_in was already set. Ignoring feature_metadata_in.')
        if self.feature_metadata_in is None:
            self._log(20, f'\tInferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.')
            self.feature_metadata_in = self._infer_feature_metadata_in(X=X)
        if self.features_in is None:
            self.features_in = self._infer_features_in(X=X)
            self.features_in = [feature for feature in self.features_in if feature in X.columns]
        self.feature_metadata_in = self.feature_metadata_in.keep_features(features=self.features_in)

    # TODO: Find way to increase flexibility here, possibly through init args
    def _infer_features_in(self, X: DataFrame) -> list:
        """
        Infers the features_in of X.
        This is used if features_in was not provided by the user prior to fit.
        This can be overwritten in a new generator to use new infer logic.
        self.feature_metadata_in is available at the time this method is called.
        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        Returns
        -------
        feature_in : list of str feature names inferred from X.
        """
        return self.feature_metadata_in.get_features(**self._infer_features_in_args)

    # TODO: Use code from problem type detection for column types. Ints/Floats could be Categorical through this method. Maybe try both?
    @staticmethod
    def _infer_feature_metadata_in(X: DataFrame) -> FeatureMetadata:
        """
        Infers the feature_metadata_in of X.
        This is used if feature_metadata_in was not provided by the user prior to fit.
        This can be overwritten in a new generator to use new infer logic, but it is preferred to keep the default logic for consistency with other generators.
        Parameters
        ----------
        X : DataFrame
            Input data used to fit the generator.
        Returns
        -------
        feature_metadata_in : FeatureMetadata object inferred from X.
        """
        type_map_raw = get_type_map_raw(X)
        type_group_map_special = get_type_group_map_special(X)
        return FeatureMetadata(type_map_raw=type_map_raw, type_group_map_special=type_group_map_special)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        raise NotImplementedError

    def _fit_generators(self, X, y, feature_metadata, generators: list, **kwargs) -> (DataFrame, FeatureMetadata, list):
        """
        Fit a list of AbstractFeatureGenerator objects in sequence, with the output of generators[i] fed as the input to generators[i+1]
        This is called to sequentially fit self._post_generators generators on the output of _fit_transform to obtain the final output of the generator.
        This should not be overwritten by implementations of AbstractFeatureGenerator.
        """
        for generator in generators:
            generator.verbosity = min(self.verbosity, generator.verbosity)
            generator.set_log_prefix(log_prefix=self.log_prefix + '\t', prepend=True)
            X = generator.fit_transform(X=X, y=y, feature_metadata_in=feature_metadata, **kwargs)
            feature_metadata = generator.feature_metadata
        return X, feature_metadata, generators

    @staticmethod
    def _transform_generators(X, generators: list) -> DataFrame:
        """
        Transforms X through a list of AbstractFeatureGenerator objects in sequence, with the output of generators[i] fed as the input to generators[i+1]
        This is called to sequentially transform self._post_generators generators on the output of _transform to obtain the final output of the generator.
        This should not be overwritten by implementations of AbstractFeatureGenerator.
        """
        for generator in generators:
            X = generator.transform(X=X)
        return X

    def _remove_features_in(self, features: list):
        """
        Removes features from all relevant objects which represent the content of the input data or how the input features are used.
        For example, DropDuplicatesFeatureGenerator calls this method during _fit_transform with the list of duplicate features.
            This allows DropDuplicatesFeatureGenerator's _transform method to simply return X, as the duplicate features are already dropped in the transform method due to not being in self.features_in.
        Parameters
        ----------
        features : list of str
            List of feature names to remove from the expected input.
        """
        if features:
            if self._feature_metadata_before_post:
                feature_links_chain = self.get_feature_links_chain()
                for feature in features:
                    feature_links_chain[0].pop(feature)
                features_to_keep = set()
                for features_out in feature_links_chain[0].values():
                    features_to_keep = features_to_keep.union(features_out)
                self._feature_metadata_before_post = self._feature_metadata_before_post.keep_features(features_to_keep)

            self.feature_metadata_in = self.feature_metadata_in.remove_features(features=features)
            self.features_in = self.feature_metadata_in.get_features()
            if self._pre_astype_generator:
                self._pre_astype_generator._remove_features_out(features)

    # TODO: Ensure arbitrary feature removal does not result in inconsistencies (add unit test)
    def _remove_features_out(self, features: list):
        """
        Removes features from the output data.
        This is used for cleaning complex pipelines of unnecessary operations after fitting a sequence of generators.
        Implementations of AbstractFeatureGenerator should not need to alter this method.
        Parameters
        ----------
        features : list of str
            List of feature names to remove from the output of self.transform().
        """
        feature_links_chain = self.get_feature_links_chain()
        if features:
            self.feature_metadata = self.feature_metadata.remove_features(features=features)
            self.feature_metadata_real = self.feature_metadata_real.remove_features(features=features)
            self.features_out = self.feature_metadata.get_features()
            feature_links_chain[-1] = {feature_in: [feature_out for feature_out in features_out if feature_out not in features] for feature_in, features_out in feature_links_chain[-1].items()}
        self._remove_unused_features(feature_links_chain=feature_links_chain)

    def _remove_unused_features(self, feature_links_chain):
        unused_features = self._get_unused_features(feature_links_chain=feature_links_chain)
        self._remove_features_in(features=unused_features[0])
        for i, generator in enumerate(self._post_generators):
            for feature in unused_features[i + 1]:
                if feature in feature_links_chain[i + 1]:
                    feature_links_chain[i + 1].pop(feature)
            generated_features = set()
            for feature_in in feature_links_chain[i + 1]:
                generated_features = generated_features.union(feature_links_chain[i + 1][feature_in])
            features_out_to_remove = [feature for feature in generator.features_out if feature not in generated_features]
            generator._remove_features_out(features_out_to_remove)

    def _rename_features_in(self, column_rename_map: dict):
        if self.feature_metadata_in is not None:
            self.feature_metadata_in = self.feature_metadata_in.rename_features(column_rename_map)
        if self.features_in is not None:
            self.features_in = [column_rename_map.get(col, col) for col in self.features_in]

    def _pre_fit_validate(self, X: DataFrame, y: Series, **kwargs):
        """
        Any data validation checks prior to fitting the data should be done here.
        """
        if y is not None and isinstance(y, Series):
            if list(y.index) != list(X.index):
                raise AssertionError(f'y.index and X.index must be equal when fitting {self.__class__.__name__}, but they differ.')

    def _post_fit_cleanup(self):
        """
        Any cleanup operations after all metadata objects have been constructed, but prior to feature renaming, should be done here.
        This includes removing keys from internal lists and dictionaries of features which have been removed, and deletion of any temp variables.
        """
        pass

    def _ensure_no_duplicate_column_names(self, X: DataFrame):
        if len(X.columns) != len(set(X.columns)):
            count_dict = defaultdict(int)
            invalid_columns = []
            for column in list(X.columns):
                count_dict[column] += 1
            for column in count_dict:
                if count_dict[column] > 1:
                    invalid_columns.append(column)
            raise AssertionError(f'Columns appear multiple times in X. Columns must be unique. Invalid columns: {invalid_columns}')

    # TODO: Move to a generator
    @staticmethod
    def _get_useless_features(X: DataFrame) -> list:
        useless_features = []
        for column in X:
            if is_useless_feature(X[column]):
                useless_features.append(column)
        return useless_features

    # TODO: Consider adding _log and verbosity methods to mixin
    def set_log_prefix(self, log_prefix, prepend=False):
        if prepend:
            self.log_prefix = log_prefix + self.log_prefix
        else:
            self.log_prefix = log_prefix

    def set_verbosity(self, verbosity: int):
        self.verbosity = verbosity

    def _log(self, level, msg, log_prefix=None, verb_min=None):
        if self.verbosity == 0:
            return
        if verb_min is None or self.verbosity >= verb_min:
            if log_prefix is None:
                log_prefix = self.log_prefix

    def is_fit(self):
        return self._is_fit

    # TODO: Handle cases where self.features_in or self.feature_metadata_in was already set at init.
    def is_valid_metadata_in(self, feature_metadata_in: FeatureMetadata):
        """
        True if input data with feature metadata of feature_metadata_in could result in non-empty output.
            This is dictated by `feature_metadata_in.get_features(**self._infer_features_in_args)` not being empty.
        False if the features represented in feature_metadata_in do not contain any usable types for the generator.
            For example, if only numeric features are passed as input to TextSpecialFeatureGenerator which requires text input features, this will return False.
            However, if both numeric and text features are passed, this will return True since the text features would be valid input (the numeric features would simply be dropped).
        """
        features_in = feature_metadata_in.get_features(**self._infer_features_in_args)
        if features_in:
            return True
        else:
            return False

    def get_feature_links(self) -> Dict[str, List[str]]:
        """Returns feature links including all pre and post generators."""
        return self._get_feature_links_from_chain(self.get_feature_links_chain())

    def _get_feature_links(self, features_in: List[str], features_out: List[str]) -> Dict[str, List[str]]:
        """Returns feature links ignoring all pre and post generators."""
        feature_links = {}
        if self.get_tags().get('feature_interactions', True):
            for feature_in in features_in:
                feature_links[feature_in] = features_out
        else:
            for feat_old, feat_new in zip(features_in, features_out):
                feature_links[feat_old] = feature_links.get(feat_old, []) + [feat_new]
        return feature_links

    def get_feature_links_chain(self) -> List[Dict[str, List[str]]]:
        """Get the feature dependence chain between this generator and all of its post generators."""
        features_out_internal = self._feature_metadata_before_post.get_features()

        generators = [self] + self._post_generators
        features_in_list = [self.features_in] + [generator.features_in for generator in self._post_generators]
        features_out_list = [features_out_internal] + [generator.features_out for generator in self._post_generators]

        feature_links_chain = []
        for i in range(len(features_in_list)):
            generator = generators[i]
            features_in = features_in_list[i]
            features_out = features_out_list[i]
            feature_chain = generator._get_feature_links(features_in=features_in, features_out=features_out)
            feature_links_chain.append(feature_chain)
        return feature_links_chain

    @staticmethod
    def _get_feature_links_from_chain(feature_links_chain: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
        """Get the final input and output feature links by travelling the feature link chain"""
        features_out = []
        for val in feature_links_chain[-1].values():
            if val not in features_out:
                features_out.append(val)
        features_in = list(feature_links_chain[0].keys())
        feature_links = feature_links_chain[0]
        for i in range(1, len(feature_links_chain)):
            feature_links_new = {}
            for feature in features_in:
                feature_links_new[feature] = set()
                for feature_out in feature_links[feature]:
                    feature_links_new[feature] = feature_links_new[feature].union(feature_links_chain[i].get(feature_out, []))
                feature_links_new[feature] = list(feature_links_new[feature])
            feature_links = feature_links_new
        return feature_links

    def _get_unused_features(self, feature_links_chain: List[Dict[str, List[str]]]):
        features_in_list = [self.features_in]
        if self._post_generators:
            for i in range(len(self._post_generators)):
                if i == 0:
                    features_in = self._feature_metadata_before_post.get_features()
                else:
                    features_in = self._post_generators[i-1].features_out
                features_in_list.append(features_in)
        return self._get_unused_features_generic(feature_links_chain=feature_links_chain, features_in_list=features_in_list)

    # TODO: Unit test this
    @staticmethod
    def _get_unused_features_generic(feature_links_chain: List[Dict[str, List[str]]], features_in_list: List[List[str]]) -> List[List[str]]:
        unused_features = []
        unused_features_by_stage = []
        for i, chain in enumerate(reversed(feature_links_chain)):
            stage = len(feature_links_chain) - i
            used_features = set()
            for key in chain.keys():
                new_val = [val for val in chain[key] if val not in unused_features]
                if new_val:
                    used_features.add(key)
            features_in = features_in_list[stage - 1]
            unused_features = []
            for feature in features_in:
                if feature not in used_features:
                    unused_features.append(feature)
            unused_features_by_stage.append(unused_features)
        unused_features_by_stage = list(reversed(unused_features_by_stage))
        return unused_features_by_stage

    def print_generator_info(self, log_level: int = 20):
        """
        Outputs detailed logs of the generator, such as the fit runtime.
        Parameters
        ----------
        log_level : int, default 20
            Log level of the logging statements.
        """
        if self.fit_time:
            self._log(log_level, f'\t{round(self.fit_time, 1)}s = Fit runtime')
            self._log(log_level, f'\t{len(self.features_in)} features in original data used to generate {len(self.features_out)} features in processed data.')

    def print_feature_metadata_info(self, log_level: int = 20):
        """
        Outputs detailed logs of a fit feature generator including the input and output FeatureMetadata objects' feature types.
        Parameters
        ----------
        log_level : int, default 20
            Log level of the logging statements.
        """
        self._log(log_level, '\tTypes of features in original data (raw dtype, special dtypes):')
        self.feature_metadata_in.print_feature_metadata_full(self.log_prefix + '\t\t', log_level=log_level)
        if self.feature_metadata_real:
            self._log(log_level-5, '\tTypes of features in processed data (exact raw dtype, raw dtype):')
            self.feature_metadata_real.print_feature_metadata_full(self.log_prefix + '\t\t', print_only_one_special=True, log_level=log_level-5)
        self._log(log_level, '\tTypes of features in processed data (raw dtype, special dtypes):')
        self.feature_metadata.print_feature_metadata_full(self.log_prefix + '\t\t', log_level=log_level)

    def _more_tags(self) -> dict:
        """
        Special values to enable advanced functionality.
        Tags
        ----
        feature_interactions : bool, default True
            If True, then treat all features_out as if they depend on all features_in.
            If False, then treat each features_out as if it was generated by a 1:1 mapping (no feature interactions).
                This enables advanced functionality regarding automated feature pruning, but is only valid for generators which only transform each feature and do not perform interactions.
        allow_post_generators : bool, default True
            If False, will raise an AssertionError if post_generators is specified during init.
                This is reserved for very simple generators where including post_generators would not be sensible, such as in RenameFeatureGenerator.
        """
        return {}

    def get_tags(self) -> dict:
        """Gets the tags for this generator."""
        collected_tags = {}
        for base_class in reversed(inspect.getmro(self.__class__)):
            if hasattr(base_class, '_more_tags'):
                # need the if because mixins might not have _more_tags
                # but might do redundant work in estimators
                # (i.e. calling more tags on BaseEstimator multiple times)
                more_tags = base_class._more_tags(self)
                collected_tags.update(more_tags)
        return collected_tags


# TODO: LabelEncoderTransformer
class LabelEncoderFeatureGenerator(AbstractFeatureGenerator):
    """Converts category features to int features by mapping to the category codes."""
    def _fit_transform(self, X: DataFrame, **kwargs) -> (DataFrame, dict):
        X_out = self._transform(X)
        feature_metadata_out_type_group_map_special = copy.deepcopy(self.feature_metadata_in.type_group_map_special)
        if 'text_as_category' in feature_metadata_out_type_group_map_special:
            feature_metadata_out_type_group_map_special.pop('text_as_category')
        return X_out, feature_metadata_out_type_group_map_special

    def _transform(self, X: DataFrame) -> DataFrame:
        return self.convert_category_to_int(X)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return dict(valid_raw_types=['category'])

    @staticmethod
    def convert_category_to_int(X: DataFrame) -> DataFrame:
        # TODO: add inplace option?
        X = X.apply(lambda x: x.cat.codes)
        return X

    def _more_tags(self):
        return {'feature_interactions': False}


class RFModel(AbstractModel):
    """
    Random Forest model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model_type = self._get_model_type()
        self._feature_generator = None

    def _get_model_type(self):
        return RandomForestClassifier

    # TODO: X.fillna -inf? Add extra is_missing column?
    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        if self._feature_generator is None:
            self._feature_generator = LabelEncoderFeatureGenerator(verbosity=0)
            self._feature_generator.fit(X=X)
        if self._feature_generator.features_in:
            X = X.copy()
            X[self._feature_generator.features_in] = self._feature_generator.transform(X=X)
        X = X.fillna(0).to_numpy(dtype=np.float32)
        return X

    def _set_default_params(self):
        default_params = {
            'n_estimators': 300,
            'n_jobs': -1,
            'random_state': 0,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    # TODO: Add in documentation that Categorical default is the first index
    # TODO: enable HPO for RF models
    def _get_default_searchspace(self):
        spaces = {
            # 'n_estimators': Int(lower=10, upper=1000, default=300),
            # 'max_features': Categorical(['auto', 0.5, 0.25]),
            # 'criterion': Categorical(['gini', 'entropy']),
        }
        return spaces

    def _fit(self, X_train, y_train, time_limit=None, **kwargs):
        time_start = time.time()
        max_memory_usage_ratio = self.params_aux['max_memory_usage_ratio']
        hyperparams = self.params.copy()
        n_estimators_final = hyperparams['n_estimators']

        n_estimators_minimum = min(40, n_estimators_final)
        n_estimators_test = min(4, max(1, math.floor(n_estimators_minimum/5)))

        X_train = self.preprocess(X_train)
        n_estimator_increments = [n_estimators_final]

        # Very rough guess to size of a single tree before training
        if self.problem_type == 'multiclass':
            if self.num_classes is None:
                num_trees_per_estimator = 10  # Guess since it wasn't passed in, could also check y_train for a better value
            else:
                num_trees_per_estimator = self.num_classes
        else:
            num_trees_per_estimator = 1
        bytes_per_estimator = num_trees_per_estimator * len(X_train) / 60000 * 1e6  # Underestimates by 3x on ExtraTrees
        available_mem = 4469755084
        expected_memory_usage = bytes_per_estimator * n_estimators_final / available_mem
        expected_min_memory_usage = bytes_per_estimator * n_estimators_minimum / available_mem
        if expected_min_memory_usage > (0.5 * max_memory_usage_ratio):  # if minimum estimated size is greater than 50% memory
            raise NotEnoughMemoryError

        if n_estimators_final > n_estimators_test * 2:
            if self.problem_type == 'multiclass':
                n_estimator_increments = [n_estimators_test, n_estimators_final]
                hyperparams['warm_start'] = True
            else:
                if expected_memory_usage > (0.05 * max_memory_usage_ratio):  # Somewhat arbitrary, consider finding a better value, should it scale by cores?
                    # Causes ~10% training slowdown, so try to avoid if memory is not an issue
                    n_estimator_increments = [n_estimators_test, n_estimators_final]
                    hyperparams['warm_start'] = True

        hyperparams['n_estimators'] = n_estimator_increments[0]
        self.model = self._model_type(**hyperparams)

        time_train_start = time.time()
        for i, n_estimators in enumerate(n_estimator_increments):
            if i != 0:
                self.model.n_estimators = n_estimators
            self.model = self.model.fit(X_train, y_train)
            if (i == 0) and (len(n_estimator_increments) > 1):
                time_elapsed = time.time() - time_train_start
                model_size_bytes = 0
                for estimator in self.model.estimators_:  # Uses far less memory than pickling the entire forest at once
                    model_size_bytes += sys.getsizeof(pickle.dumps(estimator))
                expected_final_model_size_bytes = model_size_bytes * (n_estimators_final / self.model.n_estimators)
                available_mem = 4469755084
                model_memory_ratio = expected_final_model_size_bytes / available_mem

                ideal_memory_ratio = 0.15 * max_memory_usage_ratio
                n_estimators_ideal = min(n_estimators_final, math.floor(ideal_memory_ratio / model_memory_ratio * n_estimators_final))

                if n_estimators_final > n_estimators_ideal:
                    if n_estimators_ideal < n_estimators_minimum:
                        raise NotEnoughMemoryError  # don't train full model to avoid OOM error

                if time_limit is not None:
                    time_expected = time_train_start - time_start + (time_elapsed * n_estimators_ideal / n_estimators)
                    n_estimators_time = math.floor((time_limit - time_train_start + time_start) * n_estimators / time_elapsed)
                    if n_estimators_time < n_estimators_ideal:
                        if n_estimators_time < n_estimators_minimum:
                            raise TimeLimitExceeded
                        n_estimators_ideal = n_estimators_time

                for j in range(len(n_estimator_increments)):
                    if n_estimator_increments[j] > n_estimators_ideal:
                        n_estimator_increments[j] = n_estimators_ideal

        self.params_trained['n_estimators'] = self.model.n_estimators

    def get_model_feature_importance(self):
        if self.features is None:
            # TODO: Consider making this raise an exception
            return dict()
        return dict(zip(self.features, self.model.feature_importances_))

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=['object'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def predict_proba(self, X, normalize=None, **kwargs):
        X = self.preprocess(X, **kwargs)
        y_pred_proba = self.model.predict_proba(X)
        y_pred_proba = y_pred_proba.astype(np.float32)
        return y_pred_proba



class XTModel(RFModel):
    """
    Extra Trees model (scikit-learn): https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
    """
    def _get_model_type(self):
        return ExtraTreesClassifier


if __name__ == '__main__':
    for task in [40981, 40996]:
        X, y = sklearn.datasets.fetch_openml(data_id=task, return_X_y=True, as_frame=True)
        X = X.convert_dtypes()
        for x in X.columns:
            if not is_numeric_dtype(X[x]):
                X[x] = X[x].astype('category')
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        y = pd.DataFrame(y)

        problem_type = type_of_target(y)
        print(f"X={X.dtypes} problem={problem_type} {y}")
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
        model = RFModel(path='/tmp', problem_type=problem_type, metric=accuracy, name='random_forest', eval_metric=accuracy)
        model.fit(X_train=X_train, y_train=y_train)
        print(model.predict_proba(X_test).shape)
        print(accuracy(y_test.to_numpy().astype(int), model.predict(X_test)))
        model = XTModel(path='/tmp', problem_type=problem_type, metric=accuracy, name='ExtraTrees', eval_metric=accuracy)
        model.fit(X_train=X_train, y_train=y_train)
        print(model.predict_proba(X_test).shape)
        print(accuracy(y_test.to_numpy().astype(int), model.predict(X_test)))
