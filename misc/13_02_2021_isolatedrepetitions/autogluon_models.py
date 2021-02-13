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

HPO = {
    'autogluon': {
        'XGBClassifier': {
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
        },
        'DecisionTreeClassifier': {},
        'LinearDiscriminantAnalysis': {},

    },
    'sklearn': {
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
        },
        'DecisionTreeClassifier': {},
        'LinearDiscriminantAnalysis': {},

    },
}
model_func = {
    'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
    'RandomForestClassifier': RandomForestClassifier,
    'DecisionTreeClassifier': DecisionTreeClassifier,
    'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis,
}
