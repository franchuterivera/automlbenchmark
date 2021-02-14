from __future__ import division
import json
import logging
import os
import random
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, QuantileTransformer, FunctionTransformer  # PowerTransformer

from abstract_model import TimeLimitExceeded, fixedvals_from_searchspaces, AbstractModel, accuracy

from xgboost_model import OneHotMergeRaresHandleUnknownEncoder, OrdinalMergeRaresHandleUnknownEncoder
from rf_model import FeatureMetadata

from collections import OrderedDict
import mxnet as mx
from mxnet import nd, gluon
from mxboard import SummaryWriter
from math import pi, cos
from mxnet import lr_scheduler


class LRSequential(lr_scheduler.LRScheduler):
    r"""Compose Learning Rate Schedulers
    Parameters
    ----------
    schedulers: list
        list of LRScheduler objects
    """
    def __init__(self, schedulers):
        super(LRSequential, self).__init__()
        assert(len(schedulers) > 0)

        self.update_sep = []
        self.count = 0
        self.learning_rate = 0
        self.schedulers = []
        for lr in schedulers:
            self.add(lr)

    def add(self, scheduler):
        assert(isinstance(scheduler, LRScheduler))

        scheduler.offset = self.count
        self.count += scheduler.niters
        self.update_sep.append(self.count)
        self.schedulers.append(scheduler)

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        num_update = min(num_update, self.count - 1)
        ind = len(self.schedulers) - 1
        for i, sep in enumerate(self.update_sep):
            if sep > num_update:
                ind = i
                break
        lr = self.schedulers[ind]
        lr.update(num_update)
        self.learning_rate = lr.learning_rate


class LRScheduler(lr_scheduler.LRScheduler):
    r"""Learning Rate Scheduler
    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    nepochs : int
        Number of epochs to be scheduled.
    iters_per_epoch : int
        Number of iterations in each epoch.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step_iter : list
        A list of iterations to decay the learning rate.
    step_epoch : list
        A list of epochs to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """
    def __init__(self, mode, base_lr=0.1, target_lr=0,
                 niters=0, nepochs=0, iters_per_epoch=0, offset=0,
                 power=2, step_iter=None, step_epoch=None, step_factor=0.1,
                 baselr=None, targetlr=None):
        super(LRScheduler, self).__init__()
        assert(mode in ['constant', 'step', 'linear', 'poly', 'cosine'])

        self.mode = mode
        if mode == 'step':
            assert(step_iter is not None or step_epoch is not None)
        if baselr is not None:
            warnings.warn("baselr is deprecated. Please use base_lr.")
            if base_lr == 0.1:
                base_lr = baselr
        self.base_lr = base_lr
        if targetlr is not None:
            warnings.warn("targetlr is deprecated. Please use target_lr.")
            if target_lr == 0:
                target_lr = targetlr
        self.target_lr = target_lr
        if self.mode == 'constant':
            self.target_lr = self.base_lr

        self.niters = niters
        self.step = step_iter
        epoch_iters = nepochs * iters_per_epoch
        if epoch_iters > 0:
            self.niters = epoch_iters
            if step_epoch is not None:
                self.step = [s*iters_per_epoch for s in step_epoch]

        self.offset = offset
        self.power = power
        self.step_factor = step_factor

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        N = self.niters - 1
        T = num_update - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + cos(pi * T / N)) / 2
        elif self.mode == 'step':
            if self.step is not None:
                count = sum([1 for s in self.step if s <= T])
                factor = pow(self.step_factor, count)
            else:
                factor = 1
        else:
            raise NotImplementedError

        if self.mode == 'step':
            self.learning_rate = self.base_lr * factor
        else:
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor


def save_pkl_save(path, object, format=None, verbose=True):
    pickle_fn = lambda o, buffer: pickle.dump(o, buffer, protocol=4)
    save_with_fn(path, object, pickle_fn, format=format, verbose=verbose)


def load_pkl_load(path, format=None, verbose=True):
    if path.endswith('.pointer'):
        format = 'pointer'
    elif s3_utils.is_s3_url(path):
        format = 's3'
    if format == 'pointer':
        content_path = load_pointer.get_pointer_content(path)
        if content_path == path:
            raise RecursionError('content_path == path! : ' + str(path))
        return load(path=content_path)
    elif format == 's3':
        if verbose: logger.log(15, 'Loading: %s' % path)
        s3_bucket, s3_prefix = s3_utils.s3_path_to_bucket_prefix(s3_path=path)
        s3 = boto3.resource('s3')
        return pickle.loads(s3.Bucket(s3_bucket).Object(s3_prefix).get()['Body'].read())

    if verbose: logger.log(15, 'Loading: %s' % path)
    with open(path, 'rb') as fin:
        object = pickle.load(fin)
    return object




class TabularNNDataset:
    """ Class for preprocessing & storing/feeding data batches used by tabular data neural networks. Assumes entire dataset can be loaded into numpy arrays.
        Original Data table may contain numerical, categorical, and text (language) fields.
        Attributes:
            dataset (mxnet.gluon.data.dataset): Contains the raw data (use dataset._data to access).
                                                Different indices in this list correspond to different types of inputs to the neural network (each is 2D ND array)
                                                All vector-valued (continuous & one-hot) features are concatenated together into a single index of the dataset.
            data_desc (list[str]): Describes the data type of each index of dataset (options: 'vector','embed_<featname>', 'language_<featname>')
            dataloader (mxnet.gluon.data.DataLoader): Loads batches of data from dataset for neural net training and inference.
            embed_indices (list): which columns in dataset correspond to embed features (order matters!)
            language_indices (list): which columns in dataset correspond to language features (order matters!)
            vecfeature_col_map (dict): maps vector_feature_name ->  columns of dataset._data[vector] array that contain the data for this feature
            feature_dataindex_map (dict): maps feature_name -> i such that dataset._data[i] = data array for this feature. Cannot be used for vector-valued features, instead use vecfeature_col_map
            feature_groups (dict): maps feature_type (ie. 'vector' or 'embed' or 'language') to list of feature names of this type (empty list if there are no features of this type)
            vectordata_index (int): describes which element of the dataset._data list holds the vector data matrix (access via self.dataset._data[self.vectordata_index]); None if no vector features
            label_index (int): describing which element of the dataset._data list holds labels (access via self.dataset._data[self.label_index].asnumpy()); None if no labels
            num_categories_per_embedfeature (list): Number of categories for each embedding feature (order matters!)
            num_examples (int): number of examples in this dataset
            num_features (int): number of features (we only consider original variables as features, so num_features may not correspond to dimensionality of the data eg in the case of one-hot encoding)
            num_classes (int): number of classes (only used for multiclass classification)
        Note: Default numerical data-type is converted to float32 (as well as labels in regression).
    """

    DATAOBJ_SUFFIX = '_tabNNdataset.pkl' # hard-coded names for files. This file contains pickled TabularNNDataset object
    DATAVALUES_SUFFIX = '_tabNNdata.npz' # This file contains raw data values as data_list of NDArrays

    def __init__(self, processed_array, feature_arraycol_map, feature_type_map, batch_size, num_dataloading_workers, problem_type,
                 labels=None, is_test=True):
        """ Args:
                processed_array: 2D numpy array returned by preprocessor. Contains raw data of all features as columns
                feature_arraycol_map (OrderedDict): Mapsfeature-name -> list of column-indices in processed_array corresponding to this feature
                feature_type_map (OrderedDict): Maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
                labels (pd.Series): list of labels (y) if available
                batch_size (int): number of examples to put in each mini-batch
                num_dataloading_workers (int): number of threads to devote to loading mini-batches of data rather than model-training
        """
        self.dataset = None
        self.dataloader = None
        self.problem_type = problem_type
        self.num_examples = processed_array.shape[0]
        self.num_features = len(feature_arraycol_map) # number of features (!=dim(processed_array) because some features may be vector-valued, eg one-hot)
        self.batch_size = min(self.num_examples, batch_size)
        self.is_test = is_test
        self.num_dataloading_workers = num_dataloading_workers
        last_batch_size = self.num_examples % self.batch_size
        if last_batch_size == 0:
            last_batch_size = self.batch_size
        # TODO: The code fixes the crash on mxnet gluon interpreting a single value in a batch incorrectly.
        #  Comment out to see crash if data would have single row as final batch on test prediction (such as 1025 rows for batch size 512)
        if (self.num_examples != 1) and self.is_test and (last_batch_size == 1):
            init_batch_size = self.batch_size
            while last_batch_size == 1:
                self.batch_size = self.batch_size + 1
                last_batch_size = self.num_examples % self.batch_size
                if last_batch_size == 0:
                    last_batch_size = self.batch_size
                if self.batch_size > init_batch_size+10:
                    # Hard set to avoid potential infinite loop, don't think its mathematically possible to reach this code however.
                    self.batch_size = self.num_examples
                    last_batch_size = 0

        if feature_arraycol_map.keys() != feature_type_map.keys():
            raise ValueError("feature_arraycol_map and feature_type_map must share same keys")
        self.feature_groups = {'vector': [], 'embed': [], 'language': []} # maps feature_type -> list of feature_names (order is preserved in list)
        self.feature_type_map = feature_type_map
        for feature in feature_type_map:
            if feature_type_map[feature] == 'vector':
                self.feature_groups['vector'].append(feature)
            elif feature_type_map[feature] == 'embed':
                self.feature_groups['embed'].append(feature)
            elif feature_type_map[feature] == 'language':
                self.feature_groups['language'].append(feature)
            else:
                raise ValueError("unknown feature type: %s" % feature)

        if not self.is_test and labels is None:
            raise ValueError("labels must be provided when is_test = False")
        if labels is not None and len(labels) != self.num_examples:
            raise ValueError("number of labels and training examples do not match")

        data_list = [] # stores all data of each feature-type in list used to construct MXNet dataset. Each index of list = 2D NDArray.
        self.label_index = None # int describing which element of the dataset._data list holds labels
        self.data_desc = [] # describes feature-type of each index of data_list
        self.vectordata_index = None # int describing which element of the dataset._data list holds the vector data matrix
        self.vecfeature_col_map = {} # maps vector_feature_name ->  columns of dataset._data[vector] array that contain data for this feature
        self.feature_dataindex_map = {} # maps feature_name -> i such that dataset._data[i] = data array for this feature. Cannot be used for vector-valued features, instead use: self.vecfeature_col_map

        if len(self.feature_groups['vector']) > 0:
            vector_inds = [] # columns of processed_array corresponding to vector data
            for feature in feature_type_map:
                if feature_type_map[feature] == 'vector':
                    current_last_ind = len(vector_inds) # current last index of the vector datamatrix
                    vector_inds += feature_arraycol_map[feature]
                    new_last_ind = len(vector_inds) # new last index of the vector datamatrix
                    self.vecfeature_col_map[feature] = list(range(current_last_ind, new_last_ind))
            data_list.append(mx.nd.array(processed_array[:,vector_inds], dtype='float32')) # Matrix of data from all vector features
            self.data_desc.append("vector")
            self.vectordata_index = len(data_list) - 1

        if len(self.feature_groups['embed']) > 0:
            for feature in feature_type_map:
                if feature_type_map[feature] == 'embed':
                    feature_colind = feature_arraycol_map[feature]
                    data_list.append(mx.nd.array(processed_array[:,feature_colind], dtype='int32')) # array of ints with data for this embedding feature
                    self.data_desc.append("embed")
                    self.feature_dataindex_map[feature]  = len(data_list)-1

        if len(self.feature_groups['language']) > 0:
            for feature in feature_type_map:
                if feature_type_map[feature] == 'language':
                    feature_colinds = feature_arraycol_map[feature]
                    data_list.append(mx.nd.array(processed_array[:,feature_colinds], dtype='int32')) # array of ints with data for this language feature
                    self.data_desc.append("language")
                    self.feature_dataindex_map[feature]  = len(data_list)-1

        self.num_classes = None
        if labels is not None:
            labels = np.array(labels)
            self.data_desc.append("label")
            self.label_index = len(data_list) # To access data labels, use: self.dataset._data[self.label_index]
            self.num_classes = None
            self.num_classes = len(set(np.unique(labels)))
            data_list.append(mx.nd.array(labels.reshape(len(labels),1)))

        self.embed_indices = [i for i in range(len(self.data_desc)) if 'embed' in self.data_desc[i]] # list of indices of embedding features in self.dataset, order matters!
        self.language_indices = [i for i in range(len(self.data_desc)) if 'language' in self.data_desc[i]]  # list of indices of language features in self.dataset, order matters!
        self.num_categories_per_embed_feature = None
        self.generate_dataset_and_dataloader(data_list=data_list)
        if not self.is_test:
            self.num_categories_per_embedfeature = self.getNumCategoriesEmbeddings()

    def generate_dataset_and_dataloader(self, data_list):
        self.dataset = mx.gluon.data.dataset.ArrayDataset(*data_list)  # Access ith embedding-feature via: self.dataset._data[self.data_desc.index('embed_'+str(i))].asnumpy()
        self.dataloader = mx.gluon.data.DataLoader(self.dataset, self.batch_size, shuffle=not self.is_test,
                                                   last_batch='keep' if self.is_test else 'rollover',
                                                   num_workers=self.num_dataloading_workers)  # no need to shuffle test data

    def has_vector_features(self):
        """ Returns boolean indicating whether this dataset contains vector features """
        return self.vectordata_index is not None

    def num_embed_features(self):
        """ Returns number of embed features in this dataset """
        return len(self.feature_groups['embed'])

    def num_language_features(self):
        """ Returns number of language features in this dataset """
        return len(self.feature_groups['language'])

    def num_vector_features(self):
        """ Number of vector features (each onehot feature counts = 1, regardless of how many categories) """
        return len(self.feature_groups['vector'])

    def get_labels(self):
        """ Returns numpy array of labels for this dataset """
        if self.label_index is not None:
            return self.dataset._data[self.label_index].asnumpy().flatten()
        else:
            return None

    def getNumCategoriesEmbeddings(self):
        """ Returns number of categories for each embedding feature.
            Should only be applied to training data.
            If training data feature contains unique levels 1,...,n-1, there are actually n categories,
            since category n is reserved for unknown test-time categories.
        """
        if self.num_categories_per_embed_feature is not None:
            return self.num_categories_per_embedfeature
        else:
            num_embed_feats = self.num_embed_features()
            num_categories_per_embedfeature = [0] * num_embed_feats
            for i in range(num_embed_feats):
                feat_i = self.feature_groups['embed'][i]
                feat_i_data = self.get_feature_data(feat_i).flatten().tolist()
                num_categories_i = len(set(feat_i_data)) # number of categories for ith feature
                num_categories_per_embedfeature[i] = num_categories_i + 1 # to account for unknown test-time categories
            return num_categories_per_embedfeature

    def get_feature_data(self, feature, asnumpy=True):
        """ Returns all data for this feature.
            Args:
                feature (str): name of feature of interest (in processed dataframe)
                asnumpy (bool): should we return 2D numpy array or MXNet NDarray
        """
        nonvector_featuretypes = set(['embed', 'language'])
        if feature not in self.feature_type_map:
            raise ValueError("unknown feature encountered: %s" % feature)
        if self.feature_type_map[feature] == 'vector':
            vector_datamatrix = self.dataset._data[self.vectordata_index] # does not work for one-hot...
            feature_data = vector_datamatrix[:, self.vecfeature_col_map[feature]]
        elif self.feature_type_map[feature] in nonvector_featuretypes:
            feature_idx = self.feature_dataindex_map[feature]
            feature_data = self.dataset._data[feature_idx]
        else:
            raise ValueError("Unknown feature specified: " % feature)
        if asnumpy:
            return feature_data.asnumpy()
        else:
            return feature_data

    def get_feature_batch(self, feature, data_batch, asnumpy=False):
        """ Returns part of this batch corresponding to data from a single feature
            Args:
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:
        """
        nonvector_featuretypes = set(['embed', 'language'])
        if feature not in self.feature_type_map:
            raise ValueError("unknown feature encountered: %s" % feature)
        if self.feature_type_map[feature] == 'vector':
            vector_datamatrix = data_batch[self.vectordata_index]
            feature_data = vector_datamatrix[:, self.vecfeature_col_map[feature]]
        elif self.feature_type_map[feature] in nonvector_featuretypes:
            feature_idx = self.feature_dataindex_map[feature]
            feature_data = data_batch[feature_idx]
        else:
            raise ValueError("Unknown feature specified: " % feature)
        if asnumpy:
            return feature_data.asnumpy()
        else:
            return feature_data

    def format_batch_data(self, data_batch, ctx):
        """ Partitions data from this batch into different data types.
            Args:
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:
                formatted_batch (dict): {'vector': array of vector_datamatrix,
                                         'embed': list of embedding features' batch data,
                                         'language': list of language features batch data,
                                         'label': array of labels}
                                        where each key in dict may be missing.
        """
        if not isinstance(data_batch, list):
            data_batch = [data_batch] # Need to convert to list if dimension was dropped during batching

        if len(data_batch[0].shape) == 1:
            data_batch[0] = data_batch[0].expand_dims(axis=0)
        formatted_batch = {}
        if self.has_vector_features(): # None if there is no vector data
            formatted_batch['vector'] = data_batch[self.vectordata_index].as_in_context(ctx)
        if self.num_embed_features() > 0:
            formatted_batch['embed'] = []
            for i in self.embed_indices:
                formatted_batch['embed'].append(data_batch[i].as_in_context(ctx))
        if self.num_language_features() > 0:
            formatted_batch['language'] = []
            for i in self.language_indices:
                formatted_batch['language'].append(data_batch[i].as_in_context(ctx))
        if self.label_index is not None: # is None if there are no labels
            formatted_batch['label'] = data_batch[self.label_index].as_in_context(ctx)

        return formatted_batch

    def mask_features_batch(self, features, mask_value, data_batch):
        """ Returns new batch where all values of the indicated features have been replaced by the provided mask_value.
            Args:
                features (list[str]): list of feature names that should be masked.
                mask_value (float): value of mask which original feature values should be replaced by. If None, we replace by mean/mode/unknown
                data_batch (nd.array): the batch of data as provided by self.dataloader
            Returns:
                new_batch (nd.array): batch of masked data in same format as data_batch
        """
        return None # TODO

    def save(self, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + self.DATAOBJ_SUFFIX
        datalist_file = file_prefix + self.DATAVALUES_SUFFIX
        data_list = self.dataset._data
        self.dataset = None  # Avoid pickling these
        self.dataloader = None
        save_pkl_save(path=dataobj_file, object=self)
        mx.nd.save(datalist_file, data_list)
        logger.debug("TabularNN Dataset saved to files: \n %s \n %s" % (dataobj_file, datalist_file))

    @classmethod
    def load(cls, file_prefix=""):
        """ Additional naming changes will be appended to end of file_prefix (must contain full absolute path) """
        dataobj_file = file_prefix + cls.DATAOBJ_SUFFIX
        datalist_file = file_prefix + cls.DATAVALUES_SUFFIX
        dataset: TabularNNDataset = load_pkl_load(path=dataobj_file)
        data_list = mx.nd.load(datalist_file)
        dataset.generate_dataset_and_dataloader(data_list=data_list)
        logger.debug("TabularNN Dataset loaded from files: \n %s \n %s" % (dataobj_file, datalist_file))
        return dataset








class NumericBlock(gluon.HybridBlock):
    """ Single Dense layer that jointly embeds all numeric and one-hot features """
    def __init__(self, params, **kwargs):
        super(NumericBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.body = gluon.nn.Dense(params['numeric_embed_dim'], activation=params['activation'])

    def hybrid_forward(self, F, x):
        return self.body(x)


class EmbedBlock(gluon.HybridBlock):
    """ Used to embed a single embedding feature. """
    def __init__(self, embed_dim, num_categories, **kwargs):
        super(EmbedBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.body = gluon.nn.Embedding(input_dim=num_categories, output_dim=embed_dim,
                                           weight_initializer=mx.init.Orthogonal(scale=0.1, rand_type='uniform')) # for Xavier-style: scale = np.sqrt(3/float(embed_dim))

    def hybrid_forward(self, F, x):
        return self.body(x)


class FeedforwardBlock(gluon.HybridBlock):
    """ Standard Feedforward layers """
    def __init__(self, params, num_net_outputs, **kwargs):
        super(FeedforwardBlock, self).__init__(**kwargs)
        layers = params['layers']
        with self.name_scope():
            self.body = gluon.nn.HybridSequential()
            if params['use_batchnorm']:
                 self.body.add(gluon.nn.BatchNorm())
            if params['dropout_prob'] > 0:
                self.body.add(gluon.nn.Dropout(params['dropout_prob']))
            for i in range(len(layers)):
                layer_width = layers[i]
                if layer_width < 1 or int(layer_width) != layer_width:
                    raise ValueError("layers must be ints >= 1")
                self.body.add(gluon.nn.Dense(layer_width, activation=params['activation']))
                if params['use_batchnorm']:
                    self.body.add(gluon.nn.BatchNorm())
                if params['dropout_prob'] > 0:
                    self.body.add(gluon.nn.Dropout(params['dropout_prob']))
            self.body.add(gluon.nn.Dense(num_net_outputs, activation=None))

    def hybrid_forward(self, F, x):
        return self.body(x)


class WideAndDeepBlock(gluon.HybridBlock):
    """ Standard feedforward layers with a single skip connection from output directly to input (ie. deep and wide network).
    """
    def __init__(self, params, num_net_outputs, **kwargs):
        super(WideAndDeepBlock, self).__init__(**kwargs)
        self.deep = FeedforwardBlock(params, num_net_outputs, **kwargs)
        with self.name_scope(): # Skip connection, ie. wide network branch
            self.wide = gluon.nn.Dense(num_net_outputs, activation=None)

    def hybrid_forward(self, F, x):
        return self.deep(x) + self.wide(x)


class EmbedNet(gluon.Block): # TODO: hybridize?
    """ Gluon net with input layers to handle numerical data & categorical embeddings
        which are concatenated together after input layer and then passed into feedforward network.
        If architecture_desc != None, then we assume EmbedNet has already been previously created,
        and we create a new EmbedNet based on the provided architecture description
        (thus ignoring train_dataset, params, num_net_outputs).
    """
    def __init__(self, train_dataset=None, params=None, num_net_outputs=None, architecture_desc=None, ctx=None, **kwargs):
        if (architecture_desc is None) and (train_dataset is None or params is None or num_net_outputs is None):
            raise ValueError("train_dataset, params, num_net_outputs cannot = None if architecture_desc=None")
        super(EmbedNet, self).__init__(**kwargs)
        if architecture_desc is None: # Adpatively specify network architecture based on training dataset
            self.from_logits = False
            self.has_vector_features = train_dataset.has_vector_features()
            self.has_embed_features = train_dataset.num_embed_features() > 0
            self.has_language_features = train_dataset.num_language_features() > 0
            if self.has_embed_features:
                num_categs_per_feature = train_dataset.getNumCategoriesEmbeddings()
                embed_dims = getEmbedSizes(train_dataset, params, num_categs_per_feature)
        else: # Ignore train_dataset, params, etc. Recreate architecture based on description:
            self.architecture_desc = architecture_desc
            self.has_vector_features = architecture_desc['has_vector_features']
            self.has_embed_features = architecture_desc['has_embed_features']
            self.has_language_features = architecture_desc['has_language_features']
            self.from_logits = architecture_desc['from_logits']
            num_net_outputs = architecture_desc['num_net_outputs']
            params = architecture_desc['params']
            if self.has_embed_features:
                num_categs_per_feature = architecture_desc['num_categs_per_feature']
                embed_dims = architecture_desc['embed_dims']

        # Define neural net parameters:
        if self.has_vector_features:
            self.numeric_block = NumericBlock(params)
        if self.has_embed_features:
            self.embed_blocks = gluon.nn.HybridSequential()
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.add(EmbedBlock(embed_dims[i], num_categs_per_feature[i]))
        if self.has_language_features:
            self.text_block = None
            raise NotImplementedError("text data cannot be handled")
        if params['network_type'] == 'feedforward':
            self.output_block = FeedforwardBlock(params, num_net_outputs)
        elif params['network_type'] == 'widedeep':
            self.output_block = WideAndDeepBlock(params, num_net_outputs)
        else:
            raise ValueError("unknown network_type specified: %s" % params['network_type'])

        y_range = params['y_range'] # Used specifically for regression. = None for classification.
        self.y_constraint = None # determines if Y-predictions should be constrained
        if y_range is not None:
            if y_range[0] == -np.inf and y_range[1] == np.inf:
                self.y_constraint = None # do not worry about Y-range in this case
            elif y_range[0] >= 0 and y_range[1] == np.inf:
                self.y_constraint = 'nonnegative'
            elif y_range[0] == -np.inf and y_range[1] <= 0:
                self.y_constraint = 'nonpositive'
            else:
                self.y_constraint = 'bounded'
            self.y_lower = nd.array(params['y_range'][0]).reshape(1,)
            self.y_upper = nd.array(params['y_range'][1]).reshape(1,)
            if ctx is not None:
                self.y_lower.as_in_context(ctx)
                self.y_upper.as_in_context(ctx)
            self.y_span = self.y_upper - self.y_lower

        if architecture_desc is None: # Save Architecture description
            self.architecture_desc = {'has_vector_features': self.has_vector_features,
                                  'has_embed_features': self.has_embed_features,
                                  'has_language_features': self.has_language_features,
                                  'params': params, 'num_net_outputs': num_net_outputs,
                                  'from_logits': self.from_logits}
            if self.has_embed_features:
                self.architecture_desc['num_categs_per_feature'] = num_categs_per_feature
                self.architecture_desc['embed_dims'] = embed_dims
            if self.has_language_features:
                self.architecture_desc['text_TODO'] = None # TODO: store text architecture

    def forward(self, data_batch):
        if self.has_vector_features:
            numerical_data = data_batch['vector'] # NDArray
            numerical_activations = self.numeric_block(numerical_data)
            input_activations = numerical_activations
        if self.has_embed_features:
            embed_data = data_batch['embed'] # List

            # TODO: Remove below lines or write logic to switch between using these lines and the multithreaded version once multithreaded version is optimized
            embed_activations = self.embed_blocks[0](embed_data[0])
            for i in range(1, len(self.embed_blocks)):
                embed_activations = nd.concat(embed_activations,
                                              self.embed_blocks[i](embed_data[i]), dim=2)

            # TODO: Optimize below to perform better before using
            # lock = threading.Lock()
            # results = {}
            #
            # def _worker(i, results, embed_block, embed_data, is_recording, is_training, lock):
            #     if is_recording:
            #         with mx.autograd.record(is_training):
            #             output = embed_block(embed_data)
            #     else:
            #         output = embed_block(embed_data)
            #     output.wait_to_read()
            #     with lock:
            #         results[i] = output
            #
            # is_training = mx.autograd.is_training()
            # is_recording = mx.autograd.is_recording()
            # threads = [threading.Thread(target=_worker,
            #                     args=(i, results, embed_block, embed_data,
            #                           is_recording, is_training, lock),
            #                     )
            #    for i, (embed_block, embed_data) in
            #    enumerate(zip(self.embed_blocks, embed_data))]
            #
            # for thread in threads:
            #     thread.start()
            # for thread in threads:
            #     thread.join()
            #
            # embed_activations = []
            # for i in range(len(results)):
            #     output = results[i]
            #     embed_activations.append(output)
            #
            # #embed_activations = []
            # #for i in range(len(self.embed_blocks)):
            # #    embed_activations.append(self.embed_blocks[i](embed_data[i]))
            # embed_activations = nd.concat(*embed_activations, dim=2)
            embed_activations = embed_activations.flatten()
            if not self.has_vector_features:
                input_activations = embed_activations
            else:
                input_activations = nd.concat(embed_activations, input_activations)
        if self.has_language_features:
            language_data = data_batch['language']
            language_activations = self.text_block(language_data) # TODO: create block to embed text fields
            if (not self.has_vector_features) and (not self.has_embed_features):
                input_activations = language_activations
            else:
                input_activations = nd.concat(language_activations, input_activations)
        if self.y_constraint is None:
            return self.output_block(input_activations)
        else:
            unscaled_pred = self.output_block(input_activations)
            if self.y_constraint == 'nonnegative':
                return self.y_lower + nd.abs(unscaled_pred)
            elif self.y_constraint == 'nonpositive':
                return self.y_upper - nd.abs(unscaled_pred)
            else:
                """
                print("unscaled_pred",unscaled_pred)
                print("nd.sigmoid(unscaled_pred)", nd.sigmoid(unscaled_pred))
                print("self.y_span", self.y_span)
                print("self.y_lower", self.y_lower)
                print("self.y_lower.shape", self.y_lower.shape)
                print("nd.sigmoid(unscaled_pred).shape", nd.sigmoid(unscaled_pred).shape)
                """
                return nd.sigmoid(unscaled_pred) * self.y_span + self.y_lower


""" OLD
    def _create_embednet_from_architecture(architecture_desc):
        # Recreate network architecture based on provided description
        self.architecture_desc = architecture_desc
        self.has_vector_features = architecture_desc['has_vector_features']
        self.has_embed_features = architecture_desc['has_embed_features']
        self.has_language_features = architecture_desc['has_language_features']
        self.from_logits = architecture_desc['from_logits']
        num_net_outputs = architecture_desc['num_net_outputs']
        params = architecture_desc['params']
        if self.has_vector_features:
            self.numeric_block = NumericBlock(params)
        if self.has_embed_features:
            self.embed_blocks = gluon.nn.HybridSequential()
            num_categs_per_feature = architecture_desc['num_categs_per_feature']
            embed_dims = architecture_desc['embed_dims']
            for i in range(len(num_categs_per_feature)):
                self.embed_blocks.add(EmbedBlock(embed_dims[i], num_categs_per_feature[i]))
        if self.has_language_features:
            self.text_block = architecture_desc['text_TODO']
        if
        self.output_block = FeedforwardBlock(params, num_net_outputs) # TODO
        self.from_logits = False
"""


def getEmbedSizes(train_dataset, params, num_categs_per_feature):
    """ Returns list of embedding sizes for each categorical variable.
        Selects this adaptively based on training_datset.
        Note: Assumes there is at least one embed feature.
    """
    max_embedding_dim = params['max_embedding_dim']
    embed_exponent = params['embed_exponent']
    size_factor = params['embedding_size_factor']
    embed_dims = [int(size_factor*max(2, min(max_embedding_dim,
                                      1.6 * num_categs_per_feature[i]**embed_exponent)))
                   for i in range(len(num_categs_per_feature))]
    return embed_dims


















warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
logger = logging.getLogger(__name__)
EPS = 1e-10  # small number


class AbstractNeuralNetworkModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._types_of_features = None

    # TODO: v0.1 clean method
    def _get_types_of_features(self, df, skew_threshold=None, embed_min_categories=None, use_ngram_features=None, needs_extra_types=True):
        """ Returns dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = ordered list of feature-names falling into each category.
            Each value is a list of feature-names corresponding to columns in original dataframe.
            TODO: ensure features with zero variance have already been removed before this function is called.
        """
        if self._types_of_features is not None:
            Warning("Attempting to _get_types_of_features for Model, but previously already did this.")

        feature_types = self.feature_metadata.get_type_group_map_raw()
        categorical_featnames = feature_types['category'] + feature_types['object'] + feature_types['bool']
        continuous_featnames = feature_types['float'] + feature_types['int']  # + self.__get_feature_type_if_present('datetime')
        language_featnames = [] # TODO: not implemented. This should fetch text features present in the data
        valid_features = categorical_featnames + continuous_featnames + language_featnames

        if len(valid_features) < df.shape[1]:
            unknown_features = [feature for feature in df.columns if feature not in valid_features]
            logger.log(15, f"Model will additionally ignore the following columns: {unknown_features}")
            print( f"Model will additionally ignore the following columns: {unknown_features}")
            df = df.drop(columns=unknown_features)
            self.features = list(df.columns)

        self.features_to_drop = df.columns[df.isna().all()].tolist()  # drop entirely NA columns which may arise after train/val split
        if self.features_to_drop:
            logger.log(15, f"Model will additionally ignore the following columns: {self.features_to_drop}")
            df = df.drop(columns=self.features_to_drop)

        if needs_extra_types is True:
            types_of_features = {'continuous': [], 'skewed': [], 'onehot': [], 'embed': [], 'language': []}
            # continuous = numeric features to rescale
            # skewed = features to which we will apply power (ie. log / box-cox) transform before normalization
            # onehot = features to one-hot encode (unknown categories for these features encountered at test-time are encoded as all zeros). We one-hot encode any features encountered that only have two unique values.
            features_to_consider = [feat for feat in self.features if feat not in self.features_to_drop]
            for feature in features_to_consider:
                feature_data = df[feature]  # pd.Series
                num_unique_vals = len(feature_data.unique())
                if num_unique_vals == 2:  # will be onehot encoded regardless of proc.embed_min_categories value
                    types_of_features['onehot'].append(feature)
                elif feature in continuous_featnames:
                    if np.abs(feature_data.skew()) > skew_threshold:
                        types_of_features['skewed'].append(feature)
                    else:
                        types_of_features['continuous'].append(feature)
                elif feature in categorical_featnames:
                    if num_unique_vals >= embed_min_categories:  # sufficiently many categories to warrant learned embedding dedicated to this feature
                        types_of_features['embed'].append(feature)
                    else:
                        types_of_features['onehot'].append(feature)
                elif feature in language_featnames:
                    types_of_features['language'].append(feature)
        else:
            types_of_features = []
            for feature in valid_features:
                if feature in categorical_featnames:
                    feature_type = 'CATEGORICAL'
                elif feature in continuous_featnames:
                    feature_type = 'SCALAR'
                elif feature in language_featnames:
                    feature_type = 'TEXT'
                else:
                    raise ValueError(f'Invalid feature: {feature}')

                types_of_features.append({"name": feature, "type": feature_type})

        return types_of_features, df


def get_fixed_params():
    """ Parameters that currently cannot be searched during HPO """
    fixed_params = {
        'num_epochs': 500,  # maximum number of epochs for training NN
        'epochs_wo_improve': 20,  # we terminate training if validation performance hasn't improved in the last 'epochs_wo_improve' # of epochs
        # TODO: Epochs could take a very long time, we may want smarter logic than simply # of epochs without improvement (slope, difference in score, etc.)
        'seed_value': None,  # random seed for reproducibility (set = None to ignore)
        # For data processing:
        'proc.embed_min_categories': 4,  # apply embedding layer to categorical features with at least this many levels. Features with fewer levels are one-hot encoded. Choose big value to avoid use of Embedding layers
        # Options: [3,4,10, 100, 1000]
        'proc.impute_strategy': 'median',  # # strategy argument of sklearn.SimpleImputer() used to impute missing numeric values
        # Options: ['median', 'mean', 'most_frequent']
        'proc.max_category_levels': 100,  # maximum number of allowed levels per categorical feature
        # Options: [10, 100, 200, 300, 400, 500, 1000, 10000]
        'proc.skew_threshold': 0.99,  # numerical features whose absolute skewness is greater than this receive special power-transform preprocessing. Choose big value to avoid using power-transforms
        # Options: [0.2, 0.3, 0.5, 0.8, 1.0, 10.0, 100.0]
        # Old params: These are now set based off of nthreads_per_trial, ngpus_per_trial.
        # 'num_dataloading_workers': 1,  # Will be overwritten by nthreads_per_trial, can be >= 1
        # 'ctx': mx.cpu(),  # Will be overwritten by ngpus_per_trial if unspecified (can alternatively be: mx.gpu())
    }
    return fixed_params


def get_hyper_params():
    """ Parameters that currently can be tuned during HPO """
    hyper_params = {
        ## Hyperparameters for neural net architecture:
        'network_type': 'widedeep',  # Type of neural net used to produce predictions
        # Options: ['widedeep', 'feedforward']
        'layers': None,  # List of widths (num_units) for each hidden layer (Note: only specifies hidden layers. These numbers are not absolute, they will also be scaled based on number of training examples and problem type)
        # Options: List of lists that are manually created
        'numeric_embed_dim': None,  # Size of joint embedding for all numeric+one-hot features.
        # Options: integer values between 10-10000
        'activation': 'relu',  # Activation function
        # Options: ['relu', 'softrelu', 'tanh', 'softsign']
        'max_layer_width': 2056,  # maximum number of hidden units in network layer (integer > 0)
        # Does not need to be searched by default
        'embedding_size_factor': 1.0,  # scaling factor to adjust size of embedding layers (float > 0)
        # Options: range[0.01 - 100] on log-scale
        'embed_exponent': 0.56,  # exponent used to determine size of embedding layers based on # categories.
        'max_embedding_dim': 100,  # maximum size of embedding layer for a single categorical feature (int > 0).
        ## Regression-specific hyperparameters:
        'y_range': None,  # Tuple specifying whether (min_y, max_y). Can be = (-np.inf, np.inf).
        # If None, inferred based on training labels. Note: MUST be None for classification tasks!
        'y_range_extend': 0.05,  # Only used to extend size of inferred y_range when y_range = None.
        ## Hyperparameters for neural net training:
        'use_batchnorm': True,  # whether or not to utilize Batch-normalization
        # Options: [True, False]
        'dropout_prob': 0.1,  # dropout probability, = 0 turns off Dropout.
        # Options: range(0.0, 0.5)
        'batch_size': 512,  # batch-size used for NN training
        # Options: [32, 64, 128. 256, 512, 1024, 2048]
        'loss_function': None,  # MXNet loss function minimized during training
        'optimizer': 'adam',  # MXNet optimizer to use.
        # Options include: ['adam','sgd']
        'learning_rate': 3e-4,  # learning rate used for NN training (float > 0)
        'weight_decay': 1e-6,  # weight decay regularizer (float > 0)
        'clip_gradient': 100.0,  # gradient clipping threshold (float > 0)
        'momentum': 0.9,  # momentum which is only used for SGD optimizer
        'lr_scheduler': None,  # If not None, string specifying what type of learning rate scheduler to use (may override learning_rate).
        # Options: [None, 'cosine', 'step', 'poly', 'constant']
        # Below are hyperparameters specific to the LR scheduler (only used if lr_scheduler != None). For more info, see: https://gluon-cv.mxnet.io/api/utils.html#gluoncv.utils.LRScheduler
        'base_lr': 3e-5,  # smallest LR (float > 0)
        'target_lr': 1.0,  # largest LR (float > 0)
        'lr_decay': 0.1,  # step factor used to decay LR (float in (0,1))
        'warmup_epochs': 10,  # number of epochs at beginning of training in which LR is linearly ramped up (float > 1).
        ## Feature-specific hyperparameters:
        'use_ngram_features': False,  # If False, will drop automatically generated ngram features from language features. This results in worse model quality but far faster inference and training times.
        # Options: [True, False]
    }
    return hyper_params


# Note: params for original NNTabularModel were:
# weight_decay=0.01, dropout_prob = 0.1, batch_size = 2048, lr = 1e-2, epochs=30, layers= [200, 100] (semi-equivalent to our layers = [100],numeric_embed_dim=200)
def get_default_param(problem_type, num_classes=None):
    if problem_type == 'binary':
        return get_param_binary()
    elif problem_type == 'multiclass':
        return get_param_multiclass(num_classes=num_classes)
    else:
        return get_param_binary()


def get_param_multiclass(num_classes):
    params = get_fixed_params()
    params.update(get_hyper_params())
    return params


def get_param_binary():
    params = get_fixed_params()
    params.update(get_hyper_params())
    return params


def get_param_regression():
    params = get_fixed_params()
    params.update(get_hyper_params())
    return params


# TODO: Gets stuck after infering feature types near infinitely in nyc-jiashenliu-515k-hotel-reviews-data-in-europe dataset, 70 GB of memory, c5.9xlarge
#  Suspect issue is coming from embeddings due to text features with extremely large categorical counts.
class TabularNeuralNetModel(AbstractNeuralNetworkModel):
    """ Class for neural network models that operate on tabular data.
        These networks use different types of input layers to process different types of data in various columns.
        Attributes:
            _types_of_features (dict): keys = 'continuous', 'skewed', 'onehot', 'embed', 'language'; values = column-names of Dataframe corresponding to the features of this type
            feature_arraycol_map (OrderedDict): maps feature-name -> list of column-indices in df corresponding to this feature
        self.feature_type_map (OrderedDict): maps feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        processor (sklearn.ColumnTransformer): scikit-learn preprocessor object.
        Note: This model always assumes higher values of self.eval_metric indicate better performance.
    """

    # Constants used throughout this class:
    # model_internals_file_name = 'model-internals.pkl' # store model internals here
    unique_category_str = '!missing!' # string used to represent missing values and unknown categories for categorical features. Should not appear in the dataset
    params_file_name = 'net.params' # Stores parameters of final network
    temp_file_name = 'temp_net.params' # Stores temporary network parameters (eg. during the course of training)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        TabularNeuralNetModel object.
        Parameters
        ----------
        path (str): file-path to directory where to save files associated with this model
        name (str): name used to refer to this model
        problem_type (str): what type of prediction problem is this model used for
        eval_metric (func): function used to evaluate performance (Note: we assume higher = better)
        hyperparameters (dict): various hyperparameters for neural network and the NN-specific data processing
        features (list): List of predictive features to use, other features are ignored by the model.
        """
        self.feature_arraycol_map = None
        self.feature_type_map = None
        self.features_to_drop = []  # may change between different bagging folds. TODO: consider just removing these from self.features if it works with bagging
        self.processor = None  # data processor
        self.summary_writer = None
        self.ctx = None
        self.batch_size = None
        self.num_dataloading_workers = None
        self.num_dataloading_workers_inference = 0
        self.params_post_fit = None
        self.num_net_outputs = None
        self._architecture_desc = None
        self.optimizer = None
        self.verbosity = None
        self.eval_metric_name = self.stopping_metric.name

    def _set_default_params(self):
        """ Specifies hyperparameter values to use by default """
        default_params = get_default_param(self.problem_type)
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        extra_auxiliary_params = dict(
            ignored_type_group_raw=['object'],
            ignored_type_group_special=['text_ngram', 'text_as_category'],
        )
        default_auxiliary_params.update(extra_auxiliary_params)
        return default_auxiliary_params

    def set_net_defaults(self, train_dataset, params):
        """ Sets dataset-adaptive default values to use for our neural network """
        if (self.problem_type == 'multiclass'):
            self.num_net_outputs = train_dataset.num_classes
        elif self.problem_type == 'binary':
            self.num_net_outputs = 2
        else:
            raise ValueError("unknown problem_type specified: %s" % self.problem_type)

        if params['layers'] is None:  # Use default choices for MLP architecture
            default_sizes = [256, 128]  # will be scaled adaptively
            # base_size = max(1, min(self.num_net_outputs, 20)/2.0) # scale layer width based on number of classes
            base_size = max(1, min(self.num_net_outputs, 100) / 50)  # TODO: Updated because it improved model quality and made training far faster
            default_layer_sizes = [defaultsize*base_size for defaultsize in default_sizes]
            layer_expansion_factor = 1  # TODO: consider scaling based on num_rows, eg: layer_expansion_factor = 2-np.exp(-max(0,train_dataset.num_examples-10000))

            max_layer_width = params['max_layer_width']
            params['layers'] = [int(min(max_layer_width, layer_expansion_factor*defaultsize)) for defaultsize in default_layer_sizes]

        if train_dataset.has_vector_features() and params['numeric_embed_dim'] is None:  # Use default choices for numeric embedding size
            vector_dim = train_dataset.dataset._data[train_dataset.vectordata_index].shape[1]  # total dimensionality of vector features
            prop_vector_features = train_dataset.num_vector_features() / float(train_dataset.num_features)  # Fraction of features that are numeric
            min_numeric_embed_dim = 32
            max_numeric_embed_dim = params['max_layer_width']
            params['numeric_embed_dim'] = int(min(max_numeric_embed_dim, max(min_numeric_embed_dim,
                                                    params['layers'][0]*prop_vector_features*np.log10(vector_dim+10) )))
        return

    def _fit(self, X_train, y_train, X_val=None, y_val=None, time_limit=None, num_cpus=1, num_gpus=0, reporter=None, **kwargs):
        """ X_train (pd.DataFrame): training data features (not necessarily preprocessed yet)
            X_val (pd.DataFrame): test data features (should have same column names as Xtrain)
            y_train (pd.Series):
            y_val (pd.Series): are pandas Series
            kwargs: Can specify amount of compute resources to utilize (num_cpus, num_gpus).
        """
        start_time = time.time()
        params = self.params.copy()
        self.verbosity = kwargs.get('verbosity', 2)
        params = fixedvals_from_searchspaces(params)
        if self.feature_metadata is None:
            #raise ValueError("Trainer class must set feature_metadata for this model")
            self.feature_metadata = FeatureMetadata.from_df(X_train)
        if num_cpus is not None:
            self.num_dataloading_workers = max(1, int(num_cpus/2.0))
        else:
            self.num_dataloading_workers = 1
        if self.num_dataloading_workers == 1:
            self.num_dataloading_workers = 0  # 0 is always faster and uses less memory than 1
        self.batch_size = params['batch_size']
        train_dataset, val_dataset = self.generate_datasets(X_train=X_train, y_train=y_train, params=params, X_val=X_val, y_val=y_val)
        logger.log(15, "Training data for neural network has: %d examples, %d features (%d vector, %d embedding, %d language)" %
              (train_dataset.num_examples, train_dataset.num_features,
               len(train_dataset.feature_groups['vector']), len(train_dataset.feature_groups['embed']),
               len(train_dataset.feature_groups['language']) ))
        # self._save_preprocessor()  # TODO: should save these things for hyperparam tunning. Need one HP tuner for network-specific HPs, another for preprocessing HPs.

        if num_gpus is not None and num_gpus >= 1:
            self.ctx = mx.gpu()  # Currently cannot use more than 1 GPU
        else:
            self.ctx = mx.cpu()
        self.get_net(train_dataset, params=params)

        if time_limit is not None:
            time_elapsed = time.time() - start_time
            time_limit_orig = time_limit
            time_limit = time_limit - time_elapsed
            if time_limit <= time_limit_orig * 0.4:  # if 60% of time was spent preprocessing, likely not enough time to train model
                raise TimeLimitExceeded

        self.train_net(train_dataset=train_dataset, params=params, val_dataset=val_dataset, initialize=True, setup_trainer=True, time_limit=time_limit, reporter=reporter)
        self.params_post_fit = params
        """
        # TODO: if we don't want to save intermediate network parameters, need to do something like saving in temp directory to clean up after training:
        with make_temp_directory() as temp_dir:
            save_callback = SaveModelCallback(self.model, monitor=self.metric, mode=save_callback_mode, name=self.name)
            with progress_disabled_ctx(self.model) as model:
                original_path = model.path
                model.path = Path(temp_dir)
                model.fit_one_cycle(self.epochs, self.lr, callbacks=save_callback)
                # Load the best one and export it
                model.load(self.name)
                print(f'Model validation metrics: {model.validate()}')
                model.path = original_path
        """

    def get_net(self, train_dataset, params):
        """ Creates a Gluon neural net and context for this dataset.
            Also sets up trainer/optimizer as necessary.
        """
        self.set_net_defaults(train_dataset, params)
        self.model = EmbedNet(train_dataset=train_dataset, params=params, num_net_outputs=self.num_net_outputs, ctx=self.ctx)

        # TODO: Below should not occur until at time of saving
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def train_net(self, train_dataset, params, val_dataset=None, initialize=True, setup_trainer=True, time_limit=None, reporter=None):
        """ Trains neural net on given train dataset, early stops based on test_dataset.
            Args:
                train_dataset (TabularNNDataset): training data used to learn network weights
                val_dataset (TabularNNDataset): validation data used for hyperparameter tuning
                initialize (bool): set = False to continue training of a previously trained model, otherwise initializes network weights randomly
                setup_trainer (bool): set = False to reuse the same trainer from a previous training run, otherwise creates new trainer from scratch
        """
        start_time = time.time()
        logger.log(15, "Training neural network for up to %s epochs..." % params['num_epochs'])
        seed_value = params.get('seed_value')
        if seed_value is not None:  # Set seed
            random.seed(seed_value)
            np.random.seed(seed_value)
            mx.random.seed(seed_value)
        if initialize:  # Initialize the weights of network
            logging.debug("initializing neural network...")
            self.model.collect_params().initialize(ctx=self.ctx)
            self.model.hybridize()
            logging.debug("initialized")
        if setup_trainer:
            # Also setup mxboard to monitor training if visualizer has been specified:
            visualizer = self.params_aux.get('visualizer', 'none')
            if visualizer == 'tensorboard' or visualizer == 'mxboard':
                try_import_mxboard()
                self.summary_writer = SummaryWriter(logdir=self.path, flush_secs=5, verbose=False)
            self.optimizer = self.setup_trainer(params=params, train_dataset=train_dataset)
        best_val_metric = -np.inf  # higher = better
        val_metric = None
        best_val_epoch = 0
        val_improve_epoch = 0  # most recent epoch where validation-score strictly improved
        num_epochs = params['num_epochs']
        if val_dataset is not None:
            y_val = val_dataset.get_labels()
        else:
            y_val = None

        if params['loss_function'] is None:
            params['loss_function'] = mx.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=self.model.from_logits)

        loss_func = params['loss_function']
        epochs_wo_improve = params['epochs_wo_improve']
        loss_scaling_factor = 1.0  # we divide loss by this quantity to stabilize gradients

        rescale_losses = {mx.gluon.loss.L1Loss: 'std', mx.gluon.loss.HuberLoss: 'std', mx.gluon.loss.L2Loss: 'var'}  # dict of loss names where we should rescale loss, value indicates how to rescale.
        loss_torescale = [key for key in rescale_losses if isinstance(loss_func, key)]
        if loss_torescale:
            loss_torescale = loss_torescale[0]
            if rescale_losses[loss_torescale] == 'std':
                loss_scaling_factor = np.std(train_dataset.get_labels())/5.0 + EPS  # std-dev of labels
            elif rescale_losses[loss_torescale] == 'var':
                loss_scaling_factor = np.var(train_dataset.get_labels())/5.0 + EPS  # variance of labels
            else:
                raise ValueError("Unknown loss-rescaling type %s specified for loss_func==%s" % (rescale_losses[loss_torescale], loss_func))

        if self.verbosity <= 1:
            verbose_eval = -1  # Print losses every verbose epochs, Never if -1
        elif self.verbosity == 2:
            verbose_eval = 50
        elif self.verbosity == 3:
            verbose_eval = 10
        else:
            verbose_eval = 1

        net_filename = self.path + self.temp_file_name
        if num_epochs == 0:  # use dummy training loop that stops immediately (useful for using NN just for data preprocessing / debugging)
            logger.log(20, "Not training Neural Net since num_epochs == 0.  Neural network architecture is:")
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with mx.autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(mx.nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                if batch_idx > 0:
                    break
            self.model.save_parameters(net_filename)
            logger.log(15, "untrained Neural Net saved to file")
            return

        start_fit_time = time.time()
        if time_limit is not None:
            time_limit = time_limit - (start_fit_time - start_time)

        # Training Loop:
        for e in range(num_epochs):
            if e == 0:  # special actions during first epoch:
                logger.log(15, "Neural network architecture:")
                logger.log(15, str(self.model))
            cumulative_loss = 0
            for batch_idx, data_batch in enumerate(train_dataset.dataloader):
                data_batch = train_dataset.format_batch_data(data_batch, self.ctx)
                with mx.autograd.record():
                    output = self.model(data_batch)
                    labels = data_batch['label']
                    loss = loss_func(output, labels) / loss_scaling_factor
                    # print(str(mx.nd.mean(loss).asscalar()), end="\r")  # prints per-batch losses
                loss.backward()
                self.optimizer.step(labels.shape[0])
                cumulative_loss += loss.sum()
            train_loss = cumulative_loss/float(train_dataset.num_examples)  # training loss this epoch
            if val_dataset is not None:
                val_metric = self.score(X=val_dataset, y=y_val, eval_metric=self.stopping_metric, metric_needs_y_pred=self.stopping_metric.needs_pred)
            if (val_dataset is None) or (val_metric >= best_val_metric) or (e == 0):  # keep training if score has improved
                if val_dataset is not None:
                    if not np.isnan(val_metric):
                        if val_metric > best_val_metric:
                            val_improve_epoch = e
                        best_val_metric = val_metric
                best_val_epoch = e
                # Until functionality is added to restart training from a particular epoch, there is no point in saving params without test_dataset
                if val_dataset is not None:
                    self.model.save_parameters(net_filename)
            if val_dataset is not None:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s, Val %s: %s" %
                      (e, train_loss.asscalar(), self.eval_metric_name, val_metric))
                if self.summary_writer is not None:
                    self.summary_writer.add_scalar(tag='val_'+self.eval_metric_name,
                                                   value=val_metric, global_step=e)
            else:
                if verbose_eval > 0 and e % verbose_eval == 0:
                    logger.log(15, "Epoch %s.  Train loss: %s" % (e, train_loss.asscalar()))
            if self.summary_writer is not None:
                self.summary_writer.add_scalar(tag='train_loss', value=train_loss.asscalar(), global_step=e)  # TODO: do we want to keep mxboard support?
            if reporter is not None:
                # TODO: Ensure reporter/scheduler properly handle None/nan values after refactor
                if val_dataset is not None and (not np.isnan(val_metric)):  # TODO: This might work without the if statement
                    # epoch must be number of epochs done (starting at 1)
                    reporter(epoch=e + 1,
                             validation_performance=val_metric,  # Higher val_metric = better
                             train_loss=float(train_loss.asscalar()),
                             eval_metric=self.eval_metric.name,
                             greater_is_better=self.eval_metric.greater_is_better)
            if e - val_improve_epoch > epochs_wo_improve:
                break  # early-stop if validation-score hasn't strictly improved in `epochs_wo_improve` consecutive epochs
            if time_limit is not None:
                time_elapsed = time.time() - start_fit_time
                time_epoch_average = time_elapsed / (e+1)
                time_left = time_limit - time_elapsed
                if time_left < time_epoch_average:
                    logger.log(20, f"\tRan out of time, stopping training early. (Stopping on epoch {e})")
                    break

        if val_dataset is not None:
            self.model.load_parameters(net_filename)  # Revert back to best model
            try:
                os.remove(net_filename)
            except FileNotFoundError:
                pass
        if val_dataset is None:
            logger.log(15, "Best model found in epoch %d" % best_val_epoch)
        else:  # evaluate one final time:
            final_val_metric = self.score(X=val_dataset, y=y_val, eval_metric=self.stopping_metric, metric_needs_y_pred=self.stopping_metric.needs_pred)
            if np.isnan(final_val_metric):
                final_val_metric = -np.inf
            logger.log(15, "Best model found in epoch %d. Val %s: %s" %
                  (best_val_epoch, self.eval_metric_name, final_val_metric))
        self.params_trained['num_epochs'] = best_val_epoch + 1
        return

    def _predict_proba(self, X, **kwargs):
        """ To align predict with abstract_model API.
            Preprocess here only refers to feature processing steps done by all AbstractModel objects,
            not tabularNN-specific preprocessing steps.
            If X is not DataFrame but instead TabularNNDataset object, we can still produce predictions,
            but cannot use preprocess in this case (needs to be already processed).
        """
        if isinstance(X, TabularNNDataset):
            return self._predict_tabular_data(new_data=X, process=False, predict_proba=True)
        elif isinstance(X, pd.DataFrame):
            X = self.preprocess(X, **kwargs)
            return self._predict_tabular_data(new_data=X, process=True, predict_proba=True)
        else:
            raise ValueError("X must be of type pd.DataFrame or TabularNNDataset, not type: %s" % type(X))

    def _predict_tabular_data(self, new_data, process=True, predict_proba=True):  # TODO ensure API lines up with tabular.Model class.
        """ Specific TabularNN method to produce predictions on new (unprocessed) data.
            Returns 1D numpy array unless predict_proba=True and task is multi-class classification (not binary).
            Args:
                new_data (pd.Dataframe or TabularNNDataset): new data to make predictions on.
                If you want to make prediction for just a single row of new_data, pass in: new_data.iloc[[row_index]]
                process (bool): should new data be processed (if False, new_data must be TabularNNDataset)
                predict_proba (bool): should we output class-probabilities (not used for regression)
        """
        if process:
            new_data = self.process_test_data(new_data, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference, labels=None)
        if not isinstance(new_data, TabularNNDataset):
            raise ValueError("new_data must of of type TabularNNDataset if process=False")
        if  not predict_proba:
            preds = mx.nd.zeros((new_data.num_examples,1))
        else:
            preds = mx.nd.zeros((new_data.num_examples, self.num_net_outputs))
        i = 0
        for batch_idx, data_batch in enumerate(new_data.dataloader):
            data_batch = new_data.format_batch_data(data_batch, self.ctx)
            preds_batch = self.model(data_batch)
            batch_size = len(preds_batch)
            if not predict_proba: # need to take argmax
                preds_batch = mx.nd.argmax(preds_batch, axis=1, keepdims=True)
            else: # need to take softmax
                preds_batch = mx.nd.softmax(preds_batch, axis=1)
            preds[i:(i+batch_size)] = preds_batch
            i = i+batch_size
        if not predict_proba:
            return preds.asnumpy().flatten()  # return 1D numpy array
        elif self.problem_type == 'binary' and predict_proba:
            #return preds[:,1].asnumpy()  # for binary problems, only return P(Y==+1)
            return preds.asnumpy()  # for binary problems, only return P(Y==+1)

        return preds.asnumpy()  # return 2D numpy array

    def generate_datasets(self, X_train, y_train, params, X_val=None, y_val=None):
        impute_strategy = params['proc.impute_strategy']
        max_category_levels = params['proc.max_category_levels']
        skew_threshold = params['proc.skew_threshold']
        embed_min_categories = params['proc.embed_min_categories']
        use_ngram_features = params['use_ngram_features']

        if isinstance(X_train, TabularNNDataset):
            train_dataset = X_train
        else:
            X_train = self.preprocess(X_train)
            if self.features is None:
                self.features = list(X_train.columns)
            train_dataset = self.process_train_data(
                df=X_train, labels=y_train, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers,
                impute_strategy=impute_strategy, max_category_levels=max_category_levels, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features,
            )
        if X_val is not None:
            if isinstance(X_val, TabularNNDataset):
                val_dataset = X_val
            else:
                X_val = self.preprocess(X_val)
                val_dataset = self.process_test_data(df=X_val, labels=y_val, batch_size=self.batch_size, num_dataloading_workers=self.num_dataloading_workers_inference)
        else:
            val_dataset = None
        return train_dataset, val_dataset

    def process_test_data(self, df, batch_size, num_dataloading_workers, labels=None):
        """ Process train or test DataFrame into a form fit for neural network models.
        Args:
            df (pd.DataFrame): Data to be processed (X)
            labels (pd.Series): labels to be processed (y)
            test (bool): Is this test data where each datapoint should be processed separately using predetermined preprocessing steps.
                         Otherwise preprocessor uses all data to determine propreties like best scaling factors, number of categories, etc.
        Returns:
            Dataset object
        """
        warnings.filterwarnings("ignore", module='sklearn.preprocessing') # sklearn processing n_quantiles warning
        if labels is not None and len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")
        if (self.processor is None or self._types_of_features is None
           or self.feature_arraycol_map is None or self.feature_type_map is None):
            raise ValueError("Need to process training data before test data")
        if self.features_to_drop:
            drop_cols = [col for col in df.columns if col in self.features_to_drop]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        df = self.processor.transform(df) # 2D numpy array. self.feature_arraycol_map, self.feature_type_map have been previously set while processing training data.
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=True)

    def process_train_data(self, df, batch_size, num_dataloading_workers, impute_strategy, max_category_levels, skew_threshold, embed_min_categories, use_ngram_features, labels):
        """ Preprocess training data and create self.processor object that can be used to process future data.
            This method should only be used once per TabularNeuralNetModel object, otherwise will produce Warning.
        # TODO no label processing for now
        # TODO: language features are ignored for now
        # TODO: add time/ngram features
        # TODO: no filtering of data-frame columns based on statistics, e.g. categorical columns with all unique variables or zero-variance features.
                This should be done in default_learner class for all models not just TabularNeuralNetModel...
        """
        warnings.filterwarnings("ignore", module='sklearn.preprocessing')  # sklearn processing n_quantiles warning
        if set(df.columns) != set(self.features):
            raise ValueError("Column names in provided Dataframe do not match self.features")
        if labels is None:
            raise ValueError("Attempting process training data without labels")
        if len(labels) != len(df):
            raise ValueError("Number of examples in Dataframe does not match number of labels")

        self._types_of_features, df = self._get_types_of_features(df, skew_threshold=skew_threshold, embed_min_categories=embed_min_categories, use_ngram_features=use_ngram_features)  # dict with keys: : 'continuous', 'skewed', 'onehot', 'embed', 'language', values = column-names of df
        logger.log(15, "AutoGluon Neural Network infers features are of the following types:")
        logger.log(15, json.dumps(self._types_of_features, indent=4))
        logger.log(15, "\n")
        self.processor = self._create_preprocessor(impute_strategy=impute_strategy, max_category_levels=max_category_levels)
        df = self.processor.fit_transform(df) # 2D numpy array
        self.feature_arraycol_map = self._get_feature_arraycol_map(max_category_levels=max_category_levels)  # OrderedDict of feature-name -> list of column-indices in df corresponding to this feature
        num_array_cols = np.sum([len(self.feature_arraycol_map[key]) for key in self.feature_arraycol_map])  # should match number of columns in processed array
        if num_array_cols != df.shape[1]:
            raise ValueError("Error during one-hot encoding data processing for neural network. Number of columns in df array does not match feature_arraycol_map.")

        self.feature_type_map = self._get_feature_type_map()  # OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language')
        return TabularNNDataset(df, self.feature_arraycol_map, self.feature_type_map,
                                batch_size=batch_size, num_dataloading_workers=num_dataloading_workers,
                                problem_type=self.problem_type, labels=labels, is_test=False)

    def setup_trainer(self, params, train_dataset=None):
        """ Set up optimizer needed for training.
            Network must first be initialized before this.
        """
        optimizer_opts = {'learning_rate': params['learning_rate'], 'wd': params['weight_decay'], 'clip_gradient': params['clip_gradient']}
        if 'lr_scheduler' in params and params['lr_scheduler'] is not None:
            if train_dataset is None:
                raise ValueError("train_dataset cannot be None when lr_scheduler is specified.")
            base_lr = params.get('base_lr', 1e-6)
            target_lr = params.get('target_lr', 1.0)
            warmup_epochs = params.get('warmup_epochs', 10)
            lr_decay = params.get('lr_decay', 0.1)
            lr_mode = params['lr_scheduler']
            num_batches = train_dataset.num_examples // params['batch_size']
            lr_decay_epoch = [max(warmup_epochs, int(params['num_epochs']/3)), max(warmup_epochs+1, int(params['num_epochs']/2)),
                              max(warmup_epochs+2, int(2*params['num_epochs']/3))]
            lr_scheduler = LRSequential([
                LRScheduler('linear', base_lr=base_lr, target_lr=target_lr, nepochs=warmup_epochs, iters_per_epoch=num_batches),
                LRScheduler(lr_mode, base_lr=target_lr, target_lr=base_lr, nepochs=params['num_epochs'] - warmup_epochs,
                            iters_per_epoch=num_batches, step_epoch=lr_decay_epoch, step_factor=lr_decay, power=2)
            ])
            optimizer_opts['lr_scheduler'] = lr_scheduler
        if params['optimizer'] == 'sgd':
            if 'momentum' in params:
                optimizer_opts['momentum'] = params['momentum']
            optimizer = mx.gluon.Trainer(self.model.collect_params(), 'sgd', optimizer_opts)
        elif params['optimizer'] == 'adam':  # TODO: Can we try AdamW?
            optimizer = mx.gluon.Trainer(self.model.collect_params(), 'adam', optimizer_opts)
        else:
            raise ValueError("Unknown optimizer specified: %s" % params['optimizer'])
        return optimizer

    def _get_feature_arraycol_map(self, max_category_levels):
        """ Returns OrderedDict of feature-name -> list of column-indices in processed data array corresponding to this feature """
        feature_preserving_transforms = set(['continuous','skewed', 'ordinal', 'language'])  # these transforms do not alter dimensionality of feature
        feature_arraycol_map = {}  # unordered version
        current_colindex = 0
        for transformer in self.processor.transformers_:
            transformer_name = transformer[0]
            transformed_features = transformer[2]
            if transformer_name in feature_preserving_transforms:
                for feature in transformed_features:
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    feature_arraycol_map[feature] = [current_colindex]
                    current_colindex += 1
            elif transformer_name == 'onehot':
                oh_encoder = [step for (name, step) in transformer[1].steps if name == 'onehot'][0]
                for i in range(len(transformed_features)):
                    feature = transformed_features[i]
                    if feature in feature_arraycol_map:
                        raise ValueError("same feature is processed by two different column transformers: %s" % feature)
                    oh_dimensionality = min(len(oh_encoder.categories_[i]), max_category_levels+1)
                    feature_arraycol_map[feature] = list(range(current_colindex, current_colindex+oh_dimensionality))
                    current_colindex += oh_dimensionality
            else:
                raise ValueError("unknown transformer encountered: %s" % transformer_name)
        return OrderedDict([(key, feature_arraycol_map[key]) for key in feature_arraycol_map])

    def _get_feature_type_map(self):
        """ Returns OrderedDict of feature-name -> feature_type string (options: 'vector', 'embed', 'language') """
        if self.feature_arraycol_map is None:
            raise ValueError("must first call _get_feature_arraycol_map() before _get_feature_type_map()")
        vector_features = self._types_of_features['continuous'] + self._types_of_features['skewed'] + self._types_of_features['onehot']
        feature_type_map = OrderedDict()
        for feature_name in self.feature_arraycol_map:
            if feature_name in vector_features:
                feature_type_map[feature_name] = 'vector'
            elif feature_name in self._types_of_features['embed']:
                feature_type_map[feature_name] = 'embed'
            elif feature_name in self._types_of_features['language']:
                feature_type_map[feature_name] = 'language'
            else:
                raise ValueError("unknown feature type encountered")
        return feature_type_map

    def _create_preprocessor(self, impute_strategy, max_category_levels):
        """ Defines data encoders used to preprocess different data types and creates instance variable which is sklearn ColumnTransformer object """
        if self.processor is not None:
            Warning("Attempting to process training data for TabularNeuralNetModel, but previously already did this.")
        continuous_features = self._types_of_features['continuous']
        skewed_features = self._types_of_features['skewed']
        onehot_features = self._types_of_features['onehot']
        embed_features = self._types_of_features['embed']
        language_features = self._types_of_features['language']
        transformers = []  # order of various column transformers in this list is important!
        if continuous_features:
            continuous_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('scaler', StandardScaler())])
            transformers.append( ('continuous', continuous_transformer, continuous_features) )
        if skewed_features:
            power_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=impute_strategy)),
                ('quantile', QuantileTransformer(output_distribution='normal')) ])  # Or output_distribution = 'uniform'
            transformers.append( ('skewed', power_transformer, skewed_features) )
        if onehot_features:
            onehot_transformer = Pipeline(steps=[
                # TODO: Consider avoiding converting to string for improved memory efficiency
                ('to_str', FunctionTransformer(convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('onehot', OneHotMergeRaresHandleUnknownEncoder(max_levels=max_category_levels, sparse=False))])  # test-time unknown values will be encoded as all zeros vector
            transformers.append( ('onehot', onehot_transformer, onehot_features) )
        if embed_features:  # Ordinal transformer applied to convert to-be-embedded categorical features to integer levels
            ordinal_transformer = Pipeline(steps=[
                ('to_str', FunctionTransformer(convert_df_dtype_to_str)),
                ('imputer', SimpleImputer(strategy='constant', fill_value=self.unique_category_str)),
                ('ordinal', OrdinalMergeRaresHandleUnknownEncoder(max_levels=max_category_levels))])  # returns 0-n when max_category_levels = n-1. category n is reserved for unknown test-time categories.
            transformers.append( ('ordinal', ordinal_transformer, embed_features) )
        if language_features:
            raise NotImplementedError("language_features cannot be used at the moment")
        return ColumnTransformer(transformers=transformers)  # numeric features are processed in the same order as in numeric_features vector, so feature-names remain the same.

    def save(self, path: str = None, verbose=True) -> str:
        if self.model is not None:
            self._architecture_desc = self.model.architecture_desc
        temp_model = self.model
        temp_sw = self.summary_writer
        self.model = None
        self.summary_writer = None
        path_final = super().save(path=path, verbose=verbose)
        self.model = temp_model
        self.summary_writer = temp_sw
        self._architecture_desc = None

        # Export model
        if self.model is not None:
            params_filepath = path_final + self.params_file_name
            # TODO: Don't use os.makedirs here, have save_parameters function in tabular_nn_model that checks if local path or S3 path
            os.makedirs(os.path.dirname(path_final), exist_ok=True)
            self.model.save_parameters(params_filepath)
        return path_final

    @classmethod
    def load(cls, path: str, reset_paths=True, verbose=True):
        model: TabularNeuralNetModel = super().load(path=path, reset_paths=reset_paths, verbose=verbose)
        if model._architecture_desc is not None:
            model.model = EmbedNet(architecture_desc=model._architecture_desc, ctx=model.ctx)  # recreate network from architecture description
            model._architecture_desc = None
            # TODO: maybe need to initialize/hybridize?
            model.model.load_parameters(model.path + model.params_file_name, ctx=model.ctx)
            model.summary_writer = None
        return model

    def get_info(self):
        info = super().get_info()
        info['hyperparameters_post_fit'] = self.params_post_fit
        return info

    def reduce_memory_size(self, remove_fit=True, requires_save=True, **kwargs):
        super().reduce_memory_size(remove_fit=remove_fit, requires_save=requires_save, **kwargs)
        if remove_fit and requires_save:
            self.optimizer = None

    def _get_default_stopping_metric(self):
        return self.eval_metric


def convert_df_dtype_to_str(df):
    return df.astype(str)

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
        model = TabularNeuralNetModel(path='/tmp/', problem_type=problem_type, metric=accuracy, name='tabular_nn', eval_metric=accuracy)
        model.fit(X_train=X_train, y_train=y_train)
        print(model.predict_proba(X_test).shape)
        print(accuracy(y_test.to_numpy().astype(int), model.predict(X_test)))
