import numpy as np
import scipy.io as scio
import sys

import warnings
warnings.filterwarnings("ignore")

path = sys.path[0] + '/datasets'

sae = ['2V_Caltech', '6V_Caltech', 'Scene15', 'YouTube_X',
       'BDGP', 'Handwritten', 'ALOI100', 'ThreeSources']
cae = ['MNIST_UPS', 'Fashion_MV', 'Noisy-MNIST']

config = {
    'Caltech_2v': (),
    'Caltech101_20': (),
    'BDGP': (0, 1),
    'YouTube_X': (0, 4),
    'Scene15': (0, 1),
    'Reuters': (0, 1, 2),
    'ALOI100': (0, 2),
    # 'Handwritten': (3, 5, 2, 1, 4, 0),
    'Handwritten': (),
    'Fashion_MV': (),
    'MNIST_UPS': ()
}


def read_X(Data, label_mark='Y'):
    X = list(Data['X'].reshape(-1))
    X = [np.array(x) for x in X]
    if X[0].shape[0] != X[1].shape[0]:
        X = [x.T for x in X]
        assert X[0].shape[0] == X[1].shape[0]
    n_views = len(X)
    Y = Data[label_mark]
    Y = Y.reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y - 1

    return X, [Y] * n_views


def read_Xv(Data, label_mark='Y'):
    X = [Data[key] for key in Data.keys() if key[0].lower() in ['x']]
    if X[0].shape[0] != X[1].shape[0]:
        X = [x.T for x in X]
        assert X[0].shape[0] == X[1].shape[0]
    n_views = len(X)
    Y = Data[label_mark]
    Y = Y.reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y - 1
    return X, [Y] * n_views


def Caltech_2v(path_=None, filter_=config.get('Caltech101_20', False)):
    # Different IMVC methods might utilize different data preprocessing functions,
    # including but not limited to standardization, regularization, and min-max normalization, etc.
    # Upload the dataset, which is already be pre-processed.
    # distribution of class:
    #   [0.183, 0.084, 0.336, 0.013, 0.041, 0.021, 0.051, 0.021, 0.028,
    #   0.014, 0.022, 0.019, 0.024, 0.014, 0.019, 0.027, 0.015, 0.023, 0.016, 0.025]
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/Caltech_2v.mat")

    return read_Xv(Data)


def Caltech101_20(path_=None, filter_=config.get('Caltech101_20', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/Caltech_6v.mat")
    if 'X' in Data.keys():
        X, Y = read_X(Data)
    else:
        X, Y = read_Xv(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def BDGP(path_=None, filter_=config.get('BDGP', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/BDGP.mat")
    if 'X' in Data.keys():
        X, Y = read_X(Data, 'gt')
    else:
        X, Y = read_Xv(Data, 'gt')
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def YouTube_X(path_=None, filter_=config.get('YouTuce_X', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/YouTube_X.mat")
    X, Y = read_X(Data, 'gt')
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def Scene15(path_=None, filter_=config.get('Scene15', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/Scene15.mat")
    X, Y = read_X(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def Reuters(path_=None, filter_=config.get('Reuters', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/Reuters_dim10.mat")
    if 'X' in Data.keys():
        X, Y = read_X(Data)
    else:
        X, Y = read_Xv(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def ALOI100(path_=None, filter_=config.get('ALOI100', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/ALOI-100.mat")
    X, Y = read_X(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def Handwritten(path_=None,  filter_=config.get('Handwritten', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/Handwritten_numerals.mat")
    X, Y = read_X(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def Fashion_MV(path_=None,  filter_=config.get('Fashion_MV', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/Fashion_MV_3v.mat")
    X, Y = read_Xv(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y


def MNIST_UPS(path_=None, filter_=config.get('MNIST_UPS', False)):
    if path_:
        Data = scio.loadmat(path_)
    else:
        Data = scio.loadmat(path + "/MNIST_USPS_2v.mat")
    X, Y = read_Xv(Data)
    if filter_ is not False:
        if len(filter_):
            return [X[v] for v in filter_], [Y[v] for v in filter_]
        else:
            return X, Y
