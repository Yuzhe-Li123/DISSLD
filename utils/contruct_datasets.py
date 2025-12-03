import numpy as np
from scipy.optimize import linear_sum_assignment

from data.load_data_v2 import _get_mask
from numpy.random import randint, shuffle, permutation
from scipy.io import savemat, loadmat

import os
import sys


def realign(features_u):
    anchor = features_u[0]
    similarity = [np.sum(np.square(np.expand_dims(anchor, axis=1) - t), axis=2) for t in features_u[1:]]
    p_index = [linear_sum_assignment(s, maximize=False)[-1] for s in similarity]
    P = [np.eye(len(anchor))]
    for pi in p_index:
        P.append(np.eye(len(anchor))[pi])
    return P


def construct_cmp2dip(model, loader, miss_rate=0):
    data_u = loader.data_u
    if miss_rate:
        mask = _get_mask(len(data_u), len(data_u[0]), miss_rate).T
        for view, (x, m) in enumerate(zip(data_u, mask)):
            index_miss = np.where(m < 1)[0]
            index_observed = np.where(m == 1)[0]
            avg = np.mean(x[index_observed], axis=0)
            x[index_miss] = avg
    features_u = model.encoder.predict(data_u)

    P = realign(features_u)
    Pu = []
    for p, x in zip(P, data_u):
        Pu.append(p @ x)

    X = np.empty((len(Pu),), dtype=np.ndarray)
    Y = np.concatenate([loader.Yp[0], loader.Yu[0]])
    for i, x in enumerate(zip(loader.data_p, Pu)):
        X[i] = np.concatenate(x, axis=0)
    return X, Y


def construct_pvp2dip(data_name, save_dir, data, label,
                      paired_rate, miss_rate, dim_first=False):
    # template = {
    #     'X': None, 'Y': None,
    #     'train_X': None, 'train_Y': None,
    #     'test_X': None, 'test_Y': None}
    assert len(data[0]) == len(data[1])
    size = len(data[0])
    num_paired = int(size * paired_rate)
    index = permutation(size)
    X = [x[index] for x in data]
    Y = [y[index] for y in label]
    index = permutation(size)

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for x, y in zip(X, Y):
        train_index = index[:num_paired]
        test_index = index[num_paired:]

        train_X.append(x[train_index])
        train_Y.append(y[train_index])

        test_X.append(x[test_index])
        test_Y.append(y[test_index])

    mask = _get_mask(miss_rate, int(size-num_paired), len(X)).T
    for view, (x, m) in enumerate(zip(test_X, mask)):
        index_miss = np.where(m < 1)[0]
        index_observed = np.where(m == 1)[0]
        avg = np.mean(x[index_observed], axis=0)
        # print('avg:', avg)
        x[index_miss] = avg

    data = {'X': X, 'Y': Y, 'train_X': train_X, 'train_Y': train_Y,
            'test_X': test_X, 'test_Y': test_Y}

    for k, v in data.items():
        cell = np.empty((len(v), ), dtype=np.ndarray)
        for i, c in enumerate(v):
            c = c.astype(np.float64)
            cell[i] = c.T if dim_first else c
        data.update({k: cell})

    savemat(save_dir + f'{data_name}.mat', data)


def padding_with_average(dataname, save_dir, missbar: list, data: list, label: list,
                         dim_first=True, mark_x=None, mark_y=None):
    assert len(data[0]) == len(data[1])
    for mr in missbar:
        X = np.empty((len(data), ), dtype='object')
        Y = np.empty((len(data), ), dtype='object')
        mask = _get_mask(mr, len(data[0]), len(data)).T
        for view, (x, m) in enumerate(zip(data, mask)):
            index_miss = np.where(m < 1)[0]
            index_observed = np.where(m == 1)[0]
            avg = np.mean(x[index_observed], axis=0)
            x = x.copy().astype(np.float64)
            x[index_miss] = avg
            X[view] = x.T if dim_first else x
            Y[view] = label[view]
        savemat(save_dir + "{}.mat".format(dataname), {
            mark_x if mark_x else 'X': X,
            mark_y if mark_y else 'Y': Y
        })


def construct_pip(dataname, save_dir, missbar: list, data: list, label: list,
                  nfolds=5, mode=2, dim_first=True, mark_x=None, mark_y=None):
    assert len(data[0]) == len(data[1])
    for mr in missbar:
        X = np.empty((len(data), ), dtype='object')
        Y = np.empty((len(data), ), dtype='object')
        index = np.empty((len(data), ), dtype='object')
        mask = _get_mask(mr, len(data[0]), len(data)).T

        for v in range(len(data)):
            # X[v]:(n_dim, n_num)
            if dim_first:
                X[v] = data[v].astype(np.double).T
            else:
                X[v] = data[v].astype(np.double)
            Y[v] = label[v].astype(np.double)
            index[v] = np.where(mask[v] == 1)[0] + 1

        save_dict = {
            mark_x if mark_x else 'X': X,
            mark_y if mark_y else 'Y': Y,
            'index': index
        }

        if mode == 1:
            folds = [permutation(len(data[0])).reshape(1, -1)+1 for i in range(nfolds)]
            save_dict.update({'folds': np.concatenate(folds, axis=0)})
        savemat(save_dir + '/{}_Per{}.mat'.format(dataname, mr), save_dict)

        if mode == 2:
            folds = np.empty((nfolds,), dtype='object')
            folds[0] = mask
            for i in range(nfolds - 1):
                folds[i + 1] = _get_mask(mr, len(data[0]), len(data)).T
            # save_dir = os.path.splitext(save_dir)[0]
            savemat(save_dir + '{}_percentDel_{}.mat'.format(dataname, str(mr)), {
                'folds': folds
            })
