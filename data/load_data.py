import numpy as np
import scipy.io as scio
from numpy.random import randint, shuffle, permutation
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sys

import warnings
warnings.filterwarnings("ignore")

from data.read_mat import Scene15 as Scene15_, Caltech_2v as Caltech_2v_

path = sys.path[0] + '/datasets'

sae = ['2V_Caltech', '6V_Caltech', 'Scene15', 'YouTube_X',
       'BDGP', 'Handwritten', 'ALOI100', 'ThreeSources']
cae = ['MNIST_UPS', 'Fashion_MV', 'Noisy-MNIST']


def normalize_(X):
    min_max_scaler = MinMaxScaler()
    if isinstance(X, (list, tuple)):
        X = [min_max_scaler.fit_transform(x) for x in X]
        return X
    elif isinstance(X, np.ndarray):
        return min_max_scaler.fit_transform(X)


class DataLoader(object):
    def __init__(self, batch_size, normalized, pairedrate, missrate, dataset):
        # data_paired: (data, index)
        # data_unpaired: (data, index, mask)
        # Y: list, [ndarray] * n_views

        # data to be hold
        self.label, self.data, (self.Gp, self.index_u), self.mask = load_data_conv(
            dataset, normalized, pairedrate, missrate)

        self.normalized = normalized
        # dataset related
        self.n_views = len(self.data)
        self.n_classes = len(np.unique(self.label[0]))
        self.size = len(self.label[0])
        self.view_shapes = [x.shape[1:] for x in self.data]

        # mask: (n_view, incomplete_size)
        if self.mask.shape[1] == self.size:
            # fully aligned, imc or mvc
            self.mode = 'fa'
            self.mask = self.mask[:, self.Gp.shape[1]:]
        else:
            # partially aligned, dic or pvc
            self.mode = 'pa'

        self.split_ind = self.Gp.shape[1]
        batch_size = self.split_ind \
            if batch_size > self.split_ind else batch_size

        # incomplete data
        if self.Gu is not None:
            batch_size = self.split_ind \
                if batch_size > len(self.Gu[0]) else batch_size
        self.batch_size = batch_size

        self.pairedrate = pairedrate
        self.missrate = missrate
        self.dataset = dataset

        # summary
        print('prepared data:{} ; clusters:{} ; size:{} ; batch_size:{}'
              .format(self.dataset, self.n_classes, self.size, self.batch_size))

    def gen_batch(self, P, Pu=None, confp=None, confu=None):
        assert len(P) == len(self.Gp[0])
        if Pu is not None:
            assert self.Gu is not None
            assert len(Pu) == len(self.Gu)
            P = [np.concatenate([P, pu], axis=0) for pu in Pu]
            data = [np.concatenate([xp, xu], axis=0) for xp, xu in zip(self.data_p, self.data_u)]
            num_train = len(data[0])
        else:
            P = [P.copy()] * self.n_views
            data = self.copy(self.data_p)
            num_train = len(data[0])

        if confp is not None:
            if confu is not None:
                assert isinstance(confu, (tuple, list))
                conf = [np.concatenate([confp, cu], axis=0) for cu in confu]
            else:
                conf = [confp]*len(P)
        else:
            conf = None

        index = np.arange(num_train)
        n_batch = int(num_train / self.batch_size) + 1
        # while True:
        for batch_i in range(n_batch):
            one_batch_index = index[batch_i * self.batch_size: min((batch_i + 1) * self.batch_size, num_train)]
            out = []
            for d, target in zip(data, P):
                out.append(target[one_batch_index])
                out.append(d[one_batch_index])

            sample_weight = [cf[one_batch_index] for cf in conf] if conf else None
            yield [d[one_batch_index] for d in data], out, sample_weight

    @staticmethod
    def copy(item):
        if isinstance(item, list):
            assert isinstance(item[0], np.ndarray)
            return [i.copy() for i in item]
        elif isinstance(item, np.ndarray):
            return item.copy()
        else:
            return None

    @property
    def Yp(self):
        return [y[ind].copy() for y, ind in zip(self.label, self.Gp)]

    @property
    def input_p(self):
        return {'input' + str(v + 1): x[ind].copy() for v, (x, ind) in enumerate(zip(self.data, self.Gp))}

    @property
    def data_p(self):
        return [x[ind].copy() for v, (x, ind) in enumerate(zip(self.data, self.Gp))]

    @property
    def Gu(self):
        if self.index_u is None:
            return None
        else:
            assert self.mask.shape == self.index_u.shape
            # mask = np.zeros_like(self.mask)
            # for v, (ind, m) in enumerate(zip(self.index_u, self.mask)):
            #     mask[v] = m[ind - self.split_ind]
            tmp = self.index_u * self.mask
            return [ind[np.nonzero(t)[0]] for ind, t in zip(self.index_u, tmp)]

    @property
    def Yu(self):
        if self.index_u is None:
            return None
        else:
            return [y[ind].copy() for y, ind in zip(self.label, self.Gu)]

    @property
    def input_u(self):
        if self.index_u is None:
            return None
        else:
            # index_u =
            return {'input' + str(v + 1): x[ind].copy() for v, (x, ind) in
                    enumerate(zip(self.data, self.Gu))}

    @property
    def data_u(self):
        if self.index_u is None:
            return None
        else:
            return [x[ind].copy() for v, (x, ind) in enumerate(zip(self.data, self.Gu))]

    @property
    def pretraining_data(self):
        if self.Gu:
            data = [np.concatenate([xp, xu], axis=0) for xp, xu in zip(self.data_p, self.data_u)]
        else:
            data = self.copy(self.data_p)
        return data


def load_data_conv(dataset, normalization, pairedrate, missrate):

    norm = normalize_ if normalization else None
    if dataset == 'Caltech101_20':
        X, Y = Caltech_2v_()
        if normalization:
            X = normalize_(X)
    elif dataset == 'BDGP':
        X, Y = BDGP(pairedrate, missrate, norm)
    elif dataset == 'Scene15':
        X, Y = Scene15_()
        if normalization:
            X = normalize_(X)
    elif dataset == 'YouTube_X':
        X, Y = YouTube_X(pairedrate, missrate, norm)
    elif dataset == 'Handwritten':
        X, Y = Handwritten(pairedrate, missrate, norm)
    elif dataset == 'ALOI100':
        X, Y = ALOI100(pairedrate, missrate, norm)
    elif dataset == 'MNIST_UPS':
        X, Y = MNIST_UPS(pairedrate, missrate, norm)
    elif dataset == 'Reuters':
        X, Y = Reuters(pairedrate, missrate, norm)
    else:
        raise ValueError('Not defined for loading %s' % dataset)

    return construct_dataset(pairedrate, missrate, X, Y)


def Caltech_2v(pairedrate, missrate, Normalize):
    # Different IMVC methods might utilize different data preprocessing functions,
    # including but not limited to standardization, regularization, and min-max normalization, etc.
    # Upload the dataset, which is already be pre-processed.
    # distribution of class:
    #   [0.183, 0.084, 0.336, 0.013, 0.041, 0.021, 0.051, 0.021, 0.028,
    #   0.014, 0.022, 0.019, 0.024, 0.014, 0.019, 0.027, 0.015, 0.023, 0.016, 0.025]
    Data = scio.loadmat(path + "/Caltech_2v.mat")
    n_views = 0
    for k in Data.keys():
        if k[0].lower() in ['x']:
            n_views += 1
    X = []
    for v in range(n_views):
        X.append(Data['X' + str(v + 1)])
    Y = Data['Y']
    Y = Y.reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y-1
    size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def Caltech_6v(pairedrate, missrate, Normalize):
    Data = scio.loadmat(path + "/Caltech_6v.mat")
    if 'X' in Data.keys():
        X = list(Data['X'].reshape(-1))
        n_views = len(X)
        Y = Data['Y']
        Y = Y.reshape(-1).astype(np.int64)
        if np.min(Y) == 1:
            Y = Y - 1
        size = Y.shape[0]
    else:
        n_views = 0
        for k in Data.keys():
            if k[0].lower() in ['x']:
                n_views += 1
        X = []
        for v in range(n_views):
            X.append(Data['X' + str(v + 1)])
        Y = Data['Y']
        Y = Y.reshape(-1).astype(np.int64)
        if np.min(Y) == 1:
            Y = Y - 1
        size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y] * n_views


def BDGP(pairedrate, missrate, Normalize):
    Data = scio.loadmat(path + "/BDGP.mat")
    if 'X' in Data.keys():
        X = [np.array(x).T for x in Data['X'].reshape(-1)]
        n_views = len(X)
        # x1 = [0][0].T
        # x2 = Data['X'][0][1].T
        Y = Data['gt']
        Y = Y.reshape(-1).astype(np.int64)
        if np.min(Y) == 1:
            Y = Y - 1
        size = Y.shape[0]
    else:
        n_views = 0
        for k in Data.keys():
            if k[0].lower() in ['x']:
                n_views += 1
        X = []
        for v in range(n_views):
            X.append(Data['X' + str(v + 1)])
        Y = Data['Y']
        Y = Y.reshape(-1).astype(np.int64)
        if np.min(Y) == 1:
            Y = Y - 1
        size = Y.shape[0]

    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def YouTube_X(pairedrate, missrate, Normalize, two_view=True):
    Data = scio.loadmat(path + "/YouTube_X.mat")
    X = list(Data['X'].reshape(-1))
    Y = Data['gt'].reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y - 1
    if two_view:
        X = [X[0], X[4]]
    n_views = len(X)
    size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def Scene15(pairedrate, missrate, Normalize, two_view=True):
    Data = scio.loadmat(path + "/Scene15.mat")
    X = list(Data['X'].reshape(-1))
    for v, x in enumerate(X):
        X[v] = np.array(x)
    Y = Data['Y'].reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y-1
    if two_view:
        X = X[:2]
    n_views = len(X)
    size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def Reuters(pairedrate, missrate, Normalize, two_view=True):
    Data = scio.loadmat(path + "/Reuters_dim10.mat")
    if 'X' in Data.keys():
        X = list(Data['X'].reshape(-1))
        for v, x in enumerate(X):
            X[v] = np.array(x)
        if two_view:
            X = X[:2]
        n_views = len(X)
        Y = Data['Y']
        Y = Y.reshape(-1).astype(np.int64)
        if np.min(Y) == 1:
            Y = Y - 1
        size = Y.shape[0]
    else:
        n_views = 0
        for k in Data.keys():
            if k[0].lower() in ['x']:
                n_views += 1
        X = []
        for v in range(n_views):
            X.append(Data['X' + str(v + 1)])
        if two_view:
            X = X[:2]
            n_views = 2
        Y = Data['Y']
        Y = Y.reshape(-1).astype(np.int64)
        if np.min(Y) == 1:
            Y = Y - 1
        size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def ALOI100(pairedrate, missrate, Normalize, two_view=True):
    Data = scio.loadmat(path + "/ALOI-100.mat")
    X = list(Data['X'].reshape(-1))
    Y = Data['Y'].reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y - 1
    if two_view:
        X = [X[0], X[2]]
    n_views = len(X)
    size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def Handwritten(pairedrate, missrate, Normalize):
    Data = scio.loadmat(path + "/Handwritten_numerals.mat")
    X = list(Data['X'].reshape(-1))
    Y = Data['Y'].reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y-1
    X = [X[2], X[4]]
    n_views = len(X)
    size = Y.shape[0]
    X = Normalize(X) if Normalize else X
    return X, [Y]*n_views


def Fashion_MV(pairedrate, missrate, Normalize):
    Data = scio.loadmat(path + "/Fashion_MV_3v.mat")
    n_views = 0
    for k in Data.keys():
        if k[0].lower() in ['x']:
            n_views += 1
    X = []
    for v in range(n_views):
        X.append(Data['X' + str(v + 1)])
    Y = Data['Y']
    Y = Y.reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y - 1
    size = Y.shape[0]
    return X, [Y]*n_views


def MNIST_UPS(pairedrate, missrate, Normalize):
    Data = scio.loadmat(path + "/MNIST_USPS_2v.mat")
    n_views = 0
    for k in Data.keys():
        if k[0].lower() in ['x']:
            n_views += 1
    X = []
    for v in range(n_views):
        X.append(Data['X' + str(v + 1)])
    Y = Data['Y']
    Y = Y.reshape(-1).astype(np.int64)
    if np.min(Y) == 1:
        Y = Y - 1
    size = Y.shape[0]
    return X, [Y]*n_views


def construct_dataset(pairedrate, missrate, X: list, Y: list, random=False):
    assert len(X) == len(Y)
    size = len(Y[0])
    for y in Y[1:]:
        assert size == len(y)
    num_views = len(X)
    t = permutation(size)
    X = [x[t] for x in X]
    Y = [y[t] for y in Y]
    index = np.arange(size, dtype=int)
    num_paired = int(size * pairedrate)

    if num_paired == size:
        # imc or mvc
        mask, num_missing = construct_incomplete_data(missrate, (size, num_views))
        # len(mask) = len(index0) + len(index_u)
        if num_missing:
            # imvc
            index0 = np.expand_dims(index[:-num_missing], 0).repeat(num_views, axis=0)
            index_u = np.expand_dims(index[-num_missing:], 0).repeat(num_views, axis=0)
            return Y, X, (index0, index_u), mask
        else:
            # mvc
            index = np.expand_dims(index, 0).repeat(num_views, axis=0)
            return Y, X, (index, None), mask
    else:
        # pvc, dimvc
        mask, _ = construct_incomplete_data(missrate, (size-num_paired, num_views))
        if random:
            index_u = []
            for v, m in enumerate(mask):
                t = permutation(size - num_paired)
                index_u.append(index[num_paired:][t].reshape(1, -1))
                mask[v] = m[t]
            index0 = np.expand_dims(index[:num_paired], 0).repeat(num_views, axis=0)
            index_u = np.concatenate(index_u, axis=0)
            return Y, X, (index0, index_u), mask
        else:
            index0 = np.expand_dims(index[:num_paired], 0).repeat(num_views, axis=0)
            index_u = np.expand_dims(index[num_paired:], 0).repeat(num_views, axis=0)
            return Y, X, (index0, index_u), mask


def construct_incomplete_data(missrate, shape, mode='onehot', balance=True):
    """
    form_incomplete_data(missrate, X, Y, mode = None)
        Parameters
        ----------
        mode : string or int
            determine the miss situation.
            if mode=='onehot' or 1, un-missing in only one view
            if mode=='random' or 0, default, the number of missing views for each sample is generated randomly
            if mode==int and mode<len(X),  the number of missing views for each sample is determined by the integer
        balance : bool
            if return incomplete views with equal numbers
            True, default
        shape: (size, num of views)
        Returns
        ----------
        mask: same shape with shape
    """
    size, num_view = shape

    num_missing = int(missrate * size)
    mask = np.ones((num_view, size), dtype=int)
    if not missrate:
        # pvc or mvc
        return mask, num_missing

    mask_ = get_sn(num_view, size, missrate)
    mask = np.ones_like(mask_)
    mask_ = mask_[mask_.sum(1) < num_view]
    num_missing = len(mask_)
    mask[-num_missing:] = mask_
    mask = mask.T

    # if mode in ['onehot']:
    #     num_slots = np.array([num_view-1] * num_missing)
    # elif mode in ['random'] or mode % num_view == 0:
    #     num_slots = randint(1, high=num_view, size=num_missing)
    # elif isinstance(mode, int) and num_view > mode > 0:
    #     num_slots = np.array([num_view-mode] * num_missing)
    # else:
    #     raise ValueError('Not defined for the mode of {} ({})'.format(mode, type(mode)))

    # # incomplete data index
    # mask_ = np.ones((num_view, num_missing), dtype=int)
    # for i, n in enumerate(num_slots):
    #     # np.random.randint: [low, high)
    #     ind = np.random.randint(0, high=num_view, size=n)
    #     mask_[ind, i] = 0

    if balance:
        max_view = int(np.max(mask.sum(1)))
        for v, m in enumerate(mask):
            num_added = max_view - len(np.where(m > 0)[0])
            if num_added > 0:
                ind_zeros = np.where(m < 1)[0]
                # ind_added = np.random.randint(0, high=len(ind_zeros), size=num_added)
                ind_added = permutation(ind_zeros)[:num_added]
                m[ind_added] = 1
    # mask[:, -num_missing:] = mask_
    return mask, num_missing


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.3 of the paper
    :return:Sn
    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix

    matrix = np.ones((alldata_len, view_num))
    while error >= 0.005:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


