import numpy as np
import scipy.io as scio
from numpy.random import randint, shuffle, permutation
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sys
from data.read_mat import Caltech_2v, Caltech101_20, BDGP, Scene15, YouTube_X, Handwritten, ALOI100, MNIST_UPS, Reuters

import warnings
warnings.filterwarnings("ignore")

path = sys.path[0] + '/datasets'


def normalize_(X):
    min_max_scaler = MinMaxScaler()
    if isinstance(X, (list, tuple)):
        X = [min_max_scaler.fit_transform(x) for x in X]
        return X
    elif isinstance(X, np.ndarray):
        return min_max_scaler.fit_transform(X)


class DataLoader(object):
    def __init__(self, batch_size, normalized, pairedrate, missrate, dataset):
        # data to be hold
        # self.
        # label: list, [ndarray--(size,)]*n_view
        # data: list, [ndarray--(size, original_dim)]*n_view
        # Gp: ndarray, (n_views, paired_size)=1
        # index_u: ndarray, (n_views, unpaired_size) -- dic, pvc, imc;
        #          None -- mvc
        # mask: ndarray, (n_view, unpaired_size) -- dic or pvc;
        #       ndarray, (n_view, total_size) -- imc or mvc
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
                if batch_size > self.Gu.shape[1] else batch_size
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
                conf = [confp] * len(P)
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
        return [y[ind] for y, ind in zip(self.label, self.Gp)]

    @property
    def input_p(self):
        return {'input' + str(v + 1): x[ind] for v, (x, ind) in enumerate(zip(self.data, self.Gp))}

    @property
    def data_p(self):
        return [x[ind] for v, (x, ind) in enumerate(zip(self.data, self.Gp))]

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
            return np.concatenate([ind[np.nonzero(t)[0]].reshape(1, -1) for ind, t in zip(self.index_u, tmp)], axis=0)

    @property
    def Yu(self):
        if self.index_u is None:
            return None
        else:
            return [y[ind] for y, ind in zip(self.label, self.Gu)]

    @property
    def input_u(self):
        if self.index_u is None:
            return None
        else:
            # index_u =
            return {'input' + str(v + 1): x[ind] for v, (x, ind) in
                    enumerate(zip(self.data, self.Gu))}

    @property
    def data_u(self):
        if self.index_u is None:
            return None
        else:
            return [x[ind] for v, (x, ind) in enumerate(zip(self.data, self.Gu))]

    @property
    def pretraining_data(self):
        if self.Gu is not None:
            data = [np.concatenate([xp, xu], axis=0) for xp, xu in zip(self.data_p, self.data_u)]
        else:
            data = self.copy(self.data_p)
        return data


def load_data_conv(dataset, normalization, pairedrate, missrate):
    if dataset == 'Caltech101_20':
        X, Y = Caltech_2v()
    elif dataset == 'BDGP':
        X, Y = BDGP()
    elif dataset == 'Scene15':
        X, Y = Scene15()
    elif dataset == 'YouTube_X':
        X, Y = YouTube_X()
    elif dataset == 'Handwritten':
        X, Y = Handwritten()
    elif dataset == 'ALOI100':
        X, Y = ALOI100()
    elif dataset == 'MNIST_UPS':
        X, Y = MNIST_UPS()
    elif dataset == 'Reuters':
        X, Y = Reuters()
    else:
        raise ValueError('Not defined for loading %s' % dataset)

    if normalization:
        X = normalize_(X)
    return construct_dip(pairedrate, missrate, X, Y)


def construct_dip(pairedrate, missrate, X: list, Y: list, random=True):
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

    if int(num_paired) == 1:
        # imc or mvc
        mask, num_missing = construct_psp(missrate, (size, num_views))
        # len(mask) = len(index0) + len(index_u)
        if num_missing:
            # imc
            index0 = np.expand_dims(index[:-num_missing], 0).repeat(num_views, axis=0)
            index_u = np.expand_dims(index[-num_missing:], 0).repeat(num_views, axis=0)
            return Y, X, (index0, index_u), mask
        else:
            # mvc
            index = np.expand_dims(index, 0).repeat(num_views, axis=0)
            return Y, X, (index, None), mask
    else:
        # pvc, dimvc
        mask, _ = construct_psp(missrate, (size-num_paired, num_views))
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


def construct_psp(missrate, shape, balance=True, mode='default'):
    """construct partially instance-mssing data
        :param missrate: float, in (0, n_view-1)
             defined as "missing instances/number of examples"
        :param shape: tuple(int, int), (size, n_view)
        :param mode: deleted, str
            'default', default
            'onehot'
        :param balance: bool
            wether return data with each view having equal instences
            True, default
        :return mask: tuple(int, int), (n_view, size)
    """
    n_example, n_view = shape
    mask = np.ones((n_example, n_view), dtype=int)

    if not missrate:
        # pvc or mvc
        return mask.T, 0
    elif missrate > n_view-1:
        missrate = n_view-1
        mode = "onehot"
    # raw_msk: (size, num_view)
    matrix = _get_mask(missrate, n_example, n_view)
    matrix_missing = matrix[matrix.sum(1) < n_view]
    num_missing = len(matrix_missing)
    mask[-num_missing:] = matrix_missing

    if balance:
        max_view = int(np.max(mask.sum(0)))
        for v, m in enumerate(mask.T):
            num_added = max_view - len(np.where(m > 0)[0])
            if num_added > 0:
                ind_zeros = np.where(m < 1)[0]
                # ind_added = np.random.randint(0, high=len(ind_zeros), size=num_added)
                ind_added = permutation(ind_zeros)[:num_added]
                m[ind_added] = 1

    return mask.T, num_missing


def _get_mask(missrate, n_example, n_view):
    """Randomly generate mask_code for incomplete data
    :param n_view: number of views
    :param n_example: number of examples
    :param mr: missrate (=missing instances/number of examples)
    :return mask: (size, num_view)
    """
    # global missrate
    missrate = missrate/n_view
    # global preserve_rate
    pr = 1.0 - missrate
    if pr <= (1 / n_view):
        enc = OneHotEncoder()
        return enc.fit_transform(randint(0, n_view, size=(n_example, 1))).toarray()
    elif pr == 1:
        return np.ones((n_example, n_view)).astype(int)

    sum_instances = n_view * n_example
    residual_preserve_gt = sum_instances * pr - n_example
    residual_ratio_gt = residual_ratio = abs(pr - n_example/sum_instances)

    matrix = np.ones((n_example, n_view)).astype(int)
    while residual_ratio >= 0.005:
        enc = OneHotEncoder(dtype=int)  # n_values=view_num
        base_matrix = enc.fit_transform(randint(0, n_view, size=(n_example, 1))).toarray()
        residual_matrix = (randint(0, 100, size=(n_example, n_view)) < int(residual_ratio_gt * 100)).astype(int)

        overlap = ((base_matrix + residual_matrix) > 1).astype(int).sum()
        residual_preserve_corrected = residual_preserve_gt**2 / (residual_preserve_gt - overlap)
        residual_ratio_corrected = residual_preserve_corrected / sum_instances
        residual_matrix = (randint(0, 100, size=(n_example, n_view)) < int(residual_ratio_corrected * 100)).astype(int)

        matrix = ((base_matrix + residual_matrix) > 0).astype(int)
        residual_ratio = abs(pr - matrix.sum() / sum_instances)
    return matrix

