import numpy as np
from scipy.optimize import linear_sum_assignment
import tensorflow as tf
from utils.Nmetrics import evaBymetrics
from sklearn.metrics.pairwise import rbf_kernel
from models.optima import InitializeSIGs


def project_ytrue(y_true, n_class, M):
    if not isinstance(M, list):
        M = [M]
    Y = np.eye(n_class)[y_true]
    for i, m in enumerate(M):
        M[i] = Y+m
    return M


def cluster_match(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    true2pred = dict()
    pred2true = dict()

    matrix = np.zeros((D, D), dtype=np.int64)
    matrix[row_ind, col_ind] = 1
    for i, j in zip(row_ind, col_ind):
        pred2true[i] = j
        true2pred[j] = i

    return matrix, (pred2true, true2pred)


class MvMetrics(tf.keras.metrics.Metric):
    def __init__(self, n_views, loss_name, name='dpimvc_metrics', **kwargs):
        super(MvMetrics, self).__init__(name=name, dtype='float32', **kwargs)
        self.n_views = n_views
        self.loss_name = loss_name

        self.count = self.add_weight(name='count', initializer='zero')
        self.loss_sum = self.add_weight(name='loss_sum', initializer='zero')
        self.loss_details = [self.add_weight(name=ln, initializer='zeros', shape=(self.n_views,))
                             for ln in self.loss_name]

        self.loss_sum_global = []
        self.details_global = []

    def update_state(self, train_loss, mask=None):
        train_loss = tf.reshape(tf.cast(train_loss, 'float32'), (-1))
        self.loss_sum.assign_add(train_loss[0])
        self.count.assign_add(1.0)
        for i, d in enumerate(self.loss_details):
            d.assign_add(train_loss[1 + i::len(self.loss_details)])

    def result(self):
        loss_sum = self.loss_sum / self.count
        loss_details = [d / self.count for d in self.loss_details]
        return loss_sum, loss_details

    def reset_states(self):
        loss_sum = self.loss_sum / self.count
        loss_details = [d / self.count for d in self.loss_details]
        self.loss_sum_global.append(loss_sum)
        self.details_global.append(loss_details)

        self.count.assign(0.0)
        self.loss_sum.assign(0.0)
        for d in self.loss_details:
            d.assign([0.0] * self.n_views)


def evaluate(Yp, pred_p, pred_with_P, Yu=None, pred_u=None,
             metrics=None, out=None, prefix=None):

    if out is not None:
        metrics = out.metrics
    assert metrics is not None

    if prefix is not None:
        assert isinstance(prefix, (bool, str, list))

    n_views = len(Yp)
    default = ['Globel:', 'Consensus-Prediction:', 'Aligned-View', 'Unaligned-View']
    if isinstance(prefix, list):
        itemlist = prefix
    elif isinstance(prefix, str):
        itemlist = [prefix + d for d in default]
    else:
        itemlist = default
    Yp = Yp[0] if isinstance(Yp, list) else Yp

    # eva with paired data
    # eva global feature
    m_global = evaBymetrics(Yp, pred_with_P, metrics, pre=itemlist[0])

    # consensus on soft labels on every view
    average_pred = np.mean(np.array(pred_p), axis=0)
    _ = evaBymetrics(Yp, average_pred, metrics, pre=itemlist[1])

    # eva each view
    itemp = itemlist[-2]
    mp = np.zeros((len(metrics), n_views))
    for v, pdp in enumerate(pred_p):
        mp[:, v] = evaBymetrics(Yp, pdp, metrics, pre=itemp + str(v + 1) + ':' if itemp else False)

    if Yu is not None:
        # eva with unpaired data
        itemu = itemlist[-1]
        mu = np.zeros((len(metrics), n_views))
        for v, (yu, pdu) in enumerate(zip(Yu, pred_u)):
            mu[:, v] = evaBymetrics(
                yu, pdu, metrics, pre=itemu + str(v + 1) + ':' if itemu else False)
        if out is not None:
            out.update_once(m_global, mp, mu)
    else:
        if out is not None:
            out.update_once(m_global, mp)


def mixup(P, features_p, features_u, Q, k, confp=None, confu=None, standard=False, standard1=True):
    if standard:
        knn_label = []
        m = len(features_p[0])
        n = [len(fu) for fu in features_u]
        features = [np.concatenate([fp, fu], axis=0) for fu, fp in zip(features_u, features_p)]
        for f, nv in zip(features, n):
            S = rbf_kernel(f, f)
            # Smn = S[:m, m:]
            Snm = S[m:, :m]
            Snn = S[m:, m:]
            knn_label_ = np.linalg.inv(np.diag(Snm.sum(1))+np.diag(Snn.sum(1))-Snn) @ Snm @ np.diag(confp) @ P
            knn_label.append(knn_label_ / np.sum(knn_label_, axis=1, keepdims=True))
        print('standard LP')
        return knn_label
    D_u2p = [np.sum(np.square(np.expand_dims(fu, 1) - fp), 2) for fu, fp in zip(features_u, features_p)]
    Similarity_u2p = []
    for d in D_u2p:
        tmp = InitializeSIGs(d, k, 'p')
        Similarity_u2p.append(tmp)
    if confp is not None:
        if standard1:
            print('standard1')
            knn_label = []
            for sm in Similarity_u2p:
                Snn = sm@sm.T
                Snm = sm
                knn_label_ = np.linalg.inv(np.diag(Snm.sum(1)) + np.diag(Snn.sum(1)) - Snn) @ Snm @ P
                knn_label.append(knn_label_ / np.sum(knn_label_, axis=1, keepdims=True))
        else:
            knn_label = [s @ np.diag(confp) @ P for s in Similarity_u2p]
    else:
        knn_label = [s @ P for s in Similarity_u2p]

    # knn_label = [kl / np.sum(kl, axis=1, keepdims=True) for kl in knn_label]

    if confu is not None:
        assert len(features_u) == len(Q)

        knn_label_with_self = [
            (np.diag(cfu) @ q + l) / (cfu + 1).reshape(-1, 1)
            for cfu, q, l in zip(confu, Q, knn_label)]
        knn_label_with_self = [kl / np.sum(kl, axis=1, keepdims=True) for kl in knn_label_with_self]
        knn_label = [kl / np.sum(kl, axis=1, keepdims=True) for kl in knn_label]
        return knn_label, knn_label_with_self

    else:
        knn_label = [kl / np.sum(kl, axis=1, keepdims=True) for kl in knn_label]
        return knn_label

