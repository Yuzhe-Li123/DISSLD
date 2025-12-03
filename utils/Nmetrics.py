import numpy as np
from sklearn.metrics import normalized_mutual_info_score, \
    adjusted_rand_score, v_measure_score, accuracy_score, mutual_info_score

# The linear_assignment function is deprecated in 0.21 and will be removed from 0.23.
# Use scipy.optimize.linear_sum_assignment instead.
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment

# nmi and vmeare are numerically equal
nmi = normalized_mutual_info_score
vmeasure = v_measure_score
ari = adjusted_rand_score


def cluster_match(y_true, y_pred):
    # y_pred to y_true
    assert y_pred.size == y_true.size
    assert y_true.dtype in [np.int64] and y_pred.dtype in [np.int64]
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for ypd, yt in zip(y_pred, y_true):
        w[ypd, yt] += 1
    return linear_sum_assignment(w, maximize=True), w


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    (ind_row, ind_col), w = cluster_match(y_true, y_pred)

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]).astype(np.float64) / y_pred.size


def purity(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    # Labels might be missing e.g with set like 0,2 where 1 is missing
    # First find the unique labels, then map the labels to an ordered set
    # 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that 
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaBymetrics(y_true, y_pred, metrics=None, prefix=None, display=True):
    CONFIG = {
        'ACC': acc,
        'NMI': nmi,
        'ARI': ari,
        'PUR': purity
    }
    if metrics is None:
        metrics = CONFIG.keys()

    re_display = {k: v(y_true, y_pred) for k, v in CONFIG.items()}
    re_return = []

    if display:
        prefix = prefix if isinstance(prefix, str) else ''
        print(prefix, end=' ')
        for k, v in re_display.items():
            if k.lower() in ['nmi', 'vme', 'nmi/vme']:
                k = 'NMI/VME'
            print(' {}: {:.5f}, '.format(k, v), end='')
        print('\n', end='')
    for m in metrics:
        m = 'NMI' if m.lower() in ['nmi', 'vme', 'nmi/vme'] else m
        re_return.append(re_display[m.upper()])
    return re_return


def statistical_analysis_with_pred_v1(y_true, y_pred1, y_pred0=None, itemlist=None):
    # pred to true
    y_true = y_true.astype(np.int64)
    y_pred = [y_pred0.astype(np.int64), y_pred1.astype(np.int64)] if y_pred0 is not None \
        else [y_pred1.astype(np.int64)]

    # true2pred : (row_ind, col_ind)
    # row:pred
    # col:true
    (row_ind, col_ind), w = cluster_match(y_true, y_pred[0])

    statistical_re = [np.array([np.sum(y_true == i) for i in col_ind], dtype=np.float64)]
    statistical_re.extend([
        np.array([np.sum(pred == i) for i in row_ind], dtype=np.float64) for pred in y_pred
    ])

    distribution_ = [st/np.sum(st) for st in statistical_re]
    if itemlist is not None:
        assert len(statistical_re) == len(itemlist)
        print('\n', end='')
        for item, st in zip(itemlist, statistical_re):
            if item:
                print('distribuiton of {}:\n  {}'.format(item, st))

    if y_pred0 is not None:
        pred_stability = mutual_info_score(*distribution_[-2:])
        pred_change_rate1 = np.mean(np.abs(
            statistical_re[-1] - statistical_re[-2]
        ) / statistical_re[-2])
        pred_change_rate2 = np.sum(np.abs(statistical_re[-1] - statistical_re[-2])) / len(y_pred1)
        print('stability of prediction: {:.5f}; change rate: {:.5f}, {:.5f}'.format(
            pred_stability, pred_change_rate1, pred_change_rate2))
        print('\n', end='')
        return statistical_re, pred_stability, pred_change_rate1, pred_change_rate2
    else:
        print('\n', end='')
        return statistical_re


def statistical_analysis_with_pred_v2(y_true, y_preds: list, itemlist: list = None):
    if itemlist is not None:
        assert len(y_preds) == len(itemlist) or len(y_preds) == len(itemlist)-1
    y_true = y_true.astype(np.int64)
    for i in range(len(y_preds)):
        y_preds[i] = y_preds[i].astype(np.int64)

    transformed_preds = []
    for ypd in y_preds:
        (row_ind, col_ind), _ = cluster_match(ypd, y_true)
        transformed_preds.append(col_ind)

    statistical_re = [np.array([np.sum(y_true == i) for i in range(y_true.max())])]
    for ypd, tr in zip(y_preds, transformed_preds):
        statistical_re.append(np.array([np.sum(ypd == i) for i in tr]))

    default = ['true', 'previous prediction',  'current prediction']
    if itemlist is not None:
        for item, st in zip(itemlist, statistical_re):
            print('distribution of {}:\n  {}'.format(item, st))

    if len(y_preds) == 2:
        d_last = statistical_re[-2]
        d_current = statistical_re[-1]
        pred_change_rate1 = np.mean(np.abs(d_last - d_current) / d_last)
        pred_change_rate2 = np.sum(np.abs(d_last - d_current)) / len(y_true)
        print('change rate:  {:.5f} (v1),  {:.5f} (v2)'.format(pred_change_rate1, pred_change_rate2))


def label_statistics(labelp, confp, yp):
    sum_num = len(labelp)
    # thresholds = np.linspace(0, 1, 5)
    thresholds = [0, 0.2, 0.4, 0.6, 0.8]
    group_label = []
    group_gt = []

    for t in thresholds:
        group_label.append(labelp[confp > t])
        group_gt.append(yp[confp > t])

    st_num = np.array([len(g) for g in group_label])
    for sn, pred, gt in zip(st_num, group_label, group_gt):
        print("number:{}".format(sn), end="")
        evaBymetrics(gt, pred)


if __name__ == '__main__':
    a = np.array([1, 1, 1, 3, 4, 2, 5, 6])
    b = np.array([1, 1, 1, 5, 4, 2, 5, 6])

