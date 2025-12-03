# this file is an implementation of Semi-Supervised Learning via Bipartite Graph Construction With Adaptive Neighbors
# customized for PIMVC
import numpy as np
from sklearn.cluster import KMeans
import scipy.io as io
from utils.Nmetrics import evaBymetrics
from utils.visualize import project_ytrue
import scipy
from models.optima import EProjSimplex_M, EProjSimplex_constraints, Optimize_consensusP, InitializeSIGs
from sklearn import preprocessing

Caltech = './results/Caltech/Caltech_500_ini.mat'
BDGP = './results/BDGP/BDGP_500_ini.mat'


def LP_anchors(X, Y, Z, alpha, gamma=1, lambda_max=1, max_iter=50, eps=1e-10,
               ini_F=None, gamma_update=False, **kwargs):
    assert isinstance(X, np.ndarray)
    assert isinstance(Z, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert X.shape[-1] == Z.shape[-1]
    assert gamma > 0
    assert alpha > 0

    n = X.shape[0]
    m = Z.shape[0]
    n_labels = Y.shape[0]
    n_class = Y.shape[1]

    if n_labels < (n + m):
        Y = np.concatenate([Y, np.zeros([n + m - n_labels, n_class])], axis=0)
    F = np.copy(Y[:n, :])
    # G=np.copy(Y[n:, :])
    if m == n_class:
        G = np.diag([1] * m)
    else:
        G = np.copy(Y[n:, :])

    if gamma < 1:
        k = int(m * gamma)
        if k == 0:
            k = 1
    else:
        k = int(gamma)
    assert k <= m

    D_base = np.sum(np.square(np.expand_dims(X, 1) - Z), 2)
    D = D_base + alpha * np.sum(np.square(np.expand_dims(F, 1) - G), 2)
    assert D_base.shape == (n, m)
    assert D.shape == (n, m)
    sort_D = np.sort(D)
    gamma = (k * sort_D[:, k] - np.sum(sort_D[:, :k], 1)) / 2
    gamma = gamma.mean()
    P = np.zeros([n, m])

    U_n = np.diag([lambda_max] * n_labels + [0] * (n - n_labels))
    I_n = np.eye(n)
    I_m = np.eye(m)
    Q = np.linalg.inv(I_n + U_n) @ U_n @ Y[:n, :]
    loss_previous = 0
    for ft in range(max_iter):
        previous_P = P.copy()
        # update Pij
        # for i in range(n):
        #     P[i, :], _ = EProjSimplex(D[i, :] / (2 * gamma + eps))
        P = EProjSimplex_M(D, gamma)
        sum_P0 = np.sum(P, 0)
        lambda_diag_inv = np.linalg.inv(np.diag(np.where(sum_P0 > 0, sum_P0, 1e-6)))
        # update F,G
        S = np.linalg.inv(I_n + U_n) @ P @ lambda_diag_inv
        F = (I_n + S @ np.linalg.inv(I_m - P.T @ S) @ P.T) @ Q
        G = lambda_diag_inv @ P.T @ F
        # update D
        D = D_base + alpha * np.sum(np.square(np.expand_dims(F, 1) - G), 2)
        if gamma_update:
            sort_D = np.sort(D)
            gamma = (k * sort_D[:, k] - np.sum(sort_D[:, :k], 1)) / 2
            gamma = gamma.mean()
        loss = np.linalg.norm(previous_P - P, ord=2)
        if ft < 7 or ft % 5 == 4:
            print('  subiter: {}; loss: {:.5f}'.format(ft + 1, loss))
        # print(loss_previous, np.abs(loss - loss_previous))
        if loss < 1e-4 or np.abs(loss - loss_previous) < 1e-4:
            break
        loss_previous = loss
    return F, G, P


def LP_dmc(X, Y, alpha,
           gamma=1, lambda_max=1, max_iter=50, gamma_update=False, **kwargs):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert gamma > 0
    assert alpha > 0

    # number of anchors
    m = Y.shape[0]
    # number of samples to be labeled
    n = X.shape[0] - m
    n_class = Y.shape[1]
    S = Y @ Y.T
    Q = Y.copy()
    if m < X.shape[0]:
        Y = np.concatenate([Y, np.zeros([n, n_class])], axis=0)
    G = np.copy(Y[:m, :])
    F = np.copy(Y[m:, :])
    Z = X[:m, :]
    X = X[m:, :]

    if gamma < 1:
        k = int(m * gamma)
    else:
        k = int(gamma)
    assert k <= m

    P = np.zeros([n, m])
    D_base = np.sum(np.square(np.expand_dims(X, 1) - Z), 2)
    D_rep1 = np.sum(np.square(np.expand_dims(F, 1) - G), 2)
    D_rep2 = np.sum(np.square(np.expand_dims(F, 1) - F), 2) @ P @ S
    assert D_base.shape == D_rep1.shape == D_rep2.shape == (n, m)
    D = D_base + alpha * (D_rep1 + 2 * D_rep2)
    sort_D = np.sort(D)
    gamma = (k * sort_D[:, k] - np.sum(sort_D[:, :k], 1)) / 2
    gamma = gamma.mean()

    U_m = np.diag([lambda_max] * m)
    I_n = np.eye(n)
    loss_previous = 0
    for ft in range(max_iter):
        previous_P = P.copy()
        # update Pij
        # for i in range(n):
        #     P[i, :], _ = EProjSimplex(D[i, :] / (2 * gamma + 1e-6))
        P = EProjSimplex_M(D, gamma)
        sum_P0 = np.sum(P, 0)
        lambda_ = np.diag(sum_P0)
        # update F,G
        L = P @ S @ P.T
        L = np.diag(np.sum(L, 0)) - L
        tmp = np.linalg.inv(I_n + L) @ P
        G = np.linalg.inv(lambda_ + U_m - P.T @ tmp) @ U_m @ Q
        F = tmp @ G

        # update D
        D_rep1 = np.sum(np.square(np.expand_dims(F, 1) - G), 2)
        D_rep2 = np.sum(np.square(np.expand_dims(F, 1) - F), 2) @ P @ S
        D = D_base + alpha * (D_rep1 + 2 * D_rep2)
        if gamma_update:
            sort_D = np.sort(D)
            gamma = (k * sort_D[:, k] - np.sum(sort_D[:, :k], 1)) / 2
            gamma = gamma.mean()

        loss = np.linalg.norm(previous_P - P, ord=2)
        print('  subiter: {}; loss: {:.5f}'.format(ft + 1, loss))
        # print(loss_previous, np.abs(loss - loss_previous))
        if loss < 1e-4 or np.abs(loss - loss_previous) < 1e-4:
            break
        loss_previous = loss
    return F, G


def LP_dmc_constraints(X, Y, Z, alpha, gamma=1, lambda_max=1, max_iter=50,
                       gamma_update=True, enhanceY=False, eps=1e-6, **kwargs):
    assert isinstance(X, np.ndarray) and isinstance(Z, np.ndarray) and isinstance(Y, np.ndarray)
    assert gamma > 0
    assert alpha > 0

    n = X.shape[0]
    m = Z.shape[0]
    n_labels = Y.shape[0]
    n_class = Y.shape[1]
    T = Y.copy()
    assert n_labels < n

    # enhance Y
    if m == n_class:
        G = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(Z, axis=1) - Z), axis=2) / 1))
        G **= (1 + 1.0) / 2.0
        G = np.transpose(np.transpose(G) / np.sum(G, axis=1))
        if enhanceY:
            Y = G[np.argmax(Y, axis=1)]
    Y = np.concatenate([Y, np.zeros([n - n_labels, n_class])], axis=0)

    # cal k
    k = min(int(m * gamma), 2) if gamma < 1 else int(gamma)
    assert k <= m

    P_periter = []
    D_base = np.sum(np.square(np.expand_dims(X, 1) - Z), 2)

    # ini P and gamma
    P, gamma = InitializeSIGs(D_base, k)
    P_periter.append(P.copy())

    # constant
    U_n = np.diag([lambda_max] * n_labels + [0] * (n - n_labels))
    Q = np.linalg.inv(np.eye(n) + U_n) @ U_n @ Y

    loss_previous = 0
    for ft in range(max_iter):
        previous_P = P.copy()
        sum_Pc = np.sum(P, 0)
        print('sum Pc:', sum_Pc)
        lambda_diag_inv = np.linalg.inv(np.diag(np.where(sum_Pc > 0, sum_Pc, eps)))
        # update F,G
        S = np.linalg.inv(np.eye(n) + U_n) @ P @ lambda_diag_inv
        F = (np.eye(n) + S @ np.linalg.inv(np.eye(m) - P.T @ S) @ P.T) @ Q
        G = lambda_diag_inv @ P.T @ F
        # normalize F,G
        F = F / np.sum(F, axis=0)
        F = np.transpose(np.transpose(F) / np.sum(F, axis=1))
        G = G / np.sum(G, axis=0)
        G = np.transpose(np.transpose(G) / np.sum(G, axis=1))
        # G = G / np.sqrt(np.sum(P, axis=0))
        # G = np.transpose(np.transpose(G) / np.sum(G, axis=1))
        # update D
        D = D_base + alpha * np.sum(np.square(np.expand_dims(F, 1) - G), 2)

        # update Pij
        # update P for A
        omega = 1 / (2 * np.linalg.norm(T - P[:n_labels, :]))
        P[:n_labels] = EProjSimplex_constraints(D[:n_labels].copy(), T, k, omega)
        # update P for U
        P[n_labels:] = EProjSimplex_M(D[n_labels:].copy(), gamma)
        P_periter.append(P)

        # update gamma
        if gamma_update:
            sort_D = np.sort(D)
            gamma = (k * sort_D[:, k] - np.sum(sort_D[:, :k], 1)) / 2
            gamma = gamma.mean()
        loss = np.linalg.norm(previous_P - P)
        if ft < 7 or ft % 5 == 4:
            print('  subiter: {}; loss: {:.5f}'.format(ft + 1, loss))
        if loss < eps or np.abs(loss - loss_previous) < eps:
            break
        loss_previous = loss
    return F, G, P_periter


def multiview_LP(X, Z, Q, n_paired, n_clusters, alpha=None, gamma=1, max_iter=50,
                 gamma_update=True, eps=1e-10, **kwargs):
    assert isinstance(X, list) and isinstance(Z, list) and len(X) == len(Z)
    if 'y_true' in kwargs.keys():
        y_true = kwargs['y_true']

    n_views = len(X)
    n_samples = [x.shape[0] for x in X]
    m = Z[0].shape[0]
    D_base = [np.sum(np.square(np.expand_dims(x, 1) - z), 2) for x, z in zip(X, Z)]
    alpha = alpha if alpha else 1
    k = max(int(m * gamma), 2) if gamma < 1 else int(gamma)
    assert k <= m

    # initialize Pv_paired and Pv_up
    Pv = []
    for v in range(n_views):
        ini_Pv, g = InitializeSIGs(D_base[v].copy(), k)
        Pv.append(ini_Pv)
    Pv_paired = [pv[:n_paired] for pv in Pv]
    Pv_up = [pv[n_paired:] for pv in Pv]

    # for pv, pvp in (Pv, Pv_paired):
    #     print('ini col sum of Pv:', np.sum(pv, axis=0))
    #     print('ini col sum of Pvp:', np.sum(pvp, axis=0))
    # ini P
    P = sum(Pv_paired) / n_views
    P = np.transpose(np.transpose(P) / np.sum(P, axis=1))
    eva_consice(Pv_paired, Pv_up, P, n_paired, y_true)

    v_var = [z.var() for z in Z]
    
    print('var_previous:{}'.format(v_var))
    for v, (x, pvp, pvu) in enumerate(zip(X, Pv_paired, Pv_up)):
        Z[v] = np.linalg.inv(np.diag(np.sum(pvp, axis=0) + np.sum(pvu, axis=0))) @ \
               (pvp.T @ x[:n_paired] + pvu.T @ x[n_paired:])
    D_base = [np.sum(np.square(np.expand_dims(x, 1) - z), 2) for x, z in zip(X, Z)]
    v_var = [z.var() for z in Z]
    print('var_new:{}'.format(v_var))
    # construct LP and ini H
    H = []
    for v, (n, pvu) in enumerate(zip(n_samples, Pv_up)):
        lp = np.zeros((n + m, n + m))
        lp[:n, -m:] = np.concatenate([P, pvu], axis=0)
        lp = lp + lp.T
        LP = np.diag(np.sum(lp, axis=1)) - lp
        evals, evecs = scipy.linalg.eigh(LP, subset_by_index=[0, n_clusters])
        H.append(evecs[:, :-1])

    D_rep = [np.sum(np.square(np.expand_dims(h[:-m], 1) - h[-m:]), 2) for h in H]
    gamma = [InitializeSIGs(db[n_paired:].copy() + alpha * dr[n_paired:].copy(), k, 'g')
             for db, dr in zip(D_base, D_rep)]
    weights = np.ones((n_views,)) / n_views
    for ft in range(max_iter):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        print('iter={} ; alpha={}'.format(ft + 1, alpha))

        # update z
        # for v, (x, pvp, pvu) in enumerate(zip(X, Pv_paired, Pv_up)):
        #     Z[v] = np.linalg.inv(np.diag(np.sum(P, axis=0) + np.sum(pvu, axis=0))) @ \
        #            (P.T @ x[:n_paired] + pvu.T @ x[n_paired:])
        #     D_base[v] = np.sum(np.square(np.expand_dims(x, 1) - Z[v]), 2)

        H_previous = H

        # update pv for paired data; update pv for unpaired data
        for v, (db, dr, wv, g) in enumerate(zip(D_base, D_rep, weights, gamma)):
            Pv_paired[v] = EProjSimplex_constraints(db[:n_paired].copy(), P.copy(), k, omega=wv)
            Pv_up[v] = EProjSimplex_M(db[n_paired:].copy() + alpha * dr[n_paired:].copy(), g)

        # update weights
        weights = [1 / (2 * np.linalg.norm(P - pvp)) for pvp in Pv_paired]

        # update P
        P = Optimize_consensusP([dr[:n_paired].copy() for dr in D_rep],
                                [pvp.copy() for pvp in Pv_paired], weights.copy(), alpha)

        eva_consice(Pv_paired, Pv_up, P, n_paired, y_true)
        # update H
        H = []
        Evals = []
        for v, (n, pvu) in enumerate(zip(n_samples, Pv_up)):
            lp = np.zeros((n + m, n + m))
            lp[:n, -m:] = np.concatenate([P, pvu], axis=0)
            lp = lp + lp.T
            LP = np.diag(np.sum(lp, axis=1)) - lp
            evals, evecs = scipy.linalg.eigh(LP, subset_by_index=[0, n_clusters])
            tmp = 5 if n_clusters > 5 else n_clusters
            print('least few evals of view {} = {}'.format(v + 1, evals[:tmp]))
            H.append(evecs[:, :-1])
            Evals.append((np.sum(evals[:-1]), np.sum(evals)))
        print('Evals for views =  {}'.format(Evals))

        if np.mean([e[0] for e in Evals]) > eps:
            alpha = alpha
        elif np.mean([e[1] for e in Evals]) < eps:
            alpha = alpha
            H = H_previous
        else:
            return Pv_paired, Pv_up, P
        D_rep = [np.sum(np.square(np.expand_dims(h[:-m], 1) - h[-m:]), 2) for h in H]
        if gamma_update:
            gamma = [InitializeSIGs(db[n_paired:].copy() + alpha * dr[n_paired:].copy(), k, 'g')
                     for db, dr in zip(D_base, D_rep)]
    return Pv_paired, Pv_up, P


def read_mat(file_path):
    data = io.loadmat(file_path)
    Y = [value.reshape(-1) for (key, value) in data.items() if key.split('_')[0] in ['Y']]
    Y = Y[0].astype(np.int64)
    n_class = len(np.unique(Y))
    index = [value.reshape(-1) for (key, value) in data.items() if key.split('_')[0] in ['index']]

    # X = [value for (key, value) in data.items() if key.split('_')[0] in ['X']]
    Q = [value for (key, value) in data.items() if key.split('_')[0] in ['Q']]
    H = [value for (key, value) in data.items() if key.split('_')[0] in ['H']]
    M = [value.reshape(-1) for (key, value) in data.items() if key.split('_')[0] in ['mask']]
    anchors_ = [value[0] for (key, value) in data.items() if key.split('_')[0] in ['centroids']]
    P = data['P']
    weights = data['weights']
    print('weights of views: ', weights)
    # tmp = [np.eye(n_class)[np.argmax(Q[i], axis=1)] for i in range(len(H))]
    # an_ = [np.linalg.inv(np.diag(np.sum(tmp[i], axis=0))) @ tmp[i].T @ H[i] for i in range(len(H))]
    # for a1, a2 in zip(an_, anchors_):
    #     print(a1-a2)
    return H, index, Y, P, M, anchors_, Q


def gen_unlabeled(X, x_ind, Y, pseudo_labels=None, anchors=None, labeled_rate=None, **kwargs):
    assert isinstance(X, list)
    assert pseudo_labels is not None or labeled_rate is not None

    n_cluster = len(np.unique(Y))
    data = {}
    n_samples = Y.shape[0]
    if pseudo_labels is None:
        n_labeled = int(n_samples * labeled_rate) if labeled_rate < 1 else int(labeled_rate)
    elif labeled_rate is None:
        assert pseudo_labels.shape[0] < n_samples
        n_labeled = pseudo_labels.shape[0]
    else:
        n_labeled = min(int(n_samples * labeled_rate) if labeled_rate < 1 else int(labeled_rate),
                        pseudo_labels.shape[0])
    data.update({'n_labeled': n_labeled,
                 'n_cluster': n_cluster})

    min_max_scalar = [preprocessing.MinMaxScaler()] * len(X)

    data.update({'X': [scalar.fit_transform(x) for x, scalar in zip(X, min_max_scalar)],
                 'index': x_ind,
                 'Pseudo_labels': pseudo_labels,
                 'y_true': Y})

    if isinstance(anchors, int) and anchors > 0:
        n_anchors = int(n_samples * anchors) if anchors < 1 else int(anchors)
        cluster = [KMeans(n_clusters=n_anchors, random_state=0).fit(x) for x in X]
        anchors = [i.cluster_centers_ for i in cluster]
    data.update({'anchors': [scalar.transform(an) for an, scalar in zip(anchors, min_max_scalar)]})

    return data


def Run_LP(data_setting, alg, Q, **kwargs):
    assert 'alpha_bar' in kwargs.keys()
    assert 'gamma' in kwargs.keys()
    n_labels = data_setting['n_labeled']
    alpha_bar = kwargs['alpha_bar']

    X = data_setting['X']
    index = data_setting['index']
    Y = data_setting['y_true']
    y_true = [Y[ind] for ind in index]

    anchors = data_setting['anchors']

    Pseudo_labels = data_setting['Pseudo_labels']
    y_pseudo = np.argmax(Pseudo_labels, axis=1)
    P = Pseudo_labels[:n_labels, :]

    n_class = len(np.unique(Y))

    check_F = []
    G_set = []
    if alg in [LP_anchors, LP_pimvc_constraints]:
        for view, y_ in enumerate(y_true):
            print('=============================================')
            print('EVA VIEW-{}'.format(view))
            print('eva for Q: ')
            pre_Q_up = np.argmax(Q[view][n_labels:], 1)
            _, class_map = evaBymetrics(y_[n_labels:], pre_Q_up, pre='    ')
            print('====================================')
            for alpha in alpha_bar:
                print('==========alpha: {}=========='.format(alpha))
                F, G, P_periter = alg(X[view].copy(), P.copy(), anchors[view].copy(), alpha=alpha, **kwargs)
                # F= P_periter[0]
                G_set.append(G)

                fitted_y = np.argmax(F[:n_labels, :], 1)
                predicted_y = np.argmax(F[n_labels:, :], 1)
                st = [np.sum(predicted_y == i) for i in range(len(np.unique(Y)))]
                print('distribution of the predicted results:\n {}'.format(st))
                print('fitted part--compared to y_true: ')
                evaBymetrics(y_[:n_labels], fitted_y, pre='    ')
                print('fitted part--compared to y_pseudo: ')
                evaBymetrics(y_pseudo[:n_labels], fitted_y, pre='    ')
                print('predicted part: ')
                evaBymetrics(y_[n_labels:], predicted_y, pre='    ')

                # ana with Q
                # hit in Q
                hit_ind = y_[n_labels:] == np.array([class_map[0][i] for i in pre_Q_up])
                check_F.append(np.concatenate(
                    project_ytrue(
                        np.array([class_map[1][i] for i in y_[n_labels:][hit_ind]]), n_class,
                        [Q[view][n_labels:, :][hit_ind].copy(), F[n_labels:, :][hit_ind].copy()]
                    ), axis=1))

                # not hit in Q
                hit_ind = y_[n_labels:] != np.array([class_map[0][i] for i in pre_Q_up])
                check_F.append(np.concatenate(
                    project_ytrue(
                        np.array([class_map[1][i] for i in y_[n_labels:][hit_ind]]), n_class,
                        [Q[view][n_labels:, :][hit_ind].copy(), F[n_labels:, :][hit_ind].copy()]
                    ), axis=1))

                lambda_max = kwargs['lambda_max'] if 'lambda_max' in kwargs.keys() else 1
                FQ = (F + lambda_max * Q[view]) / (1 + lambda_max)
                predicted_y = np.argmax(FQ[n_labels:, :], 1)
                st = [np.sum(predicted_y == i) for i in range(len(np.unique(Y)))]
                print('fuzed_distribution of the predicted results:\n {}'.format(st))
                print('fuzed_predicted part: ')
                evaBymetrics(y_[n_labels:], predicted_y, pre='    ')
                # not hit in FQ
                hit_ind = y_[n_labels:] != np.array([class_map[0][i] for i in predicted_y])
                check_F.append(np.concatenate(
                    project_ytrue(
                        np.array([class_map[1][i] for i in y_[n_labels:][hit_ind]]), n_class,
                        [Q[view][n_labels:, :][hit_ind].copy(), F[n_labels:, :][hit_ind].copy()]
                    ), axis=1))
    elif alg in [multiview_LP]:
        # for alpha in alpha_bar:
        Pv_paired, Pv_up, P = alg(X, anchors, Q, n_paired=n_labels, n_clusters=n_class,
                                  y_true=y_true, **kwargs)
        print('\n\n')
        for v, (pvp, pvu, y_) in enumerate(zip(Pv_paired, Pv_up, y_true)):
            print('=============EVA VIEW-{}============='.format(v + 1))
            paired_y = np.argmax(pvp, 1)
            unpaired_y = np.argmax(pvu, 1)
            st_p = [np.sum(paired_y == i) for i in range(len(np.unique(Y)))]
            st_up = [np.sum(unpaired_y == i) for i in range(len(np.unique(Y)))]
            print('distribution of the paired:\n {}'.format(st_p))
            print('distribution of the unpaired:\n {}'.format(st_up))
            print('paired--compared to y_true: ')
            evaBymetrics(y_[:n_labels], paired_y, pre='    ')
            print('unpaired--compared to y_true: ')
            evaBymetrics(y_[n_labels:], unpaired_y, pre='    ')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            lambda_max = kwargs['lambda_max'] if 'lambda_max' in kwargs.keys() else 1
            QP_paired = (pvp + lambda_max * Q[v][:n_labels]) / (1 + lambda_max)
            QP_up = (pvp + lambda_max * Q[v][n_labels:]) / (1 + lambda_max)
            paired_y = np.argmax(QP_paired, 1)
            unpaired_y = np.argmax(QP_up, 1)
            st_p = [np.sum(paired_y == i) for i in range(len(np.unique(Y)))]
            st_up = [np.sum(unpaired_y == i) for i in range(len(np.unique(Y)))]
            print('distribution of the paired:\n {}'.format(st_p))
            print('distribution of the unpaired:\n {}'.format(st_up))
            print('paired--compared to y_true: ')
            evaBymetrics(y_[:n_labels], paired_y, pre='    ')
            print('unpaired--compared to y_true: ')
            evaBymetrics(y_[n_labels:], unpaired_y, pre='    ')
    else:
        print('An algorithm that does not exist')
        return


def eva_consice(Pvp, Pvu, P, n_labels, y_true):
    print('======================================')
    for v, (pvp, pvu, y_) in enumerate(zip(Pvp, Pvu, y_true)):
        print('=============EVA VIEW-{}============='.format(v + 1))
        paired_y = np.argmax(pvp, 1)
        unpaired_y = np.argmax(pvu, 1)
        st_p = [np.sum(paired_y == i) for i in range(len(np.unique(Y)))]
        st_up = [np.sum(unpaired_y == i) for i in range(len(np.unique(Y)))]
        print('distribution of the paired:\n {}'.format(st_p))
        print('distribution of the unpaired:\n {}'.format(st_up))
        print('paired--compared to y_true: ')
        evaBymetrics(y_[:n_labels], paired_y, pre='  ')
        print('unpaired--compared to y_true: ')
        evaBymetrics(y_[n_labels:], unpaired_y, pre='  ')
    print('\n')
    fuzed_paired_y = np.argmax(P, 1)
    fuzed_st_p = [np.sum(fuzed_paired_y == i) for i in range(len(np.unique(Y)))]
    print('distribution of the fuzed-paired:\n {}'.format(fuzed_st_p))
    evaBymetrics(y_true[0][:n_labels], fuzed_paired_y, pre='  ')
    print('======================================')


if __name__ == '__main__':
    alpha_bar = [1e-1, 3e-1, 1e0, 3e0, 1e1]
    gamma = 0.2
    X, index, Y, P, mask, anchors, Q = read_mat(file_path=Caltech)
    data_setting = gen_unlabeled(X, index, Y, P, anchors)
    # eva alg
    Run_LP(data_setting, multiview_LP, Q, alpha=1e1,
           max_iter=20, alpha_bar=alpha_bar, gamma=gamma, lambda_max=1, enhanceY=False)
    print('eva over')
