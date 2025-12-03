import numpy as np


def EProjSimplex(v, k=1):
    assert isinstance(v, np.ndarray)
    v = v.copy()
    ft = 1
    v = v.reshape(-1)
    n = v.shape[0]

    v0 = v - v.mean() + k / n
    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while np.abs(f) > 10e-10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft += 1
            if ft > 100:
                x = np.where(v1 > 0, v1, 0)
                break
        x = np.where(v1 > 0, v1, 0)
    else:
        x = v0
    return x, ft


def EProjSimplex_M(D, gamma,  norm_constrants=1, max_iter=100, eps=1e-10):
    Q = D/(2*gamma + eps)
    n = Q.shape[-1]
    Q = Q - np.mean(Q, axis=1, keepdims=True) + norm_constrants/n

    update_ind = np.arange(Q.shape[0])
    update_ind = np.setdiff1d(update_ind, np.where(np.min(Q, axis=1) > 0))

    f = np.ones((Q.shape[0],))
    phai = np.zeros((Q.shape[0],))
    ft = 1
    while len(update_ind) > 0:
        f1 = phai[update_ind].reshape(-1, 1) - Q[update_ind]
        pos_ind = f1 > 0
        # g must be negtive
        g = np.mean(pos_ind, axis=1) - 1
        g = np.where(g != 0, g, eps)
        # f[update_ind] = np.sum(f1 * pos_ind, axis=1)/n - phai[update_ind]
        f[update_ind] = np.mean(f1 * pos_ind, axis=1) - phai[update_ind]
        phai[update_ind] = phai[update_ind] - f[update_ind] / g
        ft += 1
        if ft >= max_iter:
            P = Q - phai.reshape(-1, 1)
            return np.where(P > 0, P, 0)
        update_ind = np.setdiff1d(update_ind, np.where(f <= eps))
    P = Q - phai.reshape(-1, 1)
    return np.where(P > 0, P, 0)


def EProjSimplex_constraints(D, T, k, omega=1, eps=1e-10):
    """
    ProjSimplex_gmc(d, t, omega, k)
        Parameters
        ----------
        D : array_like
            not sorted
        T : array_like
            not sorted
    """
    assert D.shape == T.shape
    assert k <= D.shape[-1]
    P = np.zeros_like(D)
    n = D.shape[0]
    ind = np.argsort(D)
    id_ = ind[:, :k + 1]
    di = D[np.arange(n).reshape(-1, 1), id_]
    numerator = di[:, [k]] - di \
                + 2 * omega * T[np.arange(n).reshape(-1, 1), id_] \
                - 2 * omega * T[np.arange(n), id_[:, k]].reshape(-1, 1)
    denominator1 = k * di[:, [k]].reshape(-1) - np.sum(di[:, :k], axis=1)
    denominator2 = 2 * k * omega * T[np.arange(n), id_[:, k]].reshape(-1) \
                   - 2 * omega * np.sum(T[np.arange(n).reshape(-1, 1), id_[:, :k]], axis=1)
    denominator = denominator1 - denominator2
    tmp = numerator / (denominator.reshape(-1, 1) + eps)
    P[np.arange(n).reshape(-1, 1), id_] = np.where(tmp > 0, tmp, 0)
    return P


def Optimize_consensusP(D, Pv, weights, alpha, max_iter=100, eps=1e-10):
    m = len(D)
    n = D[0].shape[-1]
    Qv = [pv-alpha*d/(2*m*wv) for pv, d, wv in zip(Pv, D, weights)]
    Qv = sum(Qv)/m + 1/n - sum([np.sum(q, axis=1)/(m*n) for q in Qv]).reshape(-1, 1)
    # P: np.where((qv - phai)>0, (qv - phai), 0)
    update_ind = np.arange(Qv.shape[0])
    update_ind = np.setdiff1d(update_ind, np.where(np.min(Qv, axis=1) > 0))

    f = np.ones((Qv.shape[0],))
    phai = np.zeros((Qv.shape[0],))
    ft = 1

    while len(update_ind) > 0:
        f1 = phai[update_ind].reshape(-1, 1) - Qv[update_ind]
        pos_ind = f1 > 0
        g = np.sum(pos_ind, axis=1)/n - 1
        g = np.where(g != 0, g, eps)
        # f[update_ind] = np.sum(f1 * pos_ind, axis=1)/n - phai[update_ind]
        f[update_ind] = np.mean(f1 * pos_ind, axis=1) - phai[update_ind]
        phai[update_ind] = phai[update_ind] - f[update_ind]/g
        ft += 1
        if ft >= max_iter:
            P = Qv - phai.reshape(-1, 1)
            return np.where(P > 0, P, 0)
        update_ind = np.setdiff1d(update_ind, np.where(f <= eps))
    P = Qv - phai.reshape(-1, 1)
    return np.where(P > 0, P, 0)


def InitializeSIGs(D, k, mode='default', eps=1e-6):
    n = D.shape[0]
    S = np.zeros_like(D)

    ind = np.argsort(D)
    id_ = ind[:, :k + 1]
    di = D[np.arange(n).reshape(-1, 1), id_]

    if mode.lower() in ['beta']:
        beta = (k * di[:, k] - np.sum(di[:, :k], 1)) / 2
        beta = beta.mean()
        return beta
    elif mode.lower() in ['s', 'default']:
        molecule = di[:, [k]] - di
        denominator = k * di[:, k] - np.sum(di[:, :-1], axis=1) + eps
        weight = molecule / denominator.reshape(-1, 1)
        S[np.arange(n).reshape(-1, 1), id_] = weight.copy()

        ind_0 = np.where(S.sum(1) == 0)[0]
        if len(ind_0) > 0:
            S[ind_0.reshape(-1, 1), ind[ind_0, :k]] = 1.0 / k
        return S


def adaptive_lp(Tm, Dnm, Dnn, beta, gamma, zeta, max_iters=100):
    """
    Adaptive Label Propagation model
    Parameters
    ----------
    Tm
    Dnm
    Dnn
    beta: numer of non-zero elements for each view
    gamma: balance graph learning and LP
    zeta: balance Snm and Snn
    max_iters

    Returns Snm, Snn, Tnn
    -------
    """
    # initialization
    Snm = InitializeSIGs(Dnm, beta).astype(np.float32)
    Snn = InitializeSIGs(Dnn, beta).astype(np.float32)
    Tm = Tm.astype(np.float32)
    Dnm = Dnm.astype(np.float32)
    Dnn = Dnn.astype(np.float32)

    Tn = np.linalg.inv((1+zeta) * np.eye(len(Snn)) - zeta * Snn) @ Snm @ Tm
    Tn = Tn.astype(np.float32)

    if max_iters <= 1:
        return Tn, Snm, Snn
    else:
        iteration = 2
        while iteration <= max_iters:
            # abel correction
            Wnm = np.sum(np.square(np.expand_dims(Tn, 1) - Tm), 2)
            Wnn = Tn @ Tn.T

            Znm = Dnm + gamma * Wnm
            Znn = Dnn - gamma * Wnn

            Snm = InitializeSIGs(Znm, beta)
            Snn = InitializeSIGs(Znn, beta)
            Tn = np.linalg.inv((1+zeta) * np.eye(len(Snn)) - zeta * Snn) @ Snm @ Tm
            Tn = Tn.astype(np.float32)
            iteration = iteration + 1
        return Tn, Snm, Snn
