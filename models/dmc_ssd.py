from tensorflow.keras.models import Model
from tensorflow.keras import callbacks

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics.pairwise import rbf_kernel
import scipy.stats as sc
import numpy as np

from time import time
from datetime import datetime

from models.backbone import MvAE, ClusteringLayer

from models.optima import InitializeSIGs, adaptive_lp
from utils.Nmetrics import evaBymetrics, label_statistics


class MetricHandler(object):
    def __init__(self, metrics, n_views,
                 upmin, upmax, upstop,  consensus_threshold, tensorboard=True):

        # shape: (n_views, len(metrics))
        self.tensorboard = tensorboard
        self.callbacks_list = None
        self.metric_items = metrics
        self.shape = (n_views, len(metrics))

        # training condition
        self.upmax = upmax
        self.upmin = upmin
        self.upstop = upstop
        self.consensus_threshold = consensus_threshold
        # training
        self.update_id = None
        # vals
        self.global_performance = None

        self.aligned_ratio = None
        self.target_p = None
        self.consensus_p = None
        self.views_p = None

        self.target_u = None
        self.views_u = None

    @property
    def schedule(self):
        if self.aligned_ratio is None or len(self.aligned_ratio) <= self.upmin+1:
            return 0
        else:
            return (np.sum(np.array(self.aligned_ratio)[self.upmin+1:] > self.consensus_threshold).astype(float)
                    / self.upstop)

    def __enter__(self):
        if self.tensorboard:
            self.callbacks_list.on_train_begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tensorboard:
            self.callbacks_list.on_train_end()

    def reset(self):
        self.aligned_ratio = []
        self.target_p = []
        self.consensus_p = []
        self.views_p = []

        self.target_u = []
        self.views_u = []

        self.global_performance = {}

        # training
        self.update_id = 0

    def initialize(self, to_evaluate_paired_paired, to_evaluate_unpaired, model):
        self.reset()
        self.update_paired(*to_evaluate_paired_paired)
        Yu, target_preds_u, preds_u = to_evaluate_unpaired
        self.update_unpaired_target(Yu, target_preds_u, prefix="Target-")
        self.update_unpaired(Yu, preds_u)

        if self.tensorboard:
            TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
            self.callbacks_list = callbacks.CallbackList([
                callbacks.TensorBoard(log_dir='./logs/' + TIMESTAMP),
                # callbacks.CSVLogger(save_dir + '/model_train.log')
            ])
            self.callbacks_list.set_model(model)

        return self

    def status(self):
        self.update_id += 1
        # assert len(self.re_global_p) == len(self.re_global_u) == self.update_id
        # self.update_id, the times to begin

        if self.update_id <= self.upmin:
            return True
        elif self.update_id <= self.upmax:
            if self.schedule >= 1:
                return False
            else:
                return True
        else:
            # self.update_id > self.upmax:
            if len(self.global_performance) == 0:
                if (self.update_id - self.upmax) % 5 == 1:
                    self.consensus_threshold -= 0.05
                return True
            else:
                return False

    def update_paired(self, Yp, target_pred, preds, mean_pred=None, prefix=None):
        aligned_ratio, m_target, m_mean, msp = self.evaluate_paired(
            Yp, target_pred, preds, mean_pred, prefix, self.metric_items)

        m_target = m_target.reshape(-1)
        m_mean = m_mean.reshape(-1)

        assert len(self.metric_items) == len(m_target) == len(m_mean)
        assert msp.shape == self.shape

        self.aligned_ratio.append(aligned_ratio)
        self.target_p.append(m_target)
        self.consensus_p.append(m_mean)
        self.views_p.append(msp)

    def update_unpaired(self, Y, preds, prefix=None):
        mu = self.evaluate_unpaired(Y, preds, prefix, self.metric_items)
        assert mu.shape == self.shape

        self.views_u.append(mu)

    def update_unpaired_target(self, Y, preds, prefix="Target-"):
        mu = self.evaluate_unpaired(Y, preds, prefix, self.metric_items)
        assert mu.shape == self.shape

        self.target_u.append(mu)

    def hit_global(self):
        print("aligned ratio list:", self.aligned_ratio)
        print("schedule:", self.schedule)
        if self.update_id > self.upmin and self.aligned_ratio[-1] >= self.consensus_threshold:
            return True
        else:
            return False

    def update_global(self, Y, preds, prefix="Global-"):
        assert len(Y) == len(preds)
        gp = self.evaluate_unpaired(Y, preds, prefix, self.metric_items)
        self.global_performance.update({
            self.update_id: gp
        })

    def log(self, index, content_dict):
        if self.tensorboard:
            self.callbacks_list.on_epoch_end(index, content_dict)

    def performance(self, **kwargs):
        assert len(self.global_performance) > 0
        index = (self.global_performance.keys())

        base_log = {
            'update_index': list(index),
            'Aligned': [self.views_p[i] for i in index],
            'Unaligned': [self.views_u[i] for i in index],
            'View-leval': list(self.global_performance.values())
        }
        base_log.update(kwargs)
        return base_log

    @staticmethod
    def evaluate_paired(Yp, target_pred, preds, mean_pred=None, prefix=None, metrics=('ACC', 'NMI', 'ARI')):
        """evaluate the paired data
        :param Yp: gt, list, [ndarray, (paired_size,)]*n_views
        :param target_pred: ndarray, (paired_size) in [0, n_cluster-1]
        :param preds: list, [ndarray, (paired_size,) in [0, n_cluster-1]]*n_views
        :param mean_pred: ndarray, (paired_size) in [0, n_cluster-1]
        :param prefix: str, bool, [str]
            str, additional prompts for print
            [list], prompts for print
            bool, whether print
        :param metrics: tuple, (str)
        :return aligned_ratio: aligned ratio
        :return m_target: ndarray, (len(metrics))
        :return m_mean: ndarray, (len(metrics))
        :return m_specific: ndarray, (n_view, len(metrics))
        """

        aligned_ratio = MetricHandler.calculate_aligned_rate(preds)
        print('Aligned Ratio: {:.5f}'.format(aligned_ratio))
        if prefix is not None:
            assert isinstance(prefix, (bool, str, list))
        else:
            prefix = True

        n_views = len(Yp)
        default = ['Target-Aligned:', 'Consensus-Prediction:', 'Aligned-View']

        if isinstance(prefix, str):
            itemlist = [prefix + d for d in default]
        else:
            itemlist = default
        Yp = Yp[0] if isinstance(Yp, list) else Yp

        # eva global feature
        m_target = np.array(evaBymetrics(Yp, target_pred, metrics, prefix=itemlist[0]))

        # weighted mean on soft labels on every view
        if mean_pred is not None:
            m_mean = np.array(evaBymetrics(Yp, mean_pred, metrics, prefix=itemlist[1]))
        else:
            m_mean = np.array([0.] * len(metrics))

        # eva each view
        itemp = itemlist[2]
        m_specific = np.zeros((n_views, len(metrics)))
        for v, pdp in enumerate(preds):
            m_specific[v, :] = evaBymetrics(Yp, pdp, metrics, prefix=itemp + str(v + 1) + ':')

        return aligned_ratio, m_target, m_mean, m_specific

    @staticmethod
    def evaluate_unpaired(Yu, pred_u, prefix=None, metrics=('ACC', 'NMI', 'ARI')):
        mask = None
        # if mask:
        #     assert len(Yu) == len(pred_u) == len(mask)
        # if prefix is not None:
        #     assert isinstance(prefix, (bool, str))
        # else:
        #     prefix = True

        n_views = len(Yu)
        default = 'Unaligned-View'

        itemu = prefix + default if isinstance(prefix, str) else default
        mu = np.zeros((n_views, len(metrics)))

        if mask:
            for v, (yu, pdu, mk) in enumerate(zip(Yu, pred_u, mask)):
                yu = yu[:-mk] if mk else yu
                pdu = pdu[:-mk] if mk else pdu
                mu[v, :] = evaBymetrics(yu, pdu, metrics, prefix=itemu + str(v + 1) + ':')
        else:
            for v, (yu, pdu) in enumerate(zip(Yu, pred_u)):
                mu[v, :] = evaBymetrics(yu, pdu, metrics, prefix=itemu + str(v + 1) + ':')

        return mu

    @staticmethod
    def calculate_aligned_rate(preds):
        num_hit = len(preds[0])
        num_sample = len(preds[0])
        for prd in zip(*preds):
            tmp = np.array(prd)
            tmp -= tmp[0]
            if np.sum(tmp) != 0:
                num_hit -= 1
        return num_hit / num_sample

    @staticmethod
    def saved_as_json(file_path):
        pass


class MvDCN(object):
    def __init__(self, view_shape, n_clusters, ae=True, hdim=10):
        super(MvDCN, self).__init__()
        # compile
        self.n_loss = None

        # data
        self.targetP = None
        self.targetU = None
        # self.conf = None

        self.n_clusters = n_clusters
        self.pretrained = False
        self.n_views = len(view_shape)

        # prepare model
        self.AEs, self.encoders = MvAE(view_shape=view_shape, hdim=hdim)

        Input = [ae.input for ae in self.AEs]
        Output = [ae.output for ae in self.AEs]
        Input_e = [ec.input for ec in self.encoders]
        Output_e = [ec.output for ec in self.encoders]

        self.autoencoder = Model(inputs=Input, outputs=Output, name='ae')  # xin _ xout: pretrain
        self.encoder = Model(inputs=Input_e, outputs=Output_e, name='encoder')  # xin _ q

        model_output = []
        # model_aux_output = []
        for v, (oae, oe) in enumerate(zip(Output, Output_e)):
            # model_aux_output.append(c(oe))
            # model_aux_output.append(oe)
            clustering_layer = ClusteringLayer(self.n_clusters, name='clustering' + str(v + 1))
            model_output.append(clustering_layer(oe))
            if ae:
                model_output.append(oae)

        self.model = Model(inputs=Input, outputs=model_output, name='MvDCN')  # xin _ q _ xoutK

    def pretrain(self, x, optimizer='adam', epochs=200, batch_size=256,
                 save_dir=None, verbose=0):
        print('Begin pretraining: ', '-' * 60)
        multi_loss = ['mse'] * len(x)
        self.autoencoder.compile(optimizer=optimizer, loss=multi_loss)
        # csv_logger = callbacks.CSVLogger(save_dir + '/T_pretrain_ae_log.csv')
        # cb = [csv_logger]
        # save = '/ae_weights.h5'

        # begin pretraining
        self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs, verbose=verbose)
        if save_dir:
            self.autoencoder.save_weights(save_dir)
        print('Pretrained weights are saved to ')
        # self.pretrained = True
        print('End pretraining: ', '-' * 60)

    def load_weights(self, weights):
        # load weights of models
        self.model.load_weights(weights)
        # self.modelu.load_weights(weights[1])

    def compile(self, loss, loss_weight, optimizer='sgd'):
        n_loss = len(loss)
        if n_loss < len(self.model.outputs):
            assert n_loss * self.n_views == len(self.model.outputs)
            self.n_loss = n_loss
            loss = loss * self.n_views
            loss_weight = loss_weight * self.n_views if loss_weight else None
        else:
            assert n_loss == len(self.model.outputs)
            self.n_loss = n_loss / self.n_views

        self.model.compile(optimizer=optimizer,
                           loss=loss, loss_weights=loss_weight)

    def save_model(self, save_dir):
        print('Saving model to:', save_dir)
        self.model.save_weights(save_dir)

    def predict_v1(self, X, confs=True, aligned=False):
        # predict cluster labels using the output of clustering layer
        if not isinstance(X, dict):
            X = {'input' + str(v + 1): pv for v, pv in enumerate(X)}

        features = self.encoder.predict(X)
        y_soft = []
        for v, f in enumerate(features):
            y_soft.append(self.model.get_layer(name='clustering' + str(v + 1))(f).numpy())
        y_pred = [ys.argmax(1) for ys in y_soft]

        if aligned:
            confs = self.calculate_label_conf_entropy(y_soft, self.n_clusters) if confs else None
            y_soft_ = [np.diag(cf) @ ys for cf, ys in zip(confs, y_soft)] if confs else y_soft
            y_mean_pred = np.mean(np.array(y_soft_), axis=0).argmax(1)
            return features, y_pred, y_soft, y_mean_pred
        else:
            return features, y_pred, y_soft

    def initialize_model(self, Xp, model_path=None):
        if model_path:
            return self

        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)
        features_p = self.encoder.predict(Xp)
        preds_ini = []
        for v, fp in enumerate(features_p):
            preds_ini.append(kmeans.fit_predict(fp))
            self.model.get_layer(
                name='clustering' + str(v + 1)
            ).set_weights([
                kmeans.cluster_centers_
            ])

        return features_p, preds_ini

    def gen_target_distribution(self, features, y_pred_last=None, normalized=True):
        if isinstance(y_pred_last, list):
            assert len(y_pred_last) == len(features)
            aligned = False
        else:
            aligned = True

        min_max_scaler = preprocessing.MinMaxScaler()
        # k-means on global features: max_iter; init
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=100)

        # update weights
        Cv = [self.model.get_layer(
            name='clustering' + str(v + 1)
            ).get_weights()[0].copy()
            for v in range(len(features))]
        weights = np.array([min_max_scaler.fit_transform(cv).var() for cv in Cv])
        weights = 1 + np.log2(1 + np.array(weights) / np.sum(weights))

        n_features = []
        if normalized:
            for (f, w) in zip(features, weights):
                n_features.append(f * w)
        else:
            for (f, w) in zip(features, weights):
                n_features.append(min_max_scaler.fit_transform(f) * w)

        if aligned:
            # aligned data
            # k-means on global features
            Z = np.hstack(n_features)
            y_pred = kmeans.fit_predict(Z)
            # y_target:
            # define specific class(class index: integer number) for each cluster during the whole training and test
            # map y_pred to y_pred_last
            if y_pred_last is None:
                y_target = y_pred.copy()
                Centers = kmeans.cluster_centers_
                P = self.calculateT(Z, Centers)
                P = self.sharpening(P)
            else:
                y_target, row_ind, col_ind, matrix = self.match(y_pred_last, y_pred)
                # y_target, row_ind, col_ind, matrix = self.match(y_pred, y_pred_last)
                Centers = kmeans.cluster_centers_
                P = self.calculateT(Z, Centers)
                P = self.sharpening(P)
                P = np.dot(P, matrix)
            return y_target, P, weights
        else:
            Ps = []
            y_preds = []
            for v, (y_true, nf) in enumerate(zip(y_pred_last, n_features)):
                y_pred = kmeans.fit_predict(nf)
                yv, row_ind, col_ind, matrix = self.match(y_true, y_pred)
                y_preds.append(yv)
                Centers = kmeans.cluster_centers_
                P = self.calculateT(nf, Centers)
                print('View{} '.format(v + 1), end='-')
                P = self.sharpening(P)
                Ps.append(np.dot(P, matrix))

            return y_preds, Ps, weights

    def fit_pi_v2(self, loader,
                  update_interval_epochs, upmin, upmax, upstop,
                  consensus_threshold, num_neighbors, gamma,
                  graph_mode="default",
                  save_dir='./results/tmp',
                  aux=True,
                  wa=True,
                  tensorboard=False,
                  metrics=('ACC', 'NMI', 'ARI'),
                  update_interval_batches=None):

        print('Begin clustering:', '-' * 60)
        assert update_interval_epochs is not None or update_interval_batches is not None
        if update_interval_epochs is None:
            update_interval_epochs = int(update_interval_batches / loader.batch_size)
            assert update_interval_epochs > 0
        print('Update interval (epochs):', update_interval_epochs)

        # global metrics
        eva_matrix = MetricHandler(metrics, self.n_views, upmin, upmax, upstop, consensus_threshold, tensorboard)

        # Step 1: initialize cluster centers using k-means & generate target distribution
        # --------------------------------------------
        # --------------------------------------------
        execution_time = 0
        t_start = t_stop = time()

        print('ini target distribution for aligned data:')
        features_p, preds_p = self.initialize_model(loader.input_p)
        raw_pred_p, self.targetP, view_weights = self.gen_target_distribution(features_p, normalized=loader.normalized)
        conf_p = self.calculate_label_conf_entropy(self.targetP, self.n_clusters)
        print('Start-view weights: {} ; std(*100): {:.5f} '.format(view_weights, 100 * np.std(view_weights)))

        # ini target distribution for unaligned data
        features_u, preds_u, preds_u_soft = self.predict_v1(loader.input_u)
        targetU = self.LP(
            self.targetP, features_p, features_u,
            beta=num_neighbors, gamma=gamma, conf_labeled=conf_p, mode=graph_mode)
        conf_u = self.calculate_label_conf_entropy(targetU, self.n_clusters)
        execution_time += (time() - t_stop)

        # Step 2: deep clustering
        # --------------------------------------------
        # --------------------------------------------
        with eva_matrix.initialize(
                (loader.Yp, self.targetP.argmax(1), preds_p),
                (loader.Yu, [kl.argmax(1) for kl in targetU], preds_u),
                self.model
        ) as update_pulse:
            t_stop = time()
            sum_epochs = 0

            while update_pulse.status():
                # update P & fine-turing with unaligned data
                print('\n')
                print('-------------------------------------')
                print('{}-th fine-tuning'.format(int(update_pulse.update_id)))

                if not aux or update_pulse.update_id < 2:
                    assert self.targetU is None
                    print('no aux')
                    raw_pred_p_last = raw_pred_p
                    current_train_epochs = update_interval_epochs
                else:
                    print('aux')
                    self.targetU = targetU
                    raw_pred_p_last = raw_pred_p
                    current_train_epochs = update_interval_epochs
                print('current data size: {}'.format(len(self.targetP) + (len(self.targetU[0]) if self.targetU else 0)))
                print('current training epochs: {}'.format(current_train_epochs))
                print('mode: {}'.format(loader.mode))

                # training
                # ============================
                # ============================
                for epoch in range(current_train_epochs):
                    self.model.reset_metrics()
                    generator = loader.gen_batch(P=self.targetP, Pu=self.targetU,
                                                 confp=conf_p if wa else None, confu=conf_u if wa else None)
                    log_dict = None
                    for x_batch, y_batch, sw in generator:
                        sample_weight = [] if sw else None
                        if sample_weight is not None:
                            for sw_ in sw:
                                sample_weight.extend([sw_, np.ones(len(x_batch[0]))])
                        if self.n_loss == 1:
                            y_batch = y_batch[0::2]
                            sample_weight = sample_weight[0::2]
                        log_dict = self.model.train_on_batch(
                            x_batch, y_batch, sample_weight, reset_metrics=False, return_dict=True)
                    # if tensorboard:
                    #     callbacks_list.on_epoch_end(sum_epochs + epoch, log_dict)
                    update_pulse.log(sum_epochs + epoch, log_dict)

                features_p, preds_p, _, preds_p_mean = self.predict_v1(loader.input_p, aligned=True)
                features_u, preds_u, preds_u_soft = self.predict_v1(loader.input_u)

                # update target distribution
                # ============================
                # ============================
                # aligned data
                raw_pred_p, self.targetP, view_weights = self.gen_target_distribution(
                    features_p, normalized=loader.normalized)
                conf_p = self.calculate_label_conf_entropy(self.targetP, self.n_clusters)

                # unaligned data
                # mix up: generate target distribution for the unaligned data with mix up
                # update target distribution of the unaligned data after fetching target distribution of aligned data
                targetU = self.LP(
                    self.targetP, features_p, features_u,
                    beta=num_neighbors, gamma=gamma, conf_labeled=conf_p, mode=graph_mode)
                conf_u = self.calculate_label_conf_entropy(targetU, self.n_clusters)

                sum_epochs += current_train_epochs
                execution_time += (time() - t_stop)

                # evaluation
                # ============================
                # ============================
                # record metrics
                print('\n')
                print('== statistics: ')
                label_statistics(self.targetP.argmax(1), conf_p, loader.Yp[0])

                print('\n')
                print('== Performance on aligned data: ')

                print('view weights:{}; std*100:{:.5f}'.format(view_weights, 100 * np.std(view_weights)))
                update_pulse.update_paired(loader.Yp, self.targetP.argmax(1), preds_p, preds_p_mean)

                print('\n')
                print('== Performance on unaligned data: ')
                update_pulse.update_unpaired_target(loader.Yu, [kl.argmax(1) for kl in targetU])
                update_pulse.update_unpaired(loader.Yu, preds_u)

                print('\n')
                if update_pulse.hit_global():
                    if loader.mode == 'pa':
                        label_views = [np.concatenate([yp, yu]) for yp, yu in zip(loader.Yp, loader.Yu)]
                        pred_views = [np.concatenate([raw_pred_p_last, pdu]) for pdu in preds_u]
                        update_pulse.update_global(
                            label_views, pred_views, prefix='************view-leval************\n')

                    elif loader.mode == 'fa':
                        preds_u_mean = np.zeros(loader.size - len(preds_p_mean))
                        assert len(loader.mask.T) == len(preds_u_mean)
                        pointers = np.array([0] * len(preds_u), dtype=int)
                        for i, mr in enumerate(loader.mask.T):
                            tmp_pred = np.zeros_like(preds_u_soft[0][0])
                            for imr, pt, pdu in zip(mr, pointers, preds_u_soft):
                                tmp_pred += (pdu[pt] if imr else np.zeros_like(tmp_pred))
                            pointers += mr
                            preds_u_mean[i] = tmp_pred.argmax()
                        label_views = [loader.label[0].copy()] * len(preds_u_soft)
                        pred_views = [np.concatenate([raw_pred_p_last, preds_u_mean])] * len(preds_u_soft)
                        update_pulse.update_global(
                            label_views, pred_views, prefix='************performance************\n')

                t_stop = time()
            else:
                execution_time += (time() - t_stop)

                print('\n')
                # check
                print(f'aligned ratio history: {update_pulse.aligned_ratio}')
                print(f'Clustering time: {execution_time} s')
                print('Save interval (epochs)', int((update_pulse.update_id - 1) * update_interval_epochs))

                performance = update_pulse.performance(execution_time=execution_time)

        # save the trained model
        # only save the final model
        self.save_model(save_dir + '/model_final.h5')
        print('model is saved to {}'.format(save_dir + '/model_final.h5'))
        print('End clustering:', '-' * 60)
        return performance

    @staticmethod
    def LP(Y, features_p, features_u, beta=10, gamma=0.01, zeta=1, conf_labeled=None, mode="bipartite"):
        """"label propagation
        :param Y: consensus semantic label, ndarray, (n_paired, n_cluster)
        :param features_p: hidden representation of paired data, list, [ndarray, (n_paired, hdim)]*n_views
        :param features_u: hidden representation of paired data, list, [ndarray, (n_unpaired, hdim)]*n_views
        :param beta:
        :param gamma: the number of neighbours, int
        :param zeta:
        :param conf_labeled: cof for consensus semantic labels
        :param mode: str,
            "default", adaptive graph learning
            "bg", bipartite graph
            "gKfg", Gaussian kernel-based fully connected graph
        """

        if mode in ["static", "s"]:
            knn_label = []
            m = len(features_p[0])
            n = [len(fu) for fu in features_u]
            features = [np.concatenate([fp, fu], axis=0) for fp, fu in zip(features_p, features_u)]
            conf_labeled = np.diag(np.ones(m)) if conf_labeled is None else np.diag(conf_labeled)
            for f, nv in zip(features, n):
                S = rbf_kernel(f, f)
                # Smn = S[:m, m:]
                # Snm == Smn.T
                Snm = S[m:, :m]
                Snn = S[m:, m:]
                knn_label_ = np.linalg.inv(np.diag(Snm.sum(1))+np.diag(Snn.sum(1))-Snn) @ Snm @ conf_labeled @ Y
                knn_label.append(knn_label_ / np.sum(knn_label_, axis=1, keepdims=True))
            return knn_label

        MvDnm = [np.sum(np.square(np.expand_dims(fu, 1) - fp), 2) for fu, fp in zip(features_u, features_p)]

        if mode in ['bipartite', 'b']:
            m = len(features_p[0])
            conf_labeled = np.diag(np.ones(m)) if conf_labeled is None else np.diag(conf_labeled)
            knn_label = [InitializeSIGs(Dnm, beta) @ conf_labeled @ Y for Dnm in MvDnm]
            for v, knn in enumerate(knn_label):
                knn_label[v] = knn / np.sum(knn, axis=1, keepdims=True)
            return knn_label
        else:  # 'default' or 'd'
            MvDnn = [np.sum(np.square(np.expand_dims(fu, 1) - fu), 2) for fu in features_u]
            return [adaptive_lp(Y, Dnm, Dnn, beta=beta, gamma=gamma, zeta=zeta, max_iters=2)[0]
                    for Dnm, Dnn in zip(MvDnm, MvDnn)]

    @staticmethod
    def match(y_true, y_pred, display_matrix=False):
        # y_pred: y_pred_global; y_true: y_pred
        # y_pred to y_true
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for ypd, yt in zip(y_pred, y_true):
            w[ypd, yt] += 1
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(w, maximize=True)

        matrix = np.zeros_like(w)
        matrix[row_ind, col_ind] = 1
        if display_matrix:
            print(matrix)

        return col_ind[y_pred].copy().astype(np.int64), row_ind, col_ind, matrix

    @staticmethod
    def calculateT(inputs, centers):
        alpha = 1
        t = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
        t **= (alpha + 1.0) / 2.0
        t = np.transpose(np.transpose(t) / np.sum(t, axis=1))
        return t

    @staticmethod
    def sharpening(Q, t=2, frequency_display=True):
        if not isinstance(Q, list):
            Q = [Q]
        if frequency_display:
            for q in Q:
                print('frequency of class before enhancement:\n {}'.format(q.sum(0).astype(np.int64)))

        weights = []
        for q in Q:
            # tmp = q ** t
            tmp = q ** t / np.sum(q, axis=0, keepdims=True)
            weights.append((tmp.T / tmp.sum(1)).T)

        if len(weights) == 1:
            return weights[0]
        else:
            return weights

    @staticmethod
    def calculate_label_conf_entropy(labels, n_classes):
        # entropy based
        if not isinstance(labels, list):
            labels = [labels]
        confs = []
        for lb in labels:
            entropy = sc.entropy(lb.T)
            weights = 1 - entropy / np.log(n_classes)
            weights = weights / np.max(weights)
            confs.append(weights)
        return confs[0] if len(confs) == 1 else confs

    @staticmethod
    def calculate_label_conf_cross_entropy(labels, n_classes):
        # cross entropy based
        if not isinstance(labels, list):
            labels = [labels]

        confs = []
        size = len(labels[0])
        for lb in labels:
            gt_index = lb.argmax(1)
            entropy = - np.log(lb[np.arange(size), gt_index])
            weights = 1 - entropy / np.log(n_classes)
            weights = weights / np.max(weights)
            confs.append(weights)
        return confs[0] if len(confs) == 1 else confs

    @staticmethod
    def eva_by_conf_thresholds(conf_global, thresholds, Y, preds, global_pred):
        if not isinstance(thresholds, (tuple, list, np.ndarray)):
            thresholds = [thresholds]

        if isinstance(Y, list):
            assert len(Y) == len(preds)
            Y = Y[0]

        # index_thresholds = []
        print('=============================')
        print('=============================')
        for t in thresholds:
            if t == 0 or t == 1:
                continue
            ind_t = np.where(conf_global > t)
            if len(ind_t[0]) == 0:
                continue
            print('number of samples with conf thresholds > {:.2f} : {}'.format(t, len(ind_t[0])))
            evaBymetrics(Y[ind_t].astype(np.int64), global_pred[ind_t].astype(np.int64),
                         prefix='performance on global, conf_thresholds = {:.2f} :'.format(t))
            for v, prd in enumerate(preds):
                evaBymetrics(Y[ind_t].astype(np.int64), prd[ind_t].astype(np.int64),
                             prefix='performance on view-{}, conf_thresholds = {:.2f} :'.format(v + 1, t))
            print('\n')
