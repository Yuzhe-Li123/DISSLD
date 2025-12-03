from utils.contruct_datasets import construct_pvp2dip, construct_pip, padding_with_average, construct_cmp2dip
from models.DICSLD import MvDCN
from data.load_data_v2 import DataLoader

import sys
import os
from scipy.io import savemat
import numpy as np
from data.load_data import get_sn
from data.read_mat import Caltech_2v, BDGP, Scene15, YouTube_X, ALOI100, Handwritten, Reuters
from tensorflow.keras.optimizers import SGD, Adam

datamap = {
    'Caltech101_20': Caltech_2v,
    'BDGP': BDGP,
    'Scene15': Scene15,
    'ALOI100': ALOI100,
    'Handwritten': Handwritten,
    'Reuters': Reuters
}

path = sys.path[0]


def gen_cmp2dip(alg, pairedrate, missrate, pretrained_dir=False, mark_x='X', mark_y='Y'):
    for data_name in datamap.keys():
        loader = DataLoader(batch_size=256,
                            normalized=False,
                            pairedrate=pairedrate,
                            missrate=0,
                            dataset=data_name)
        model = MvDCN(view_shape=loader.view_shapes,
                      n_clusters=loader.n_classes, hdim=10)
        if pretrained_dir:
            model.autoencoder.load_weights(pretrained_dir)
        else:
            model.pretrain(loader.pretraining_data,
                           optimizer=Adam(lr=0.001),
                           epochs=500, batch_size=256, verbose=1)

        X, Y = construct_cmp2dip(model, loader, 0.5)
        data = {
            mark_x: X,
            mark_y: Y
        }
        from scipy.io import savemat
        # savemat('./datasets_pvc_mvc/'+args.dataset+'.mat', data)
        savemat(path + '/datasets_cmp2dip/{}/{}.mat'.format(alg, data_name), data)


def gen_pip(alg_name, miss_bar, mark_x='X', mark_y='Y'):
    save_root = '{}/datasets_pip/{}/'.format(path, alg_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if alg_name in ['PMVC']:
        mode = 1
        dim_first = False
    else:
        mode = 2
        dim_first = True
    for data_name, loader in datamap.items():
        data, label = loader()
        construct_pip(data_name, save_root,
                      miss_bar, data, label, mode=mode, dim_first=dim_first, mark_x=mark_x, mark_y=mark_y)


def gen_cmp2pip(alg, missbar):
    save_root = '{}/datasets_cmp2pip/{}/'.format(path, alg)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for data_name, loader in datamap.items():
        data, label = loader()
        padding_with_average(data_name, save_root, missbar, data, label)


def gen_pvp2pip(alg, pairedbar, missbar):
    save_root = path + '/datasets_pvp2dip/{}/'.format(alg)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for p in pairedbar:
        for m in missbar:
            second_root = save_root + '/paired{}miss{}/'.format(str(int(p * 100)), str(int(m * 100)))
            if not os.path.exists(second_root):
                os.makedirs(second_root)
            for data_name, loader in datamap.items():
                data, label = loader()
                construct_pvp2dip(data_name, second_root, data, label, p, m)


def gen_mvcln4car(pairedbar, missbar):
    datamap_ = {
        'Handwritten': (Handwritten, 6),
        'Reuters': (Reuters, 5)
    }
    save_root = path + '/datasets_pvp2dip/MvCLN/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for p in pairedbar:
        for m in missbar:
            second_root = save_root + f'/paired{str(int(p * 100))}_miss{str(int(m * 100))}/'
            if not os.path.exists(second_root):
                os.makedirs(second_root)
            for data_name, (loader, num_views) in datamap_.items():
                for i in range(1, num_views):
                    data, label = loader(filter_=(0, i))
                    construct_pvp2dip(f'{data_name}_0{i}', second_root, data, label, p, m)


def gen_pvp_upmgc_sm():
    save_dir = path + '/datasets_pvp/UPMGC_SM/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data_name, loader in datamap.items():
        data, label = loader()
        data = [d.T.astype(np.float64) for d in data]
        cell = np.empty((len(data), ), dtype=np.ndarray)
        for i in range(len(data)):
            cell[i] = data[i]
        cell.reshape(1, -1)
        data_dict = {'X': cell, 'Y': label[0].reshape(-1, 1)}
        savemat(save_dir + '/{}.mat'.format(data_name), data_dict)


if __name__ == '__main__':
    print(path)

    # gen_pvp2pip('PVC', [0.5], [0.5])
    # gen_cmp2pip('FMVACC', [0.5])
    gen_mvcln4car([0.1, 0.3, 0.5, 0.7], [0.5])



