import json
import argparse
import numpy as np
from pipeline import train
from dataset_config import data_config
from data.read_mat import config


run_times = 3
# data = 'Caltech101_20'
# data = 'BDGP'
data = 'Handwritten'
# data = 'Scene15'
# data = 'YouTube_X'
# data = 'ALOI100'
# data = 'ThreeSources'
# data = 'MNIST_UPS'
# data = 'Fashion_MV'
# data = 'Reuters'
# data = 'Noisy-MNIST'

paired_bar = [.4]
miss_bar = [.5]
# [0., .3, .5, .7, 1.]
# [0.7, 1.0, 0.3, 0.5]
base_seed = 17
train_ae = False
test = False
tensorboard = True
aux = True
wa = True
ae = True

CONFIG_DEFAULT = {
    'save_root': './results',
    # pretraining
    'pretrain_verbose': 1,
    'pretrain_epochs': 500,
    'hdim': 10,
    'normalized': True,
    # training
    'batch_size': 256,
    'upstop': 3,
    'upmax': 20,
    'upmin': 10,
    'consensus_threshold': 0.9,
    'update_interval_epochs': 100,
    # hyperparams:
    # categorical_crossentropy : lc
    # mse : lm
    # optimizer learning rate: lr
    'lc': 0.1,
    'Idec': 1.0,
    'lr': 0.001,
    'num_neighbors': 100,
    'gamma': 0.01,
    'base_seed': base_seed,
    'LP': aux,
    'wa': wa,
    'rec': ae
}


def parse_args(**kwargs):
    parser = argparse.ArgumentParser(description='dissld_main')
    parser.add_argument('-d', '--dataset', default=kwargs['dataset'],
                        help="Dataset name to train")
    parser.add_argument('-sr', '--save_root', default=kwargs['save_root'],
                        help="Root dir to save the results")
    parser.add_argument('-sd', '--save_secondary_dir', default=kwargs['save_secondary_dir'],
                        help="Secondary dir to save the results")
    parser.add_argument('-pr', '--pairedrate', default=kwargs['pairedrate'], type=float,
                        help="Paired rate")
    parser.add_argument('-mr', '--missrate', default=kwargs['missrate'], type=list,
                        help="Miss rate")

    # Parameters for pretraining
    parser.add_argument('--pretrain', default=kwargs['pretrain'], type=bool,
                        help="Pretrain the autoencoder?")
    parser.add_argument('--pretrain_dir', default=kwargs['pretrain_dir'], type=str,
                        help="Pretrained weights of the autoencoder")
    parser.add_argument('-aev', '--pretrain_verbose', default=kwargs['pretrain_verbose'], type=int,
                        help="Verbose for pretraining")
    parser.add_argument('--hdim', default=kwargs['hdim'], type=int,
                        help="dimension of the hidden layer")
    parser.add_argument('--pretrain_epochs', default=kwargs['pretrain_epochs'], type=int,
                        help="Number of epochs for pretraining")

    # Parameters for clustering: testing
    parser.add_argument('--test', default=kwargs['test'], type=bool,
                        help="Testing the clustering performance with provided weights")
    parser.add_argument('--test_weights', default=kwargs['test_weights'], type=str,
                        help="Model weights, used for testing")

    # Parameters for clustering: training
    parser.add_argument('--normalized', default=kwargs['normalized'], type=bool,
                        help="normalized dataset?")
    parser.add_argument('-upe', '--update_interval_epochs', default=kwargs['update_interval_epochs'], type=int,
                        help="Number of epochs for training with aligned data.")
    # parser.add_argument('-upb', '--update_interval_batchs', default=kwargs['update_interval_batchs'], type=int,
    #                     help="Number of batchs for training with aligned data.")
    parser.add_argument('--consensus_threshold', default=kwargs['consensus_threshold'], type=float,
                        help="stop condition for consensus ratio.")
    parser.add_argument('--upstop', default=kwargs['upstop'], type=int,
                        help="Number of updates while consensus ratio resching threshold.")
    parser.add_argument('--upmax', default=kwargs['upmax'], type=int,
                        help="Maximum number of updates for target distribution.")
    parser.add_argument('--upmin', default=kwargs['upmin'], type=int,
                        help="Minimum number of updates for target distribution.")

    parser.add_argument('--batch_size', default=kwargs['batch_size'], type=int,
                        help="Batch size")
    parser.add_argument('--num_neighbors', default=kwargs['num_neighbors'], type=int,
                        help="num of neighbors while calculating similarity between samples")
    parser.add_argument('--gamma', default=kwargs['gamma'], type=int,
                        help="hyer-parameters for LP")
    parser.add_argument('--Idec', default=kwargs['Idec'], type=float,
                        help="weight of AEs?")
    parser.add_argument('--lc', default=kwargs['lc'], type=float,
                        help="weight of clustering")
    parser.add_argument('--lr', default=kwargs['lr'], type=float,
                        help="learning rate during clustering")

    return parser.parse_args()


def check_update(data_name):
    config_ = data_config[data_name]
    for key in config_.keys():
        assert key in CONFIG_DEFAULT.keys()
        CONFIG_DEFAULT[key] = config_[key]


def nest_tolist(obj):
    # assert isinstance(obj, (list, tuple, np.ndarray))
    if isinstance(obj, (list, tuple)):
        container = []
        for item in obj:
            container.append(nest_tolist(item))
        return container
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


if __name__ == '__main__':
    check_update(data)
    item_list = [
        data,
        str(len(config[data]))+"v",
        str(CONFIG_DEFAULT['lc'])+"lc",
        str(int(CONFIG_DEFAULT['num_neighbors'])) + "nb",
        str(CONFIG_DEFAULT['gamma']) + "gm",
        str(CONFIG_DEFAULT['update_interval_epochs'])+"eps",
    ]
    if aux:
        item_list.append('aux')

    exp_name = '_'.join(item_list)
    save_root = '/'.join([CONFIG_DEFAULT['save_root'], data])
    save_secondary_dir = '/'.join([save_root, exp_name])

    # settings
    if train_ae:
        load_ae = None
    else:
        load_ae = '/'.join([save_root, 'ae_weights.h5'])

    if test:
        load_test = '/'.join([save_secondary_dir, 'model_final.h5'])
    else:
        load_test = None

    CONFIG_DEFAULT.update({
        'dataset': data,
        'pretrain': train_ae,
        'pretrain_dir': load_ae,
        'test': test,
        'test_weights': load_test,
        'save_root': save_root,
        'save_secondary_dir': save_secondary_dir
    })

    args = None
    for _, pr in enumerate(paired_bar):
        for _, mr in enumerate(miss_bar):
            if args is not None:
                cfg = vars(args)
                cfg.update({
                    'pairedrate': pr,
                    'missrate': mr,
                    'pretrain_dir': '/'.join([
                        save_root,
                        'ae_weights_{}{}.h5'.format(str(int(pr * 100)), str(int(mr * 100)) if mr != 0 else '00')
                    ])
                })
            else:
                CONFIG_DEFAULT.update({
                    'pairedrate': pr,
                    'missrate': mr,
                    'pretrain_dir': '/'.join([
                        save_root,
                        'ae_weights_{}{}.h5'.format(str(int(pr * 100)), str(int(mr * 100)) if mr != 0 else '00')
                    ])
                })
                cfg = CONFIG_DEFAULT
            args = parse_args(**cfg)

            print('++++++++++++++++++++++++++++++parameters config++++++++++++++++++++++++++++++++')
            print(f'++++++++++++++++++paired rate = {pr}, miss rate = {mr}++++++++++++++++++++++++')
            print(args)

            if args.test:
                continue
            else:
                performance = [cfg]
                for ri in range(run_times):
                    print('\n')
                    print('-----------------------------------------------------------------------')
                    print(f'run times {ri + 1}--------------------------------------------------\n')

                    tmp_performance = train(
                        args, ri + base_seed, tensorboard=tensorboard, aux=aux, wa=wa, ae=ae)
                    for k, v in tmp_performance.items():
                        if isinstance(v, (list, tuple, np.ndarray)):
                            tmp_performance.update({
                                k: nest_tolist(tmp_performance[k])
                            })
                        else:
                            print('detected unexpected type: {}'.format(type(v)))
                    performance.append(tmp_performance)

                # saved as .json file
                json_path = save_secondary_dir + '/performance_{}{}.json'.format(
                        str(int(pr * 100)), str(int(mr * 100)) if mr != 0 else '00')
                with open(json_path, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(performance, indent=2, ensure_ascii=False))
                print('results has been saved to {}'.format(json_path))
