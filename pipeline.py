from tensorflow.keras.optimizers import SGD, Adam
from models.dmc_ssd import MvDCN
from data.load_data_v2 import DataLoader
import tensorflow as tf


import os
from time import time
import numpy as np
import random


def train(args, seed, tensorboard, aux=True, ae=True, wa=True):
    # set seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # get data and model
    loader = DataLoader(batch_size=args.batch_size,
                        normalized=args.normalized,
                        pairedrate=args.pairedrate,
                        missrate=args.missrate,
                        dataset=args.dataset)

    model = MvDCN(view_shape=loader.view_shapes,
                  n_clusters=loader.n_classes,
                  hdim=args.hdim, ae=ae)
    model.compile(loss=['categorical_crossentropy', 'mse'] if ae else ['categorical_crossentropy'],
                  loss_weight=[args.lc, args.Idec] if ae else None,
                  optimizer=Adam(lr=args.lr))
    model.model.summary()

    # pretraining
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # AEs
    if args.pretrain is False and os.path.exists(args.pretrain_dir):
        # load pretrained weights
        model.autoencoder.load_weights(args.pretrain_dir)
        # model.load_weights(args.pretrain_dir)
    else:
        # train AE
        t_start = time()
        model.pretrain(loader.pretraining_data, optimizer=Adam(lr=args.lr), epochs=args.pretrain_epochs,
                       batch_size=args.batch_size, save_dir=args.pretrain_dir, verbose=args.pretrain_verbose)
        args.pretrain_dir = args.pretrain_dir
        print("Time for pretraining: %ds" % (time() - t_start))

    # start clustering
    if not os.path.exists(args.save_secondary_dir):
        os.makedirs(args.save_secondary_dir)

    tmp_performance = model.fit_pi_v2(
        loader,
        update_interval_epochs=args.update_interval_epochs,
        upmin=args.upmin, upmax=args.upmax, upstop=args.upstop, wa=wa,
        consensus_threshold=args.consensus_threshold,
        num_neighbors=args.num_neighbors,
        gamma=args.gamma,
        save_dir=args.save_secondary_dir,
        aux=aux,
        tensorboard=tensorboard
    )

    return tmp_performance
