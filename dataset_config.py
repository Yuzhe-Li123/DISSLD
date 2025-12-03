data_config = {
    # 0.1
    'BDGP': {
        'consensus_threshold': 0.95,
        'upstop': 3,
        'upmax': 10,
        'upmin': 0,
        'update_interval_epochs': 20,
        # 'update_interval_batches': 1000,
        'lc': 1.0,
        'Idec': 1.0,
        'num_neighbors': 30,
        'normalized': False
    },
    # 1.0
    'Caltech101_20': {
        'consensus_threshold': 0.95,
        'upstop': 3,
        'upmax': 15,
        'upmin': 10,
        'update_interval_epochs': 110,
        # 'update_interval_epochs': 110,
        # 'update_interval_batches': 1000,
        'lc': 1.0,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': False
    },
    # 0.75
    'Scene15': {
        'consensus_threshold': 0.9,
        'upstop': 3,
        'upmax': 10,
        'upmin': 0,
        'update_interval_epochs': 70,
        # 'update_interval_batches': 1000,
        'lc': 0.7,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': True
    },
    'Reuters': {
        'consensus_threshold': 0.75,
        'upstop': 3,
        'upmax': 5,
        'upmin': 0,
        'update_interval_epochs': 20,
        # 'update_interval_batches': 1000,
        'lc': 1.0,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': True
    },
    # 0.1
    'ALOI100': {
        'consensus_threshold': 0.8,
        'upstop': 3,
        'upmax': 10,
        'upmin': 0,
        'update_interval_epochs': 55,
        # 'update_interval_batches': 1000,
        'lc': 1.0,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': True
    },
    # 0.1
    'YouTube_X': {
        'consensus_threshold': 0.95,
        'upstop': 3,
        'upmax': 10,
        'upmin': 5,
        'update_interval_epochs': 50,
        # 'update_interval_batches': 1000,
        'lc': 0.1,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': True
    },
    'Handwritten': {
        'consensus_threshold': 0.7,
        'upstop': 3,
        'upmax': 10,
        'upmin': 5,
        'update_interval_epochs': 50,
        'lc': 1.0,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': True
    },
    'MNIST_UPS': {
        'consensus_threshold': 0.9,
        'upstop': 3,
        'upmax': 10,
        'upmin': 0,
        'update_interval_epochs': 50,
        # 'update_interval_batches': 1000,
        'lc': 0.1,
        'Idec': 1.0,
        'num_neighbors': 10,
        'normalized': None
    }
}


