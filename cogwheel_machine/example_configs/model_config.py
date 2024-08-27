"""
Example configuration file with parameters for the neural posterior
estimation.

This file may be copied into `modeldir` and edited before training the
network.
"""
# kwargs to sbi.utils.posterior_nn
POSTERIOR_NN_KWARGS = {'model': 'nsf',
                       'hidden_features': 64}

# Embedding network
EMBEDDING_LAYER_SIZES = None  # list of ints (optional)
PARAMS_REFWF_SIZE = 10

# kwargs to sbi.inference.SNPE.train
TRAIN_KWARGS = {'training_batch_size': 65536,
                'stop_after_epochs': (20, 0.2),
                'max_num_epochs': 10000,
                'learning_rate': 1e-3,
                'show_train_summary': True}

MAX_TRAINING_EXAMPLES = None  # int
DEVICE = 'cuda'
