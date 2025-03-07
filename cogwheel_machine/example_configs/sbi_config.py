"""
Example configuration file with parameters for the neural posterior
estimation.

This file may be copied into ``sbidir`` and edited before training the
network.
"""
# kwargs to sbi.utils.posterior_nn
POSTERIOR_NN_KWARGS = {'model': 'nsf',
                       'hidden_features': 64}

# Embedding network
EMBEDDING_LAYER_SIZES = None  # list of ints (optional)

# kwargs to sbi.inference.SNPE.train
TRAIN_KWARGS = {'training_batch_size': 65536,
                'stop_after_epochs': 32,
                'max_num_epochs': 10000,
                'learning_rate': 1e-3,
                'show_train_summary': True}

MAX_TRAINING_EXAMPLES = None  # int

DEVICE = None  # ``None`` will try to use 'cuda' or fall back to 'cpu'.
